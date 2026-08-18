#!/usr/bin/env python3
"""Generate RGB robot videos from an RGB first frame and white-robot motion.

Requires an IC-LoRA trained to interpret the white-robot control video. The
stock LTX-2.5 distilled LoRA is not an IC-LoRA and is intentionally rejected.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path("/lpai/models/Lightricks__LTX-2_5/26-08-11-2005")
DATA_ROOT = Path("/lpai/volumes/mind-eb-ali-sh/dulingyi/sunyilu/RoboTwin2_0_cleaned_full/white")
STOCK_DISTILLED_LORA = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"


def file_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def directory_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def extract_first_frame(video_path: Path, image_path: Path) -> Path:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is unavailable; run scripts/setup_world_model_env.sh first.") from error
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        frame = next(container.decode(video=0), None)
    if frame is None:
        raise ValueError(f"No frame found in RGB video: {video_path}")
    frame.to_image().save(image_path)
    return image_path


def required_model(models_root: Path, relative_path: str) -> Path:
    path = models_root / relative_path
    if not path.is_file():
        raise ValueError(f"Missing LTX-2.5 component: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--episode", type=Path, help="Episode directory, relative to --data-root or absolute.")
    inputs.add_argument("--data-root", type=directory_path, default=DATA_ROOT)
    inputs.add_argument("--first-frame", type=file_path, help="RGB first-frame image.")
    inputs.add_argument("--rgb-video", type=file_path, help="RGB video; its first frame is extracted when needed.")
    inputs.add_argument("--reference-video", type=file_path, help="White robot motion-control video.")
    inputs.add_argument("--prompt", help="Instruction text.")
    inputs.add_argument("--prompt-file", type=file_path, help="UTF-8 instruction text file.")

    models = parser.add_argument_group("models")
    models.add_argument("--models-root", type=directory_path, default=MODEL_ROOT)
    models.add_argument("--ic-lora-path", type=file_path, required=True, help="Trained white-video-to-RGB IC-LoRA.")
    models.add_argument("--lora-strength", type=float, default=1.0)
    models.add_argument("--reference-strength", type=float, default=1.0)
    models.add_argument("--image-strength", type=float, default=1.0)

    generation = parser.add_argument_group("generation")
    generation.add_argument("--output-path", type=Path, help="Output RGB MP4 path.")
    generation.add_argument("--output-dir", type=Path, default=Path("outputs/world_model"))
    generation.add_argument(
        "--height",
        type=int,
        default=480,
        help="Final output height (default: 480). The two-stage pipeline uses a compatible internal size when needed.",
    )
    generation.add_argument(
        "--width",
        type=int,
        default=640,
        help="Final output width (default: 640). The two-stage pipeline uses a compatible internal size when needed.",
    )
    generation.add_argument("--num-frames", type=int, default=121)
    generation.add_argument("--frame-rate", type=float, default=24.0)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--offload", choices=("cpu", "disk"), default="cpu")
    generation.add_argument("--quantization", choices=("none", "fp8-cast"), default="fp8-cast")
    generation.add_argument("--skip-stage-2", action="store_true")
    generation.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, str, Path]:
    episode_name = "sample"
    if args.episode:
        episode = args.episode if args.episode.is_absolute() else args.data_root / args.episode
        episode = episode.resolve()
        rgb_video = episode / "head_color.mp4"
        reference_video = episode / "white_robot.mp4"
        default_prompt_file = episode / "instruction.txt"
        for path in (rgb_video, reference_video, default_prompt_file):
            if not path.is_file():
                raise ValueError(f"Episode is missing required input: {path}")
        first_frame, prompt_file, episode_name = args.first_frame, args.prompt_file or default_prompt_file, episode.name
    else:
        if args.reference_video is None or (args.first_frame is None and args.rgb_video is None):
            raise ValueError("Without --episode, provide --reference-video and --first-frame or --rgb-video.")
        rgb_video, reference_video, first_frame, prompt_file = args.rgb_video, args.reference_video, args.first_frame, args.prompt_file

    prompt = args.prompt.strip() if args.prompt else ""
    if not prompt:
        if prompt_file is None:
            raise ValueError("Provide --prompt or --prompt-file.")
        prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Prompt is empty.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if first_frame is None:
        if rgb_video is None:
            raise ValueError("Cannot extract a first frame without --rgb-video.")
        first_frame = extract_first_frame(rgb_video, output_dir / f"{episode_name}_first_frame.png")
    output_path = args.output_path.resolve() if args.output_path else output_dir / f"{episode_name}_rgb.mp4"
    return first_frame, reference_video, prompt, output_path


def resolve_generation_resolution(output_height: int, output_width: int) -> tuple[int, int]:
    """Choose a 64-aligned internal generation size with the closest aspect ratio."""
    if output_height % 64 == 0 and output_width % 64 == 0:
        return output_height, output_width

    output_aspect_ratio = output_width / output_height
    candidates = [
        (height, width)
        for height in range(64, output_height + 1, 64)
        for width in range(64, output_width + 1, 64)
    ]
    if not candidates:
        raise ValueError("--height and --width must both be at least 64.")

    return min(
        candidates,
        key=lambda size: (
            abs(size[1] / size[0] - output_aspect_ratio),
            abs(size[0] * size[1] - output_height * output_width),
        ),
    )


def resize_video(input_path: Path, output_path: Path, width: int, height: int) -> None:
    """Resize an MP4 to the requested output dimensions without cropping."""
    import av

    with av.open(input_path) as input_container:
        input_stream = next((stream for stream in input_container.streams if stream.type == "video"), None)
        if input_stream is None:
            raise ValueError(f"No video stream found in generated file: {input_path}")

        frame_rate = input_stream.average_rate or 24
        with av.open(output_path, mode="w") as output_container:
            output_stream = output_container.add_stream("libx264", rate=frame_rate)
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = "yuv420p"

            for frame in input_container.decode(input_stream):
                resized_frame = frame.reformat(width=width, height=height, format="yuv420p")
                for packet in output_stream.encode(resized_frame):
                    output_container.mux(packet)
            for packet in output_stream.encode():
                output_container.mux(packet)


def main() -> int:
    args = parse_args()
    if args.ic_lora_path.name == STOCK_DISTILLED_LORA:
        raise ValueError("The stock distilled LoRA is not a white-robot IC-LoRA. Supply a trained IC-LoRA checkpoint.")
    if args.num_frames < 9 or (args.num_frames - 1) % 8:
        raise ValueError("--num-frames must be 8k+1, for example 121 or 241.")
    if args.height < 64 or args.width < 64:
        raise ValueError("--height and --width must both be at least 64.")

    first_frame, reference_video, prompt, output_path = resolve_inputs(args)
    generation_height, generation_width = resolve_generation_resolution(args.height, args.width)
    resize_required = (generation_height, generation_width) != (args.height, args.width)
    native_output_path = (
        output_path.with_name(f"{output_path.stem}_native_{generation_width}x{generation_height}{output_path.suffix}")
        if resize_required
        else output_path
    )
    components = {
        "--transformer-path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "--text-encoder-path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "--video-vae-path": "vae/ltx-2.5-video-vae-bf16.safetensors",
        "--audio-vae-path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "--spatial-upsampler-path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    }
    command = [sys.executable, "-m", "ltx_pipelines.ic_lora"]
    for option, relative_path in components.items():
        command.extend((option, str(required_model(args.models_root, relative_path))))
    command.extend((
        "--lora", str(args.ic_lora_path), str(args.lora_strength),
        "--image", str(first_frame), "0", str(args.image_strength),
        "--video-conditioning", str(reference_video), str(args.reference_strength),
        "--prompt", prompt,
        "--height", str(generation_height), "--width", str(generation_width),
        "--num-frames", str(args.num_frames), "--frame-rate", str(args.frame_rate),
        "--seed", str(args.seed), "--offload", args.offload,
        "--output-path", str(native_output_path),
    ))
    if args.quantization != "none":
        command.extend(("--quantization", args.quantization))
    if args.skip_stage_2:
        command.append("--skip-stage-2")

    print("Executing:\n" + shlex.join(command), flush=True)
    if resize_required:
        print(
            f"Resizing pipeline output {generation_width}x{generation_height} to final "
            f"{args.width}x{args.height}: {output_path}",
            flush=True,
        )
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
        if resize_required:
            resize_video(native_output_path, output_path, args.width, args.height)
            native_output_path.unlink()
        print(f"Generated RGB video: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
