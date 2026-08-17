#!/usr/bin/env python3
"""Create an IC-LoRA JSONL manifest from paired RoboTwin RGB and white-robot videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/lpai/volumes/mind-eb-ali-sh/dulingyi/sunyilu/RoboTwin2_0_cleaned_full/white")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Optional RoboTwin manifest with task, variant, and episode fields. Defaults to manifest_train_white.jsonl.",
    )
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"Data root does not exist: {data_root}")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be zero or positive.")

    source_manifest = args.source_manifest or data_root / "manifest_train_white.jsonl"
    source_manifest_exists = source_manifest.is_file()
    if source_manifest_exists:
        episode_dirs = []
        for line in source_manifest.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            episode_dirs.append(data_root / item["task"] / item["variant"] / f"episode{item['episode']}")
    else:
        episode_dirs = [path.parent for path in sorted(data_root.rglob("head_color.mp4"))]

    rows: list[dict[str, str]] = []
    for episode_dir in episode_dirs:
        rgb_video = episode_dir / "head_color.mp4"
        reference_video = episode_dir / "white_robot.mp4"
        instruction_file = episode_dir / "instruction.txt"
        if args.verify_files and (not rgb_video.is_file() or not reference_video.is_file() or not instruction_file.is_file()):
            continue
        try:
            caption = instruction_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if not caption:
            continue
        rows.append(
            {
                "video": str(rgb_video),
                "reference_video": str(reference_video),
                "caption": caption,
            }
        )
        if args.max_samples and len(rows) >= args.max_samples:
            break

    if not rows:
        raise ValueError(f"No paired head_color.mp4, white_robot.mp4, instruction.txt samples found below {data_root}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} paired IC-LoRA samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
