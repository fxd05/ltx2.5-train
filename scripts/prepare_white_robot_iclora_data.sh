#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-/lpai/volumes/mind-eb-ali-sh/dulingyi/sunyilu/RoboTwin2_0_cleaned_full/white}"
model_root="${MODEL_ROOT:-/lpai/models/Lightricks__LTX-2_5/26-08-11-2005}"
manifest_path="${MANIFEST_PATH:-$repo_root/data/white_robot_iclora_train.jsonl}"
preprocessed_root="${PREPROCESSED_ROOT:-$repo_root/data/white_robot_iclora_preprocessed}"
max_samples="${MAX_SAMPLES:-0}"
device="${DEVICE:-cuda:0}"
load_text_encoder_in_8bit="${LOAD_TEXT_ENCODER_IN_8BIT:-1}"
overwrite="${OVERWRITE:-0}"
resolution_buckets="${RESOLUTION_BUCKETS:-512x384x121}"

cd "$repo_root"
#bash scripts/setup_world_model_training_env.sh

preprocess_options=(--skip-audio)
if [[ "$load_text_encoder_in_8bit" == "1" ]]; then
    preprocess_options+=(--load-text-encoder-in-8bit)
fi
if [[ "$overwrite" == "1" ]]; then
    preprocess_options+=(--overwrite)
fi

"$repo_root/.venv/bin/python" scripts/build_white_robot_iclora_manifest.py \
    --data-root "$data_root" \
    --output "$manifest_path" \
    --max-samples "$max_samples"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$repo_root/.venv/bin/python" \
    packages/ltx-trainer/scripts/process_dataset.py "$manifest_path" \
    --resolution-buckets "$resolution_buckets" \
    --model-path "$model_root/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$model_root/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$model_root/vae/ltx-2.5-video-vae-bf16.safetensors" \
    --audio-vae-path "$model_root/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --output-dir "$preprocessed_root" \
    --batch-size 1 \
    --device "$device" \
    "${preprocess_options[@]}"
