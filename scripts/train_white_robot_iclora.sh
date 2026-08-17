#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${CONFIG_PATH:-$repo_root/configs/white_robot_to_rgb_iclora.yaml}"
accelerate_config="${ACCELERATE_CONFIG:-$repo_root/packages/ltx-trainer/configs/accelerate/fsdp_size_wrap.yaml}"
num_processes="${NUM_PROCESSES:-8}"
preprocessed_root="${PREPROCESSED_ROOT:-$repo_root/data/white_robot_iclora_preprocessed_512x384x121}"

cd "$repo_root"
#bash scripts/setup_world_model_training_env.sh

latents_dir="$preprocessed_root/latents"
if [[ ! -d "$latents_dir" ]] || ! find "$latents_dir" -type f -name "*.pt" -print -quit | grep -q .; then
    echo "Preprocessed latents are missing. Run scripts/prepare_white_robot_iclora_data.sh first." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    "$repo_root/.venv/bin/accelerate" launch \
    --config_file "$accelerate_config" \
    --num_processes "$num_processes" \
    packages/ltx-trainer/scripts/train.py "$config_path" \
    --disable-progress-bars
