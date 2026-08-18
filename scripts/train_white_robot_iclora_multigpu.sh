#!/usr/bin/env bash
# Launch distributed white-robot V2V IC-LoRA training with FSDP.
# Example:
#   GPU_IDS=0,1,2,3 CONFIG_PATH=configs/white_robot_to_rgb_iclora.yaml \
#     PREPROCESSED_ROOT=data/white_robot_iclora_preprocessed_640x480x121 \
#     bash scripts/train_white_robot_iclora_multigpu.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${CONFIG_PATH:-$repo_root/configs/white_robot_to_rgb_iclora.yaml}"
accelerate_config="${ACCELERATE_CONFIG:-$repo_root/packages/ltx-trainer/configs/accelerate/fsdp_size_wrap.yaml}"
preprocessed_root="${PREPROCESSED_ROOT:-$repo_root/data/white_robot_iclora_preprocessed_640x480x121}"
gpu_ids="${GPU_IDS:-}"
min_free_gb="${MIN_FREE_GB:-100}"
skip_gpu_check="${SKIP_GPU_CHECK:-0}"
log_path="${LOG_PATH:-$repo_root/logs/train_multigpu_$(date +%Y%m%d_%H%M%S).log}"
runtime_config_path="${RUNTIME_CONFIG_PATH:-$repo_root/logs/runtime_configs/$(basename "${config_path%.*}")_$(date +%Y%m%d_%H%M%S)_$$.yaml}"

if [[ -z "$gpu_ids" ]]; then
    echo "GPU_IDS is required, for example: GPU_IDS=0,1,2,3" >&2
    exit 2
fi
if [[ ! "$min_free_gb" =~ ^[0-9]+$ ]]; then
    echo "MIN_FREE_GB must be a non-negative integer, got: $min_free_gb" >&2
    exit 2
fi

IFS=',' read -r -a gpu_array <<< "$gpu_ids"
if [[ "${#gpu_array[@]}" -lt 2 ]]; then
    echo "Use scripts/train_white_robot_iclora.sh for one GPU; multi-GPU launch requires at least two GPU IDs." >&2
    exit 2
fi

for index in "${!gpu_array[@]}"; do
    gpu_array[$index]="${gpu_array[$index]//[[:space:]]/}"
    if [[ ! "${gpu_array[$index]}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID: '${gpu_array[$index]}'" >&2
        exit 2
    fi
    for previous in "${gpu_array[@]:0:$index}"; do
        if [[ "${gpu_array[$index]}" == "$previous" ]]; then
            echo "GPU_IDS contains duplicate GPU ID: ${gpu_array[$index]}" >&2
            exit 2
        fi
    done
done

num_processes="${NUM_PROCESSES:-${#gpu_array[@]}}"
if [[ ! "$num_processes" =~ ^[0-9]+$ ]] || [[ "$num_processes" -ne "${#gpu_array[@]}" ]]; then
    echo "NUM_PROCESSES ($num_processes) must equal the number of GPU_IDS (${#gpu_array[@]})." >&2
    exit 2
fi

cd "$repo_root"

for required_path in \
    "$config_path" \
    "$accelerate_config" \
    "$repo_root/scripts/render_white_robot_training_config.py"; do
    if [[ ! -f "$required_path" ]]; then
        echo "Required file does not exist: $required_path" >&2
        exit 1
    fi
done

latents_dir="$preprocessed_root/latents"
if [[ ! -d "$latents_dir" ]] || ! find "$latents_dir" -type f -name "*.pt" -print -quit | grep -q .; then
    echo "Preprocessed latents are missing under: $preprocessed_root" >&2
    echo "Run scripts/prepare_white_robot_iclora_data.sh first." >&2
    exit 1
fi

if [[ "$skip_gpu_check" != "1" ]]; then
    required_free_mib=$((min_free_gb * 1024))
    echo "GPU preflight: require at least ${min_free_gb} GiB free per selected GPU."
    for gpu_id in "${gpu_array[@]}"; do
        free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" | head -n 1 | tr -dc '0-9')"
        if [[ -z "$free_mib" ]]; then
            echo "Could not read free memory for GPU $gpu_id." >&2
            exit 1
        fi
        printf '  GPU %s: %.1f GiB free\n' "$gpu_id" "$((free_mib / 1024))"
        if (( free_mib < required_free_mib )); then
            echo "GPU $gpu_id does not satisfy MIN_FREE_GB=$min_free_gb. Choose another GPU set or set SKIP_GPU_CHECK=1 only when ownership is confirmed." >&2
            exit 1
        fi
    done
fi

mkdir -p "$(dirname "$log_path")"
"$repo_root/.venv/bin/python" scripts/render_white_robot_training_config.py \
    --config "$config_path" \
    --preprocessed-root "$preprocessed_root" \
    --output "$runtime_config_path" >/dev/null

echo "Launching FSDP training with ${num_processes} processes on physical GPUs: ${gpu_ids}"
echo "Source config: $config_path"
echo "Runtime config: $runtime_config_path"
echo "Preprocessed data: $preprocessed_root"
echo "Log: $log_path"

CUDA_VISIBLE_DEVICES="$gpu_ids" \
    "$repo_root/.venv/bin/accelerate" launch \
    --config_file "$accelerate_config" \
    --num_processes "$num_processes" \
    packages/ltx-trainer/scripts/train.py "$runtime_config_path" \
    --disable-progress-bars 2>&1 | tee "$log_path"
