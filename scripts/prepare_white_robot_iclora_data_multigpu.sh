#!/usr/bin/env bash
# Preprocess paired white-robot IC-LoRA data on multiple GPUs.
# Example:
#   GPU_IDS=2,4,5,6 RESOLUTION_BUCKETS=640x480x121 \
#     PREPROCESSED_ROOT=data/white_robot_iclora_preprocessed_640x480x121 \
#     bash scripts/prepare_white_robot_iclora_data_multigpu.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-/lpai/volumes/mind-eb-ali-sh/dulingyi/sunyilu/RoboTwin2_0_cleaned_cut/cut121/white}"
model_root="${MODEL_ROOT:-/lpai/models/Lightricks__LTX-2_5/26-08-11-2005}"
manifest_path="${MANIFEST_PATH:-$repo_root/data/white_robot_iclora_train_640x480x121.jsonl}"
preprocessed_root="${PREPROCESSED_ROOT:-$repo_root/data/white_robot_iclora_preprocessed_640x480x121}"
max_samples="${MAX_SAMPLES:-0}"
resolution_buckets="${RESOLUTION_BUCKETS:-640x480x121}"
batch_size="${BATCH_SIZE:-1}"
load_text_encoder_in_8bit="${LOAD_TEXT_ENCODER_IN_8BIT:-1}"
overwrite="${OVERWRITE:-0}"
gpu_ids="${GPU_IDS:-}"
min_free_gb="${MIN_FREE_GB:-100}"
skip_gpu_check="${SKIP_GPU_CHECK:-0}"
log_path="${LOG_PATH:-$repo_root/logs/prepare_multigpu_$(date +%Y%m%d_%H%M%S).log}"

if [[ -z "$gpu_ids" ]]; then
    echo "GPU_IDS is required, for example: GPU_IDS=0,1,2,3" >&2
    exit 2
fi
if [[ ! "$min_free_gb" =~ ^[0-9]+$ ]]; then
    echo "MIN_FREE_GB must be a non-negative integer, got: $min_free_gb" >&2
    exit 2
fi
if [[ ! "$max_samples" =~ ^[0-9]+$ ]]; then
    echo "MAX_SAMPLES must be a non-negative integer, got: $max_samples" >&2
    exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer, got: $batch_size" >&2
    exit 2
fi

IFS=',' read -r -a gpu_array <<< "$gpu_ids"
if [[ "${#gpu_array[@]}" -lt 2 ]]; then
    echo "Use scripts/prepare_white_robot_iclora_data.sh for one GPU; multi-GPU preprocessing requires at least two GPU IDs." >&2
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
    "$repo_root/.venv/bin/python" \
    "$repo_root/.venv/bin/accelerate" \
    "$repo_root/scripts/build_white_robot_iclora_manifest.py" \
    "$repo_root/packages/ltx-trainer/scripts/process_dataset.py" \
    "$model_root/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    "$model_root/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    "$model_root/vae/ltx-2.5-video-vae-bf16.safetensors" \
    "$model_root/vae/ltx-2.5-audio-vae-bf16.safetensors"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$manifest_path")" "$preprocessed_root"
"$repo_root/.venv/bin/python" scripts/build_white_robot_iclora_manifest.py \
    --data-root "$data_root" \
    --output "$manifest_path" \
    --max-samples "$max_samples"

if [[ ! -s "$manifest_path" ]]; then
    echo "Manifest is empty: $manifest_path" >&2
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

preprocess_options=(--skip-audio)
if [[ "$load_text_encoder_in_8bit" == "1" ]]; then
    preprocess_options+=(--load-text-encoder-in-8bit)
fi
if [[ "$overwrite" == "1" ]]; then
    preprocess_options+=(--overwrite)
fi

mkdir -p "$(dirname "$log_path")"
echo "Launching ${num_processes}-GPU preprocessing on physical GPUs: $gpu_ids"
echo "Manifest: $manifest_path"
echo "Preprocessed output: $preprocessed_root"
echo "Resolution buckets: $resolution_buckets"
echo "Log: $log_path"

CUDA_VISIBLE_DEVICES="$gpu_ids" \
    "$repo_root/.venv/bin/accelerate" launch \
    --multi_gpu \
    --num_processes "$num_processes" \
    packages/ltx-trainer/scripts/process_dataset.py "$manifest_path" \
    --resolution-buckets "$resolution_buckets" \
    --model-path "$model_root/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$model_root/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$model_root/vae/ltx-2.5-video-vae-bf16.safetensors" \
    --audio-vae-path "$model_root/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --output-dir "$preprocessed_root" \
    --batch-size "$batch_size" \
    --device cuda \
    "${preprocess_options[@]}" 2>&1 | tee "$log_path"
