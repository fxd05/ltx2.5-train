#!/usr/bin/env bash
# Launch white-robot V2V IC-LoRA FSDP training on the LPAI multi-node runtime.
# The platform provides NODE_NUM, RANK, GPU_NUM, MASTER_ADDR, and MASTER_PORT.
# Run this script once on every node with identical shared data/model paths and RUN_ID.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${CONFIG_PATH:-$repo_root/configs/white_robot_to_rgb_iclora.yaml}"
accelerate_config="${ACCELERATE_CONFIG:-$repo_root/packages/ltx-trainer/configs/accelerate/fsdp_size_wrap.yaml}"
preprocessed_root="${PREPROCESSED_ROOT:-$repo_root/data/white_robot_iclora_preprocessed_640x480x121}"
min_free_gb="${MIN_FREE_GB:-40}"
skip_gpu_check="${SKIP_GPU_CHECK:-0}"

: "${NODE_NUM:?NODE_NUM is required from the server}"
: "${RANK:?RANK is required from the server}"
: "${GPU_NUM:?GPU_NUM is required from the server}"
: "${MASTER_ADDR:?MASTER_ADDR is required from the server}"
: "${MASTER_PORT:?MASTER_PORT is required from the server}"

[[ "$NODE_NUM" =~ ^[1-9][0-9]*$ ]] || { echo "NODE_NUM must be a positive integer: $NODE_NUM" >&2; exit 2; }
[[ "$RANK" =~ ^[0-9]+$ ]] || { echo "RANK must be a non-negative integer: $RANK" >&2; exit 2; }
[[ "$GPU_NUM" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_NUM must be a positive integer: $GPU_NUM" >&2; exit 2; }
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ && "$MASTER_PORT" -le 65535 ]] || {
    echo "MASTER_PORT must be an integer from 1 to 65535: $MASTER_PORT" >&2
    exit 2
}
(( RANK < NODE_NUM )) || { echo "RANK=$RANK must be smaller than NODE_NUM=$NODE_NUM" >&2; exit 2; }
(( NODE_NUM >= 2 )) || { echo "Use scripts/train_white_robot_iclora_multigpu.sh for one node." >&2; exit 2; }
[[ "$min_free_gb" =~ ^[0-9]+$ ]] || { echo "MIN_FREE_GB must be a non-negative integer: $min_free_gb" >&2; exit 2; }

if [[ -n "${GPU_IDS:-}" ]]; then
    gpu_ids="$GPU_IDS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_ids="$CUDA_VISIBLE_DEVICES"
else
    gpu_ids="$(seq -s, 0 $((GPU_NUM - 1)))"
fi

IFS=',' read -r -a gpu_array <<< "$gpu_ids"
if [[ "${#gpu_array[@]}" -ne "$GPU_NUM" ]]; then
    echo "GPU_IDS contains ${#gpu_array[@]} entries, but GPU_NUM=$GPU_NUM." >&2
    exit 2
fi
for index in "${!gpu_array[@]}"; do
    gpu_array[$index]="${gpu_array[$index]//[[:space:]]/}"
    if [[ ! "${gpu_array[$index]}" =~ ^[0-9]+$ ]]; then
        echo "GPU_IDS must contain numeric GPU IDs, got: '${gpu_array[$index]}'" >&2
        exit 2
    fi
    for previous in "${gpu_array[@]:0:$index}"; do
        if [[ "${gpu_array[$index]}" == "$previous" ]]; then
            echo "GPU_IDS contains duplicate GPU ID: ${gpu_array[$index]}" >&2
            exit 2
        fi
    done
done
gpu_ids="$(IFS=,; echo "${gpu_array[*]}")"

total_processes=$((NODE_NUM * GPU_NUM))
if [[ -n "${NUM_PROCESSES:-}" ]]; then
    if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || [[ "$NUM_PROCESSES" -ne "$total_processes" ]]; then
        echo "NUM_PROCESSES must equal NODE_NUM * GPU_NUM ($total_processes), got: $NUM_PROCESSES" >&2
        exit 2
    fi
fi

safe_master_addr="${MASTER_ADDR//[^[:alnum:]]/_}"
run_id="${RUN_ID:-multinode_${safe_master_addr}_${MASTER_PORT}}"
runtime_config_path="${RUNTIME_CONFIG_PATH:-$repo_root/logs/runtime_configs/$(basename "${config_path%.*}")_${run_id}.yaml}"
log_path="${LOG_PATH:-$repo_root/logs/train_multinode_${run_id}_node${RANK}.log}"
dry_run="${DRY_RUN:-0}"

cd "$repo_root"

for required_path in \
    "$repo_root/.venv/bin/python" \
    "$repo_root/.venv/bin/accelerate" \
    "$config_path" \
    "$accelerate_config" \
    "$repo_root/scripts/render_white_robot_training_config.py"; do
    if [[ ! -f "$required_path" ]]; then
        echo "Required file does not exist: $required_path" >&2
        exit 1
    fi
done

latents_dir="$preprocessed_root/latents"
if [[ ! -d "$latents_dir" ]] || ! find "$latents_dir" -type f -name '*.pt' -print -quit | grep -q .; then
    echo "Preprocessed latents are missing under: $preprocessed_root" >&2
    exit 1
fi

if [[ "$skip_gpu_check" != "1" ]]; then
    required_free_mib=$((min_free_gb * 1024))
    echo "GPU preflight on node $RANK: require at least ${min_free_gb} GiB free per selected GPU."
    for gpu_id in "${gpu_array[@]}"; do
        free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" | head -n 1 | tr -dc '0-9')"
        if [[ -z "$free_mib" ]]; then
            echo "Could not read free memory for GPU $gpu_id." >&2
            exit 1
        fi
        printf '  GPU %s: %.1f GiB free\n' "$gpu_id" "$((free_mib / 1024))"
        if (( free_mib < required_free_mib )); then
            echo "GPU $gpu_id does not satisfy MIN_FREE_GB=$min_free_gb." >&2
            exit 1
        fi
    done
fi

mkdir -p "$(dirname "$runtime_config_path")" "$(dirname "$log_path")"
if (( RANK == 0 )); then
    "$repo_root/.venv/bin/python" scripts/render_white_robot_training_config.py \
        --config "$config_path" \
        --preprocessed-root "$preprocessed_root" \
        --output "$runtime_config_path" >/dev/null
else
    for (( attempt = 1; attempt <= 60; attempt++ )); do
        [[ -s "$runtime_config_path" ]] && break
        sleep 2
    done
    if [[ ! -s "$runtime_config_path" ]]; then
        echo "Timed out waiting for rank 0 to create runtime config: $runtime_config_path" >&2
        exit 1
    fi
fi

distributed_args=(
    --config_file "$accelerate_config"
    --num_machines "$NODE_NUM"
    --machine_rank "$RANK"
    --num_processes "$total_processes"
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$MASTER_PORT"
    --same_network
)

echo "Launching multi-node FSDP training"
echo "  node rank: $RANK/$NODE_NUM"
echo "  GPUs per node: $GPU_NUM ($gpu_ids)"
echo "  world size: $total_processes"
echo "  rendezvous: $MASTER_ADDR:$MASTER_PORT"
echo "  source config: $config_path"
echo "  runtime config: $runtime_config_path"
echo "  preprocessed data: $preprocessed_root"
echo "  log: $log_path"
echo "  accelerate: ${distributed_args[*]}"

if [[ "$dry_run" == "1" ]]; then
    echo "DRY_RUN=1: launch skipped."
    exit 0
fi

CUDA_VISIBLE_DEVICES="$gpu_ids" \
    "$repo_root/.venv/bin/accelerate" launch \
    "${distributed_args[@]}" \
    packages/ltx-trainer/scripts/train.py "$runtime_config_path" \
    --disable-progress-bars 2>&1 | tee "$log_path"
