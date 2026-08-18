#!/usr/bin/env bash
# Run the paired white-robot pipeline on the LPAI multi-node runtime.
# Rank 0 preprocesses the shared dataset once; every node then starts the
# existing multi-node FSDP launcher with the same PREPROCESSED_ROOT and RUN_ID.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
preprocessed_root="${PREPROCESSED_ROOT:-$repo_root/data/white_robot_iclora_preprocessed_640x480x121}"
wait_timeout_seconds="${PREPROCESS_WAIT_TIMEOUT_SECONDS:-86400}"
poll_interval_seconds="${PREPROCESS_POLL_INTERVAL_SECONDS:-10}"
dry_run="${DRY_RUN:-0}"

: "${NODE_NUM:?NODE_NUM is required from the server}"
: "${RANK:?RANK is required from the server}"
: "${GPU_NUM:?GPU_NUM is required from the server}"
: "${MASTER_ADDR:?MASTER_ADDR is required from the server}"
: "${MASTER_PORT:?MASTER_PORT is required from the server}"

[[ "$NODE_NUM" =~ ^[1-9][0-9]*$ ]] || { echo "NODE_NUM must be a positive integer: $NODE_NUM" >&2; exit 2; }
[[ "$RANK" =~ ^[0-9]+$ ]] || { echo "RANK must be a non-negative integer: $RANK" >&2; exit 2; }
[[ "$GPU_NUM" =~ ^([2-9]|[1-9][0-9]+)$ ]] || {
    echo "GPU_NUM must be at least 2 because preprocessing uses the multi-GPU launcher: $GPU_NUM" >&2
    exit 2
}
[[ "$wait_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "PREPROCESS_WAIT_TIMEOUT_SECONDS must be a positive integer: $wait_timeout_seconds" >&2
    exit 2
}
[[ "$poll_interval_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "PREPROCESS_POLL_INTERVAL_SECONDS must be a positive integer: $poll_interval_seconds" >&2
    exit 2
}
(( RANK < NODE_NUM )) || { echo "RANK=$RANK must be smaller than NODE_NUM=$NODE_NUM" >&2; exit 2; }
(( NODE_NUM >= 2 )) || { echo "Use the single-node launchers when NODE_NUM=1." >&2; exit 2; }

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

safe_master_addr="${MASTER_ADDR//[^[:alnum:]]/_}"
run_id="${RUN_ID:-multinode_${safe_master_addr}_${MASTER_PORT}}"
pipeline_id="${PIPELINE_ID:-$run_id}"
[[ "$pipeline_id" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "PIPELINE_ID may contain only letters, digits, dot, underscore, and hyphen: $pipeline_id" >&2
    exit 2
}

state_dir="${PIPELINE_STATE_DIR:-$preprocessed_root/.multinode_pipeline_state}"
started_path="$state_dir/${pipeline_id}.started"
ready_path="$state_dir/${pipeline_id}.ready"
failed_path="$state_dir/${pipeline_id}.failed"
preprocess_log_path="${PREPROCESS_LOG_PATH:-$repo_root/logs/prepare_multinode_${pipeline_id}_node0.log}"
preprocess_launcher="$repo_root/scripts/prepare_white_robot_iclora_data_multigpu.sh"
train_launcher="$repo_root/scripts/train_white_robot_iclora_multinode.sh"

write_state() {
    local path="$1"
    local status="$2"
    local temporary_path="${path}.$$"
    {
        printf 'status=%s\n' "$status"
        printf 'pipeline_id=%s\n' "$pipeline_id"
        printf 'node_rank=%s\n' "$RANK"
        printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$temporary_path"
    mv "$temporary_path" "$path"
}

latents_are_ready() {
    local latents_dir="$preprocessed_root/latents"
    [[ -d "$latents_dir" ]] && find "$latents_dir" -type f -name '*.pt' -print -quit | grep -q .
}

export GPU_IDS="$gpu_ids"
export PREPROCESSED_ROOT="$preprocessed_root"
export RUN_ID="$run_id"

cd "$repo_root"
for required_path in "$preprocess_launcher" "$train_launcher"; do
    if [[ ! -f "$required_path" ]]; then
        echo "Required launcher does not exist: $required_path" >&2
        exit 1
    fi
done

if [[ "$dry_run" == "1" ]]; then
    echo "DRY_RUN=1: pipeline launch skipped."
    echo "  node rank: $RANK/$NODE_NUM"
    echo "  rank 0 preprocessor: $preprocess_launcher"
    echo "  all-node trainer: $train_launcher"
    echo "  pipeline ID: $pipeline_id"
    echo "  shared state directory: $state_dir"
    echo "  preprocessed data: $preprocessed_root"
    exit 0
fi

mkdir -p "$state_dir"

if [[ "$RANK" == "0" ]]; then
    rm -f "$started_path" "$ready_path" "$failed_path"
    write_state "$started_path" "preprocessing"

    echo "Rank 0 preprocessing the shared dataset before multi-node training."
    echo "  pipeline ID: $pipeline_id"
    echo "  GPUs: $gpu_ids"
    echo "  output: $preprocessed_root"
    echo "  log: $preprocess_log_path"

    if GPU_IDS="$gpu_ids" NUM_PROCESSES="$GPU_NUM" LOG_PATH="$preprocess_log_path" \
        bash "$preprocess_launcher"; then
        if ! latents_are_ready; then
            echo "Preprocessing exited successfully but latents are missing under: $preprocessed_root" >&2
            write_state "$failed_path" "latents_missing"
            exit 1
        fi
        write_state "$ready_path" "ready"
        echo "Shared preprocessing completed. Starting multi-node training on all nodes."
    else
        preprocess_status=$?
        write_state "$failed_path" "failed"
        echo "Preprocessing failed on rank 0; wrote failure state: $failed_path" >&2
        exit "$preprocess_status"
    fi
else
    echo "Rank $RANK waiting for rank 0 preprocessing."
    echo "  pipeline ID: $pipeline_id"
    echo "  state directory: $state_dir"
    elapsed_seconds=0
    while true; do
        if [[ -f "$failed_path" ]]; then
            echo "Rank 0 preprocessing failed. State:" >&2
            cat "$failed_path" >&2
            exit 1
        fi
        if [[ -f "$ready_path" ]]; then
            if latents_are_ready; then
                echo "Rank 0 preprocessing is ready. Starting multi-node training."
                break
            fi
            echo "Ready state exists but latents are unavailable under: $preprocessed_root" >&2
            exit 1
        fi
        if (( elapsed_seconds >= wait_timeout_seconds )); then
            echo "Timed out after ${wait_timeout_seconds}s waiting for rank 0 preprocessing." >&2
            exit 1
        fi
        sleep "$poll_interval_seconds"
        elapsed_seconds=$((elapsed_seconds + poll_interval_seconds))
    done
fi

exec bash "$train_launcher"
