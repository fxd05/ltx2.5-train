#!/usr/bin/env bash
# Evaluate selected IC-LoRA checkpoints on fixed, cross-task robot episodes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training_output_dir="${TRAINING_OUTPUT_DIR:-$repo_root/outputs/white_robot_to_rgb_iclora}"
checkpoint_dir="$training_output_dir/checkpoints"
final_step="${FINAL_STEP:-3000}"
checkpoint_stride="${CHECKPOINT_STRIDE:-500}"
tasks_file="${TASKS_FILE:-$repo_root/configs/white_robot_checkpoint_eval_tasks.tsv}"
eval_root="${EVAL_ROOT:-$training_output_dir/checkpoint_eval}"
eval_gpu_id="${EVAL_GPU_ID:-5}"
min_free_gb="${MIN_FREE_GB:-40}"
height="${EVAL_HEIGHT:-480}"
width="${EVAL_WIDTH:-640}"
num_frames="${EVAL_NUM_FRAMES:-121}"
frame_rate="${EVAL_FRAME_RATE:-24}"
seed="${EVAL_SEED:-2026}"
offload="${EVAL_OFFLOAD:-cpu}"
quantization="${EVAL_QUANTIZATION:-fp8-cast}"
run_log_dir="${RUN_LOG_DIR:-$repo_root/logs/checkpoint_eval}"
lock_file="$run_log_dir/evaluation.lock"
report_path="$eval_root/REPORT.md"

if [[ ! "$final_step" =~ ^[1-9][0-9]*$ ]] || [[ ! "$checkpoint_stride" =~ ^[1-9][0-9]*$ ]]; then
    echo "FINAL_STEP and CHECKPOINT_STRIDE must be positive integers." >&2
    exit 2
fi
if [[ ! "$eval_gpu_id" =~ ^[0-9]+$ ]] || [[ ! "$min_free_gb" =~ ^[0-9]+$ ]]; then
    echo "EVAL_GPU_ID and MIN_FREE_GB must be non-negative integers." >&2
    exit 2
fi
if [[ ! "$height" =~ ^[1-9][0-9]*$ ]] || [[ ! "$width" =~ ^[1-9][0-9]*$ ]] \
    || (( num_frames < 9 || (num_frames - 1) % 8 )); then
    echo "Evaluation dimensions must be positive and frame count must be 8k+1." >&2
    exit 2
fi

for required_path in \
    "$repo_root/.venv/bin/python" \
    "$repo_root/scripts/infer_white_robot_iclora.py" \
    "$tasks_file"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done

mkdir -p "$run_log_dir" "$eval_root"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Checkpoint evaluation is already running."
    exit 0
fi

final_checkpoint="$checkpoint_dir/$(printf 'lora_weights_step_%05d.safetensors' "$final_step")"
final_state="$checkpoint_dir/$(printf 'training_state_step_%05d.pt' "$final_step")"
if [[ ! -f "$final_checkpoint" || ! -f "$final_state" ]]; then
    echo "Waiting for final checkpoint: $final_checkpoint"
    exit 0
fi

if pgrep -f "packages/ltx-trainer/scripts/train.py.*white_robot_to_rgb_iclora" >/dev/null; then
    echo "Final checkpoint exists, but white-robot training processes are still active."
    exit 0
fi

required_free_mib=$((min_free_gb * 1024))
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$eval_gpu_id" | head -n 1 | tr -dc '0-9')"
if [[ -z "$free_mib" ]]; then
    echo "Could not read free memory for GPU $eval_gpu_id." >&2
    exit 1
fi
if (( free_mib < required_free_mib )); then
    echo "Waiting for GPU $eval_gpu_id: ${free_mib} MiB free, need ${required_free_mib} MiB."
    exit 0
fi

if [[ -f "$report_path" ]] && grep -q '^status: complete$' "$report_path"; then
    echo "Checkpoint evaluation is already complete: $report_path"
    exit 0
fi

report_tmp="$report_path.tmp"
{
    echo "# White-Robot IC-LoRA Checkpoint Evaluation"
    echo
    echo "status: running"
    echo "- Evaluation GPU: $eval_gpu_id"
    echo "- Resolution: ${width}x${height}x${num_frames}"
    echo "- Seed: $seed"
    echo "- Checkpoint stride: $checkpoint_stride steps"
    echo
    echo "| Checkpoint | Task | Video | Log | Status |"
    echo "| --- | --- | --- | --- | --- |"
} > "$report_tmp"

mapfile -t checkpoints < <(
    find "$checkpoint_dir" -maxdepth 1 -type f -name 'lora_weights_step_*.safetensors' -printf '%f\n' \
        | sort -V
)

if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "No LoRA checkpoints found in $checkpoint_dir" >&2
    exit 1
fi

for checkpoint_name in "${checkpoints[@]}"; do
    checkpoint_step="${checkpoint_name#lora_weights_step_}"
    checkpoint_step="${checkpoint_step%.safetensors}"
    checkpoint_step=$((10#$checkpoint_step))
    if (( checkpoint_step % checkpoint_stride != 0 && checkpoint_step != final_step )); then
        continue
    fi

    checkpoint_path="$checkpoint_dir/$checkpoint_name"
    checkpoint_label="step_$(printf '%05d' "$checkpoint_step")"
    while IFS=$'\t' read -r task_id episode_dir; do
        [[ -z "$task_id" || "$task_id" == \#* ]] && continue
        for required_file in head_color.mp4 white_robot.mp4 instruction.txt; do
            if [[ ! -f "$episode_dir/$required_file" ]]; then
                echo "Task '$task_id' is missing $required_file: $episode_dir" >&2
                exit 1
            fi
        done

        output_dir="$eval_root/$checkpoint_label/$task_id"
        output_path="$output_dir/generated_rgb.mp4"
        task_log="$run_log_dir/${checkpoint_label}_${task_id}.log"
        mkdir -p "$output_dir"

        if [[ -s "$output_path" ]]; then
            printf '| %s | %s | `%s` | `%s` | skipped-existing |\n' \
                "$checkpoint_label" "$task_id" "${output_path#$repo_root/}" "${task_log#$repo_root/}" >> "$report_tmp"
            continue
        fi

        echo "Evaluating $checkpoint_label on $task_id"
        if CUDA_VISIBLE_DEVICES="$eval_gpu_id" "$repo_root/.venv/bin/python" \
            scripts/infer_white_robot_iclora.py \
            --episode "$episode_dir" \
            --ic-lora-path "$checkpoint_path" \
            --height "$height" --width "$width" \
            --num-frames "$num_frames" --frame-rate "$frame_rate" \
            --seed "$seed" --offload "$offload" --quantization "$quantization" \
            --output-path "$output_path" >"$task_log" 2>&1; then
            status="generated"
        else
            status="failed"
        fi
        printf '| %s | %s | `%s` | `%s` | %s |\n' \
            "$checkpoint_label" "$task_id" "${output_path#$repo_root/}" "${task_log#$repo_root/}" "$status" >> "$report_tmp"
    done < "$tasks_file"
done

{
    echo
    echo "## Review"
    echo
    echo "Compare each task vertically across checkpoints using the fixed input episode and seed."
    echo "Assess first-frame consistency, white-motion following, task-object interaction, and temporal stability."
    echo
    echo "status: complete"
} >> "$report_tmp"
mv "$report_tmp" "$report_path"
echo "Checkpoint evaluation complete: $report_path"
