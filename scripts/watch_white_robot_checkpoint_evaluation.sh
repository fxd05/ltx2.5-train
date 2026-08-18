#!/usr/bin/env bash
# Poll until final training completes, then run the checkpoint evaluation once.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
poll_seconds="${POLL_SECONDS:-300}"
run_log="${WATCHDOG_LOG:-$repo_root/logs/checkpoint_eval/watchdog.log}"

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "POLL_SECONDS must be a positive integer." >&2
    exit 2
fi

mkdir -p "$(dirname "$run_log")"
while true; do
    timestamp="$(date '+%F %T %Z')"
    echo "[$timestamp] checkpoint-evaluation watchdog tick" >> "$run_log"
    if bash "$repo_root/scripts/run_white_robot_checkpoint_evaluation.sh" >> "$run_log" 2>&1; then
        if [[ -f "$repo_root/outputs/white_robot_to_rgb_iclora/checkpoint_eval/REPORT.md" ]] \
            && grep -q '^status: complete$' "$repo_root/outputs/white_robot_to_rgb_iclora/checkpoint_eval/REPORT.md"; then
            echo "[$timestamp] checkpoint evaluation completed" >> "$run_log"
            exit 0
        fi
    else
        echo "[$timestamp] evaluator exited with an error; retrying after next interval" >> "$run_log"
    fi
    sleep "$poll_seconds"
done
