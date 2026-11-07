#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/configs/models.sh"
PYTHON="${STAR_PYTHON:-$ROOT/.venv/bin/python}"
[[ -f "$TEACHER_MODEL/config.json" ]] || { echo 'Missing original star-lab/Teacher-8B'; exit 1; }
"$PYTHON" "$ROOT/scripts/check_model_files.py" "$TEACHER_MODEL"
[[ "${GPU_ALLOCATION_CONFIRMED:-0}" == 1 ]] || { echo 'Request GPU allocation before teacher rollout; then set GPU_ALLOCATION_CONFIRMED=1'; exit 1; }
source "$ROOT/scripts/gpu_lock.sh"
"$PYTHON" "$ROOT/upstream/data_process/teacher_rollout.py" --input "${CKD_SEED_MESSAGES:-$ROOT/upstream/example_messages.jsonl}" --output "${CKD_MESSAGES_OUTPUT:-$ROOT/data/kd_messages.jsonl}" --model-path "$TEACHER_MODEL" --rollout-n 8 --dp-size "${ROLLOUT_DP_SIZE:-2}"
"$PYTHON" "$ROOT/upstream/data_process/messages_to_trainset.py" --input "${CKD_MESSAGES_OUTPUT:-$ROOT/data/kd_messages.jsonl}" --output "${CKD_DATA:-$ROOT/data/kd_data.jsonl}" --tokenizer-path "$TEACHER_MODEL" --add-reasoning-content
