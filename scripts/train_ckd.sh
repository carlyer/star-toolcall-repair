#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/configs/models.sh"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT/openrlhf:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export WANDB_MODE=offline
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXP_PATH="$ROOT/results/ckd-$RUN_ID"
MODEL_PATH="$STUDENT_MODEL"
TEACHER_MODEL_PATH="$TEACHER_MODEL"
REWARD_PATH="$ROOT/upstream/simrl.py"
DATA_PATH="${RL_DATA:-$ROOT/data/rl_data.jsonl}"
REF_DATA_PATH="${CKD_DATA:-$ROOT/data/kd_data.jsonl}"
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Missing STAR environment'; exit 1; }
[[ -f "$MODEL_PATH/config.json" && -s "$DATA_PATH" ]] || { echo 'Missing model or RL dataset'; exit 1; }
[[ -f "$TEACHER_MODEL_PATH/config.json" ]] || { echo "Missing teacher model"; exit 1; }
IFS=',' read -r -a REF_FILES <<< "$REF_DATA_PATH"
for REF_FILE in "${REF_FILES[@]}"; do
 [[ -s "$REF_FILE" ]] || { echo "Missing CKD source: $REF_FILE"; exit 1; }
done
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_model_files.py" "$MODEL_PATH" "$TEACHER_MODEL_PATH"
[[ "${GPU_ALLOCATION_CONFIRMED:-0}" == 1 ]] || { echo "Request GPU allocation before training; then set GPU_ALLOCATION_CONFIRMED=1"; exit 1; }
[[ ! -e "$EXP_PATH" ]] || { echo "Run directory already exists; choose a new RUN_ID"; exit 1; }
source "$ROOT/scripts/gpu_lock.sh"
mkdir -p "$EXP_PATH"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=1
python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --enforce_eager \
   --vllm_enable_sleep \
   --prompt_max_len 5120 \
   --pretrain $MODEL_PATH \
   --teacher $TEACHER_MODEL_PATH \
   --remote_rm_url $REWARD_PATH \
   --save_path $EXP_PATH/checkpoints \
   --save_hf_ckpt \
   --max_ckpt_num 1 \
   --micro_train_batch_size "${CKD_MICRO_TRAIN_BATCH:-2}" \
   --train_batch_size 128 \
   --micro_rollout_batch_size "${CKD_MICRO_ROLLOUT_BATCH:-4}" \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 1 \
   --prompt_max_len 5120 \
   --max_samples 100000000 \
   --generate_max_len 512 \
   --advantage_estimator dr_grpo \
   --zero_stage 1 \
   --gradient_checkpointing \
   --bf16 \
   --actor_learning_rate 3e-6 \
   --init_kl_coef 0.0 \
   --kd_kl_coef 1.0 \
   --op_kl_coef 0.0 \
   --ppo_coef 0.0 \
   --gamma 1.0 \
   --kd_ratio 1.0 \
   --kd_kl_l1_coef 10.0 \
   --l2 1e-4 \
   --use_kl_loss \
   --kd_kl_type fkl \
   --kl_estimator k2 \
   --ref_mask_type all \
   --prompt_data $DATA_PATH \
   --ref_prompt_data $REF_DATA_PATH \
   --input_key inputs \
   --normalize_reward \
   --packing_samples \
   --flash_attn \
   --save_steps -1 \
   --eval_steps -1 \
   --ckpt_path $EXP_PATH/tmp_checkpoints \
   --label_key targets \
   --ref_key targets \
   --token_level_kl \
   --use_tensorboard $EXP_PATH/tensorboard

