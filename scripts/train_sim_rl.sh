#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/configs/models.sh"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT/openrlhf:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export WANDB_MODE=offline
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
# On 2 GPUs, restore the original 8-GPU global microbatch (8*2 = 2*8).
if [[ "${EXPERIMENT_ID:-}" == generic-teacher-hammer-* ]]; then
    SIM_MICRO_TRAIN_BATCH="${SIM_MICRO_TRAIN_BATCH:-8}"
    SIM_MICRO_ROLLOUT_BATCH="${SIM_MICRO_ROLLOUT_BATCH:-16}"
fi
EXP_PATH="$ROOT/results/sim_rl-$RUN_ID"
MODEL_PATH="${CKD_CHECKPOINT:?Set CKD_CHECKPOINT to the completed CKD final model}"
TEACHER_MODEL_PATH="$TEACHER_MODEL"
REWARD_PATH="$ROOT/upstream/simrl.py"
DATA_PATH="${RL_DATA:-$ROOT/data/rl_data.jsonl}"
REF_DATA_PATH="${CKD_DATA:-$ROOT/data/kd_data.jsonl}"
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Missing STAR environment'; exit 1; }
[[ -f "$MODEL_PATH/config.json" && -s "$DATA_PATH" ]] || { echo 'Missing model or RL dataset'; exit 1; }
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_model_files.py" "$MODEL_PATH"
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
   --vllm_gpu_memory_utilization 0.7 \
   --init_kl_coef 1e-3 \
   --enable_prefix_caching \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator group_norm \
   --pretrain $MODEL_PATH \
   --remote_rm_url $REWARD_PATH \
   --save_path $EXP_PATH/checkpoints \
   --ckpt_path $EXP_PATH/tmp_checkpoints \
   --save_steps -1 \
   --eval_steps -1 \
   --save_hf_ckpt \
   --max_ckpt_num 1 \
   --micro_train_batch_size "${SIM_MICRO_TRAIN_BATCH:-2}" \
   --train_batch_size 128 \
   --micro_rollout_batch_size "${SIM_MICRO_ROLLOUT_BATCH:-4}" \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 3 \
   --prompt_max_len 5120 \
   --max_samples 100000000 \
   --generate_max_len 512 \
   --zero_stage 1 \
   --flash_attn \
   --bf16 \
   --actor_learning_rate 3e-7 \
   --prompt_data $DATA_PATH \
   --input_key inputs \
   --label_key targets \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --use_tensorboard $EXP_PATH/tensorboard \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0.01 0.99 \
   --dynamic_filtering_min_diff 0 \
   --kd_kl_coef 0.0 \
   --op_kl_coef 0.0 \
   --ppo_coef 1.0

