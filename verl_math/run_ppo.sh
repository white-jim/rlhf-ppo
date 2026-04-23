#!/usr/bin/env bash
# verl PPO 训练启动脚本 — GSM8K_zh 数学推理
# 用法: bash verl_math/run_ppo.sh

set -euo pipefail

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_DIR="verl_math/log"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ppo_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "===== Run started at $(date '+%Y-%m-%d %H:%M:%S') ====="

# ── 环境变量 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1

N_GPUS=1

# ── 路径 ──────────────────────────────────────────────────────────────────────
MODEL_PATH="models/qwen2.5-3b-instruct"
TRAIN_PARQUET="data/gsm8k_zh/verl_parquet/train.parquet"
VAL_PARQUET="data/gsm8k_zh/verl_parquet/test.parquet"
OUTPUT_DIR="outputs/verl_ppo_gsm8k"

# ── 启动 ──────────────────────────────────────────────────────────────────────
ray stop --force 2>/dev/null || true

python -m verl.trainer.main_ppo \
    \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=true \
    data.truncation=left \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.02 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    critic.model.path="${MODEL_PATH}" \
    critic.optim.lr=3e-6 \
    critic.model.enable_gradient_checkpointing=true \
    critic.fsdp.param_offload=false \
    critic.ppo_micro_batch_size_per_gpu=2 \
    \
    reward.custom_reward_function.path=verl_math/reward_fn.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.reward_model.enable=false \
    \
    algorithm.kl_ctrl.kl_coef=0.15 \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    algorithm.adv_estimator=gae \
    \
    trainer.critic_warmup=50 \
    trainer.logger='["console"]' \
    trainer.project_name=verl_ppo_gsm8k \
    trainer.experiment_name=qwen2.5-3b-ppo-gsm8k \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.max_actor_ckpt_to_keep=4 \
    trainer.max_critic_ckpt_to_keep=4 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUTPUT_DIR}"