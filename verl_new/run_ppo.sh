#!/usr/bin/env bash
# verl PPO 训练启动脚本（适配当前 verl 版本，使用 Ray 分布式）
# 用法: bash verl_new/run_ppo.sh

set -euo pipefail

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_DIR="verl_new/log"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ppo_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "===== Run started at $(date '+%Y-%m-%d %H:%M:%S') ====="

# ── 环境变量 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1

N_GPUS=1

# ── 路径 ──────────────────────────────────────────────────────────────────────
MODEL_PATH="models/qwen2.5-3b-instruct"
REWARD_MODEL_PATH="models/skywork-reward-v2-qwen3-4b"
TRAIN_PARQUET="data/coig-cqia/verl_parquet/train.parquet"
VAL_PARQUET="data/coig-cqia/verl_parquet/val.parquet"
OUTPUT_DIR="outputs/verl_ppo"

# ── 启动 ──────────────────────────────────────────────────────────────────────
python -m verl.trainer.main_ppo \
    \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=4 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    critic.model.path="${MODEL_PATH}" \
    critic.optim.lr=1e-5 \
    critic.model.enable_gradient_checkpointing=true \
    critic.fsdp.param_offload=false \
    critic.ppo_micro_batch_size_per_gpu=2 \
    \
    reward.reward_model.enable=true \
    reward.reward_model.model_path="${REWARD_MODEL_PATH}" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.gpu_memory_utilization=0.8 \
    reward.reward_model.rollout.tensor_model_parallel_size=1 \
    reward.reward_model.rollout.prompt_length=512 \
    reward.reward_model.rollout.response_length=512 \
    \
    algorithm.kl_ctrl.kl_coef=0.15 \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    algorithm.adv_estimator=gae \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=verl_ppo_coig \
    trainer.experiment_name=qwen2.5-3b-ppo \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.max_actor_ckpt_to_keep=4 \
    trainer.max_critic_ckpt_to_keep=4 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUTPUT_DIR}"
