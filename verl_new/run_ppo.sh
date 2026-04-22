#!/usr/bin/env bash
# verl PPO 训练启动脚本
# 用法: bash verl_ppo/run_ppo.sh

set -euo pipefail

# ── 环境变量 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1
export REWARD_MODEL_PATH="models/internlm2-7b-reward"
export REWARD_MODEL_DTYPE="bfloat16"

# 单卡训练,nproc_per_node=1
N_GPUS=1

# ── 路径 ──────────────────────────────────────────────────────────────────────
MODEL_PATH="models/qwen2.5-3b-instruct"
TRAIN_PARQUET="data/coig-cqia/verl_parquet/train.parquet"
VAL_PARQUET="data/coig-cqia/verl_parquet/val.parquet"
OUTPUT_DIR="outputs/verl_ppo"

# ── 启动 ──────────────────────────────────────────────────────────────────────
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=${N_GPUS} \
  -m verl.trainer.main_ppo \
    algorithm=ppo \
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
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_epochs=4 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.lora.enable=true \
    actor_rollout_ref.actor.lora.r=8 \
    actor_rollout_ref.actor.lora.lora_alpha=32 \
    actor_rollout_ref.actor.lora.lora_dropout=0.05 \
    "actor_rollout_ref.actor.lora.target_modules=[q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj]" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    critic.model.path="${MODEL_PATH}" \
    critic.optim.lr=1e-5 \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.fsdp_config.param_offload=false \
    \
    reward_model.enable=true \
    reward_model.model.path="${REWARD_MODEL_PATH}" \
    reward_model.model.dtype=bfloat16 \
    reward_model.model.fsdp_config.param_offload=true \
    \
    algorithm.kl_ctrl.kl_coef=0.15 \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    algorithm.adv_estimator=gae \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=verl_ppo_coig \
    trainer.experiment_name=qwen2.5-3b-ppo \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUTPUT_DIR}"
