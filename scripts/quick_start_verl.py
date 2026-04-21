"""
verl PPO 训练入口。

用法（从项目根目录）：
    # 第一步：准备数据（仅需运行一次）
    python scripts/quick_start_verl.py --prepare-data

    # 第二步：开始训练
    python scripts/quick_start_verl.py
    python scripts/quick_start_verl.py --config verl_ppo/config.yml

GPU 分配在 verl_ppo/config.yml 的 visible_gpus 字段中配置。
"""

import argparse
import datetime
import os
import subprocess
import sys
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_data(config_path: str):
    """调用 prepare_data.py 将 jsonl 转换为 parquet。"""
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verl_ppo", "prepare_data.py")
    subprocess.run([sys.executable, script, "--config", config_path], check=True)


def build_verl_args(cfg: dict, config_path: str, visible_gpus: str = None) -> list[str]:
    """根据 config.yml 构建 verl CLI 参数列表。"""
    ppo = cfg["ppo"]
    gen = cfg["generation"]
    lora = cfg["lora"]
    dataset = cfg["dataset"]
    rm = cfg["reward_model"]

    if visible_gpus is None:
        visible_gpus = cfg.get("visible_gpus", "0")

    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 转换相对路径为绝对路径
    def to_abs_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)

    model_path = to_abs_path(cfg["model_path"])
    rm_path = to_abs_path(rm["path"])
    parquet_dir = to_abs_path(dataset["parquet_dir"])
    train_parquet = os.path.join(parquet_dir, "train.parquet")
    val_parquet = os.path.join(parquet_dir, "val.parquet")
    output_dir = to_abs_path(cfg["output_dir"])

    # ppo_mini_batch_size = batch_size // num_mini_batches
    mini_batch_size = ppo["batch_size"] // ppo.get("num_mini_batches", 1)
    # micro batch size: 单卡内存有限，设为 1
    micro_batch = 1

    # LoRA target_modules: verl 接受逗号分隔字符串或列表
    lora_targets = ",".join(lora["target_modules"]) if lora["enable"] else ""

    args = [
        # data
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        f"data.train_batch_size={ppo['batch_size']}",
        f"data.max_prompt_length={cfg['max_prompt_length']}",
        f"data.max_response_length={gen['max_new_tokens']}",
        "data.return_raw_chat=True",
        "data.truncation=left",
        # actor
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.actor.optim.lr={ppo['learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch}",
        f"actor_rollout_ref.actor.ppo_epochs={ppo['ppo_epochs']}",
        f"actor_rollout_ref.actor.grad_clip={ppo['max_grad_norm']}",
        f"actor_rollout_ref.actor.clip_ratio={ppo['clip_range']}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(cfg.get('use_gradient_checkpointing', True)).lower()}",
        # LoRA
        f"actor_rollout_ref.model.lora_rank={lora['r'] if lora['enable'] else 0}",
        f"actor_rollout_ref.model.lora_alpha={lora['lora_alpha']}",
        f"actor_rollout_ref.model.target_modules=[{','.join(lora['target_modules'])}]",
        # rollout (vllm)
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.temperature={gen['temperature']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_batch}",
        # ref (fused into actor via LoRA)
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_batch}",
        # critic
        f"critic.model.path={model_path}",
        f"critic.optim.lr={ppo.get('critic_lr', 1e-5)}",
        f"critic.ppo_micro_batch_size_per_gpu={micro_batch}",
        f"critic.model.enable_gradient_checkpointing={str(cfg.get('use_gradient_checkpointing', True)).lower()}",
        # algorithm
        "algorithm.adv_estimator=gae",
        f"algorithm.gamma={ppo['gamma']}",
        f"algorithm.lam={ppo['lam']}",
        "algorithm.use_kl_in_reward=True",
        f"algorithm.kl_ctrl.kl_coef={ppo['kl_coef']}",
        "algorithm.kl_ctrl.type=fixed",
        # reward: vLLM reward model (InternLM2-7B-reward)
        "reward.num_workers=1",
        "reward.reward_model.enable=True",
        "reward.reward_model.enable_resource_pool=False",
        f"reward.reward_model.model_path={rm_path}",
        "reward.reward_model.rollout.name=vllm",
        "reward.reward_model.rollout.gpu_memory_utilization=0.5",
        "reward.reward_model.rollout.tensor_model_parallel_size=1",
        "reward.reward_model.rollout.free_cache_engine=True",
        "actor_rollout_ref.model.trust_remote_code=True",
        "critic.model.trust_remote_code=True",
        # trainer
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.total_epochs={cfg.get('num_train_epochs', 1)}",
        f"trainer.save_freq={cfg.get('save_freq', 200)}",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.logger=console",
        f"trainer.default_local_dir={output_dir}",
        "trainer.project_name=rlhf_ppo",
        "trainer.experiment_name=verl_ppo",
        # Ray runtime env - pass CUDA_VISIBLE_DEVICES and HF cache to workers
        f"+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES='{visible_gpus}'",
        "+ray_kwargs.ray_init.runtime_env.env_vars.HF_MODULES_CACHE='/tmp/hf_modules_cache'",
        "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_CACHE='/tmp/hf_cache'",
    ]
    return args


def main():
    parser = argparse.ArgumentParser(description="verl PPO RLHF Quick Start")
    parser.add_argument("--config", default="verl_ppo/config.yml")
    parser.add_argument("--prepare-data", action="store_true", help="仅运行数据预处理后退出")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 设置 CUDA_VISIBLE_DEVICES（必须在 import torch 之前，这里通过子进程传递）
    visible_gpus = cfg.get("visible_gpus")
    env = os.environ.copy()
    if visible_gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_gpus)
        print(f"[quick_start_verl] CUDA_VISIBLE_DEVICES={visible_gpus}")

    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 转换相对路径为绝对路径
    def to_abs_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)

    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_MODULES_CACHE"] = "/tmp/hf_modules_cache"
    env["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
    # reward_fn.py 需要知道 reward model 路径
    env["REWARD_MODEL_PATH"] = to_abs_path(cfg["reward_model"]["path"])
    env["REWARD_MODEL_DTYPE"] = cfg["reward_model"].get("dtype", "bfloat16")
    env["REWARD_PROMPT_TEMPLATE"] = cfg["reward_model"].get(
        "prompt_template",
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
    )

    if args.prepare_data:
        prepare_data(args.config)
        return

    # 检查 parquet 是否存在，不存在则自动准备
    parquet_dir = to_abs_path(cfg["dataset"]["parquet_dir"])
    train_parquet = os.path.join(parquet_dir, "train.parquet")
    if not os.path.exists(train_parquet):
        print("[quick_start_verl] Parquet not found, running data preparation...")
        prepare_data(args.config)

    verl_args = build_verl_args(cfg, args.config, visible_gpus)

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + verl_args

    output_dir = to_abs_path(cfg.get("output_dir", "outputs/verl_ppo"))
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")

    print("[quick_start_verl] Launching verl PPO trainer...")
    print(f"[quick_start_verl] Log file: {log_path}")
    print("Command:", " ".join(cmd))

    returncode = _run_tee(cmd, env, log_path)
    sys.exit(returncode)


def _run_tee(cmd: list, env: dict, log_path: str) -> int:
    """Run cmd, writing output to both terminal and log_path in real time."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        process.wait()
    return process.returncode


if __name__ == "__main__":
    main()
