"""
TRL PPO quick start — entry point for the trl_ppo implementation.

Usage (from project root):
    python scripts/quick_start.py
    python scripts/quick_start.py --config trl_ppo/config.yml

GPU 分配在 trl_ppo/config.yml 的 visible_gpus 字段中配置。
"""

import os
import sys
import argparse
import yaml

# 必须在 import torch / trl 之前读取 config 并设置 CUDA_VISIBLE_DEVICES
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--config", type=str, default="trl_ppo/config.yml")
_pre_args, _ = _pre_parser.parse_known_args()

with open(_pre_args.config, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_visible_gpus = _cfg.get("visible_gpus")
if _visible_gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_visible_gpus)
    print(f"[quick_start] CUDA_VISIBLE_DEVICES={_visible_gpus}")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trl_ppo.trainer import TRLPPOTrainer


def main():
    parser = argparse.ArgumentParser(description="TRL PPO RLHF Quick Start")
    parser.add_argument(
        "--config",
        type=str,
        default="trl_ppo/config.yml",
        help="Path to trl_ppo config file (default: trl_ppo/config.yml)",
    )
    args = parser.parse_args()

    trainer = TRLPPOTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
