"""
TRL PPO quick start — entry point for the trl_ppo implementation.

Usage (from project root):
    python scripts/quick_start.py
    python scripts/quick_start.py --config trl_ppo/config.yml
"""

import os
import sys
import argparse

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
