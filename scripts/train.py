import os
import argparse
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.ppo import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="PPO RLHF Training")
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model_config.yml",
        help="Path to model config file"
    )
    parser.add_argument(
        "--ppo_config",
        type=str,
        default="config/ppo_config.yml",
        help="Path to PPO config file"
    )
    parser.add_argument(
        "--eval_log_config",
        type=str,
        default="config/eval_log_config.yml",
        help="Path to eval/log config file"
    )
    
    args = parser.parse_args()
    
    trainer = PPOTrainer(
        model_config_path=args.model_config,
        ppo_config_path=args.ppo_config,
        eval_log_config_path=args.eval_log_config
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
