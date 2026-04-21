"""
将 COIG-CQIA jsonl 数据集转换为 verl PPO 训练所需的 parquet 格式。

verl 要求每条数据包含以下字段：
  - prompt: list[dict]  — chat messages，格式为 [{"role": "user", "content": "..."}]
  - data_source: str    — 数据来源标识，reward_fn 用于路由
  - reward_model: dict  — {"ground_truth": ""}（neural RM 不需要 ground_truth）
  - extra_info: dict    — 附加信息

用法（从项目根目录）：
    python verl_ppo/prepare_data.py
    python verl_ppo/prepare_data.py --config verl_ppo/config.yml
"""

import argparse
import json
import os
import sys

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def convert_jsonl_to_parquet(jsonl_path: str, output_dir: str, split: str = "train") -> str:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt_text = item.get("prompt", item.get("instruction", item.get("input", "")))
            records.append(
                {
                    "data_source": "coig-cqia",
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "reward_model": {"ground_truth": ""},
                    "extra_info": {"index": idx, "split": split},
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split}.parquet")
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    print(f"[prepare_data] Saved {len(records)} records → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare COIG-CQIA data for verl PPO")
    parser.add_argument("--config", default="verl_ppo/config.yml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_jsonl = cfg["dataset"]["train_path"]
    parquet_dir = cfg["dataset"]["parquet_dir"]

    if not os.path.exists(train_jsonl):
        print(f"[prepare_data] ERROR: train jsonl not found: {train_jsonl}", file=sys.stderr)
        sys.exit(1)

    convert_jsonl_to_parquet(train_jsonl, parquet_dir, split="train")
    # val set: 取 train 的最后 200 条作为验证集（如无单独 val 文件）
    val_jsonl = cfg["dataset"].get("val_path")
    if val_jsonl and os.path.exists(val_jsonl):
        convert_jsonl_to_parquet(val_jsonl, parquet_dir, split="val")
    else:
        _make_val_from_train(train_jsonl, parquet_dir, n=200)


def _make_val_from_train(train_jsonl: str, output_dir: str, n: int = 200):
    records = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines[-n:]):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        prompt_text = item.get("prompt", item.get("instruction", item.get("input", "")))
        records.append(
            {
                "data_source": "coig-cqia",
                "prompt": [{"role": "user", "content": prompt_text}],
                "reward_model": {"ground_truth": ""},
                "extra_info": {"index": idx, "split": "val"},
            }
        )
    out_path = os.path.join(output_dir, "val.parquet")
    pd.DataFrame(records).to_parquet(out_path, index=False)
    print(f"[prepare_data] Saved {len(records)} val records (from train tail) → {out_path}")


if __name__ == "__main__":
    main()
