"""
将 GSM8K_zh 数据集转换为 verl PPO 训练所需的 parquet 格式。

verl 要求每条数据包含以下字段：
  - prompt: list[dict]  — chat messages，格式为 [{"role": "user", "content": "..."}]
  - data_source: str    — 数据来源标识，reward_fn 用于路由
  - reward_model: dict  — {"ground_truth": "<数字答案>"}
  - extra_info: dict    — 附加信息

用法（从项目根目录）：
    python verl_math/prepare_data.py
    python verl_math/prepare_data.py --local_dataset_path /path/to/local/gsm8k_zh
"""

import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://huggingface.mirrors.cn"
import argparse
import re
import sys

import pandas as pd
from datasets import load_dataset


DATA_SOURCE = "openai/gsm8k"
INSTRUCTION = '请逐步思考，并在最后用 "####" 输出最终答案。'


def extract_solution(answer_str: str) -> str:
    """从 GSM8K 答案中提取 #### 后的数值。"""
    match = re.search(r"####\s*(\-?[\d\.\,]+)", answer_str)
    assert match is not None, f"Cannot extract solution from: {answer_str[:100]}"
    return match.group(1).replace(",", "").replace("$", "")


def convert_split(dataset, split: str) -> list[dict]:
    records = []
    for idx, item in enumerate(dataset):
        question = item["question"]
        answer_raw = item["answer"]
        ground_truth = extract_solution(answer_raw)

        prompt_text = question + " " + INSTRUCTION

        records.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": prompt_text}],
                "reward_model": {"ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    "answer": answer_raw,
                },
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K_zh data for verl PPO")
    parser.add_argument("--local_dataset_path", default=None, help="本地数据集路径")
    parser.add_argument("--output_dir", default="data/gsm8k_zh/verl_parquet")
    args = parser.parse_args()

    if args.local_dataset_path:
        dataset = load_dataset(args.local_dataset_path)
    else:
        dataset = load_dataset("meta-math/GSM8K_zh")

    train_data = convert_split(dataset["train"], "train")
    test_data = convert_split(dataset["test"], "test")

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.parquet")
    pd.DataFrame(train_data).to_parquet(train_path, index=False)
    print(f"[prepare_data] Saved {len(train_data)} train records -> {train_path}")

    test_path = os.path.join(args.output_dir, "test.parquet")
    pd.DataFrame(test_data).to_parquet(test_path, index=False)
    print(f"[prepare_data] Saved {len(test_data)} test records -> {test_path}")


if __name__ == "__main__":
    main()