"""
使用 COIG-CQIA 数据集中的真实样例测试 reward model 打分。

用法：
    python test/test_reward_on_data.py --model_path models/wildreward-8b --data_path data/coig-cqia/processed/train.jsonl --num_samples 20
    python test/test_reward_on_data.py --model_path models/skywork-reward-v2-qwen3-4b --data_path data/coig-cqia/processed/train.jsonl --num_samples 20
"""

import argparse
import json
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_samples(data_path: str, num_samples: int = 20) -> list[dict]:
    """从 jsonl 中随机采样。"""
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    sampled = random.sample(lines, min(num_samples, len(lines)))
    return [json.loads(line) for line in sampled]


def test_on_data(model_path: str, data_path: str, num_samples: int):
    print(f"\n{'='*60}")
    print(f"模型: {model_path}")
    print(f"数据: {data_path}")
    print(f"采样数: {num_samples}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    num_labels = model.config.num_labels
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
    print(f"num_labels: {num_labels}, id2label: {id2label}")

    samples = load_samples(data_path, num_samples)

    scores = []
    all_probs = []

    for idx, item in enumerate(samples):
        prompt_text = item.get("prompt", item.get("instruction", item.get("input", "")))
        response_text = item.get("response", item.get("output", item.get("chosen", "")))

        if not response_text:
            print(f"  [{idx}] 跳过（无回复）")
            continue

        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1)

        verl_score = probs[-1].item()
        scores.append(verl_score)
        all_probs.append(probs.tolist())

        prompt_preview = prompt_text[:40].replace("\n", " ") + "..."
        resp_preview = response_text[:40].replace("\n", " ") + "..."
        print(f"  [{idx}] verl_score={verl_score:.4f}  prompt={prompt_preview}  resp={resp_preview}")

    if scores:
        import numpy as np
        arr = np.array(scores)
        print(f"\n--- 统计 ---")
        print(f"  样本数: {len(scores)}")
        print(f"  verl_score 均值: {arr.mean():.4f}")
        print(f"  verl_score 标准差: {arr.std():.4f}")
        print(f"  verl_score 范围: [{arr.min():.4f}, {arr.max():.4f}]")

        # 各类别概率分布
        probs_arr = np.array(all_probs)
        print(f"\n  各类别概率均值:")
        for i in range(num_labels):
            lbl = id2label.get(i, str(i))
            print(f"    类别 {i} ({lbl}): 均值={probs_arr[:, i].mean():.4f}  标准差={probs_arr[:, i].std():.4f}")

        print(f"\n  说明: verl 取 probs[-1] 作为分数")
        print(f"  如果 probs[-1] 的标准差很小，说明该类别区分度差")
        print(f"  如果某个中间类别概率最高，说明 verl 的取法可能不对")


def main():
    parser = argparse.ArgumentParser(description="Test reward model on real dataset")
    parser.add_argument("--model_path", required=True, help="Reward model path")
    parser.add_argument("--data_path", required=True, help="Data jsonl path")
    parser.add_argument("--num_samples", default=20, type=int, help="Number of samples to test")
    args = parser.parse_args()

    test_on_data(args.model_path, args.data_path, args.num_samples)


if __name__ == "__main__":
    main()