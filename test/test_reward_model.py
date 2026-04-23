"""
测试 reward model 的打分行为。

分别测试：
1. 直接用 transformers 加载模型打分（AutoModelForSequenceClassification）
2. 通过 vLLM /classify 接口打分（模拟 verl 的调用方式）
3. 对不同质量的回复打分，观察区分度

用法：
    python test/test_reward_model.py --model_path models/wildreward-8b
    python test/test_reward_model.py --model_path models/skywork-reward-v2-qwen3-4b
    python test/test_reward_model.py --model_path models/wildreward-8b --method vllm
"""

import argparse
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ── 测试数据 ──────────────────────────────────────────────────────────────────
# 中文指令 + 不同质量的回复
TEST_CASES = [
    {
        "prompt": "请介绍一下人工智能的发展历史。",
        "responses": [
            # 高质量
            "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支，旨在研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。\n\n人工智能的发展历史可以追溯到20世纪50年代：\n\n1. **萌芽期（1950s）**：1950年，艾伦·图灵提出了著名的'图灵测试'，为人工智能奠定了理论基础。1956年，达特茅斯会议正式确立了'人工智能'这一术语，标志着AI作为独立学科的诞生。\n\n2. **第一次繁荣期（1950s-1960s）**：这一时期出现了符号主义方法，研究者们对AI前景非常乐观，认为很快就能实现人类级别的智能。\n\n3. **第一次低谷期（1970s）**：由于计算能力有限和理论瓶颈，AI研究进展缓慢，资金大幅削减，进入所谓的'AI寒冬'。\n\n4. **专家系统时代（1980s）**：基于规则的专家系统在特定领域取得成功，AI迎来了第二次繁荣。\n\n5. **统计学习时代（1990s-2000s）**：随着数据量增加和计算能力提升，机器学习方法（特别是SVM、随机森林等）开始占据主导地位。\n\n6. **深度学习革命（2010s至今）**：2012年AlexNet在ImageNet竞赛中的突破性表现开启了深度学习时代。随后，AlphaGo击败围棋世界冠军、GPT系列大语言模型的出现，将AI推向了新的高度。\n\n当前，以大语言模型为代表的生成式AI正在深刻改变各行各业，人工智能的发展进入了前所未有的加速期。",
            # 中等质量
            "人工智能的发展历史大致如下：\n\n20世纪50年代，图灵提出了图灵测试的概念。1956年的达特茅斯会议标志着人工智能正式诞生。\n\n60-70年代，AI经历了第一次繁荣和低谷，研究者对前景过于乐观，后来遇到瓶颈。\n\n80年代，专家系统流行了一阵。\n\n90年代到2000年代，机器学习方法开始兴起。\n\n2010年代，深度学习取得了重大突破，特别是2012年的AlexNet。之后AlphaGo和大语言模型的出现让AI更加火热。\n\n现在AI正在快速发展中。",
            # 低质量但相关
            "人工智能就是让电脑像人一样思考。它大概是从1950年代开始的，中间经历了一些起伏。现在深度学习很火，大模型也很厉害。",
            # 极短回复
            "AI从1950年代开始发展，现在很火。",
            # 无关回复
            "今天天气不错，我打算去公园散步。顺便买点水果回来。",
            # 乱码
            "asdfghjkl 12345 qwerty",
        ],
    },
    {
        "prompt": "用Python写一个快速排序算法。",
        "responses": [
            # 高质量
            '```python\ndef quicksort(arr):\n    """快速排序算法实现"""\n    if len(arr) <= 1:\n        return arr\n    \n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    \n    return quicksort(left) + middle + quicksort(right)\n\n\n# 使用示例\ndata = [3, 6, 8, 10, 1, 2, 1]\nprint(quicksort(data))  # 输出: [1, 1, 2, 3, 6, 8, 10]\n```\n\n快速排序的核心思想是分治法：\n1. 选择一个基准元素（pivot）\n2. 将数组分为小于、等于和大于基准的三部分\n3. 递归地对左右两部分排序\n\n时间复杂度：平均 O(n log n)，最坏 O(n²)\n空间复杂度：O(n)（上述实现）或 O(log n)（原地排序版本）',
            # 中等质量
            '```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n```',
            # 低质量
            "快速排序就是选一个基准，然后分成两半递归排序。代码我记不太清了，大概就是递归调用。",
            # 极短
            "用sort()函数就行。",
        ],
    },
]


def test_transformers(model_path: str):
    """用 transformers 直接加载模型打分。"""
    print(f"\n{'='*60}")
    print(f"方法: transformers (AutoModelForSequenceClassification)")
    print(f"模型: {model_path}")
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

    for case_idx, case in enumerate(TEST_CASES):
        print(f"\n--- 测试用例 {case_idx+1}: {case['prompt'][:30]}... ---")
        for resp_idx, response in enumerate(case["responses"]):
            messages = [
                {"role": "user", "content": case["prompt"]},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

            # 移除 bos token（与 verl 行为一致）
            if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
                text = text[len(tokenizer.bos_token):]

            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                logits = model(**enc).logits[0]
                probs = torch.softmax(logits, dim=-1)

            # verl 取 probs[-1]（最后一个类别的概率）
            verl_score = probs[-1].item()

            label_str = ""
            if num_labels <= 10:
                parts = []
                for i in range(num_labels):
                    lbl = id2label.get(i, str(i))
                    parts.append(f"{lbl}={probs[i].item():.4f}")
                label_str = " | ".join(parts)

            resp_preview = response[:50].replace("\n", " ") + ("..." if len(response) > 50 else "")
            print(f"  [{resp_idx}] verl_score={verl_score:.4f}  logits={logits.tolist()[:5]}...  probs=[{label_str}]")
            print(f"       回复: {resp_preview}")


def test_vllm(model_path: str, port: int = 8000):
    """通过 vLLM /classify 接口打分（模拟 verl 的调用方式）。"""
    import subprocess
    import time

    import requests

    print(f"\n{'='*60}")
    print(f"方法: vLLM /classify 接口")
    print(f"模型: {model_path}")
    print(f"{'='*60}")

    # 启动 vLLM server
    print("启动 vLLM server...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--gpu-memory-utilization", "0.5",
            "--dtype", "bfloat16",
            "--max-model-len", "2048",
            "--trust-remote-code",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 等待 server 就绪
    for _ in range(120):
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("ERROR: vLLM server 启动超时")
        proc.terminate()
        return

    print("vLLM server 已就绪")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        for case_idx, case in enumerate(TEST_CASES):
            print(f"\n--- 测试用例 {case_idx+1}: {case['prompt'][:30]}... ---")
            for resp_idx, response in enumerate(case["responses"]):
                messages = [
                    {"role": "user", "content": case["prompt"]},
                    {"role": "assistant", "content": response},
                ]
                text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

                if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
                    text = text[len(tokenizer.bos_token):]

                # 调用 /classify（与 verl 完全一致）
                payload = {
                    "model": model_path,
                    "input": text,
                    "use_activation": False,
                }
                r = requests.post(f"http://localhost:{port}/classify", json=payload, timeout=30)
                output = r.json()

                # verl 的取法: output["data"][-1]["probs"][-1]
                probs = output["data"][-1]["probs"]
                verl_score = probs[-1]

                resp_preview = response[:50].replace("\n", " ") + ("..." if len(response) > 50 else "")
                print(f"  [{resp_idx}] verl_score={verl_score:.4f}  probs={[f'{p:.4f}' for p in probs]}")
                print(f"       回复: {resp_preview}")
    finally:
        print("\n关闭 vLLM server...")
        proc.terminate()
        proc.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description="Test reward model scoring behavior")
    parser.add_argument("--model_path", required=True, help="Reward model path")
    parser.add_argument("--method", default="transformers", choices=["transformers", "vllm", "both"],
                        help="Test method: transformers, vllm, or both")
    parser.add_argument("--port", default=8000, type=int, help="vLLM server port")
    args = parser.parse_args()

    if args.method in ("transformers", "both"):
        test_transformers(args.model_path)

    if args.method in ("vllm", "both"):
        test_vllm(args.model_path, args.port)

    print(f"\n{'='*60}")
    print("说明：verl_score = probs[-1]，即最后一个类别的概率")
    print("对于 5 档满意度模型，probs[-1] 是'最高分'类别的概率")
    print("如果所有回复的 verl_score 都很接近，说明区分度差")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()