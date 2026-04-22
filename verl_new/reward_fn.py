"""
Neural reward function — 在 verl reward worker 进程里直接加载 internlm2-7b-reward 打分。

verl 通过 `reward.custom_reward_function.path` 和 `name` 配置加载此函数。
reward worker 是独立的 Ray actor，与训练进程隔离，可以安全地在此加载 reward model。

函数签名必须为:
    def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict,
        **kwargs
    ) -> dict | float
"""

import os
from typing import Any, Optional

import torch
from transformers import AutoModel, AutoTokenizer


PROMPT_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n{response}<|im_end|>"
)

_model = None
_tokenizer = None
_device = None


def _load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return
    rm_path = os.environ.get("REWARD_MODEL_PATH", "models/internlm2-7b-reward")
    rm_dtype = os.environ.get("REWARD_MODEL_DTYPE", "bfloat16")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        rm_path,
        trust_remote_code=True,
        torch_dtype=getattr(torch, rm_dtype),
        device_map={"": _device},
    ).eval()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
) -> dict:
    """使用 internlm2-7b-reward 计算奖励分数。"""
    _load_model()

    prompt_text = extra_info.get("prompt_text", "")
    text = PROMPT_TEMPLATE.format(prompt=prompt_text, response=solution_str)

    enc = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(_device)

    with torch.no_grad():
        score = _model(**enc).logits[0].item()

    return {"score": score}
