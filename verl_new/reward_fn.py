"""
Neural reward function — 使用 internlm2-7b-reward 对 (prompt, response) 打分。
verl 通过 `reward.custom_reward_function.path` 和 `name` 配置加载此函数。

函数签名必须为:
    async def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict,
        reward_router_address: str = None,
        reward_model_tokenizer = None,
        **kwargs
    ) -> dict | float:
"""

import os
from typing import Any, Optional

import torch
from transformers import AutoModel, AutoTokenizer


_model = None
_tokenizer = None
_device = None


def _load_model(model_path: str, dtype: str = "bfloat16"):
    global _model, _tokenizer, _device
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=getattr(torch, dtype),
    ).to(_device).eval()


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Any = None,
    **kwargs,
) -> dict | float:
    """
    使用 internlm2-7b-reward 模型计算奖励分数。

    Args:
        data_source: 数据来源标识
        solution_str: 模型生成的回复文本
        ground_truth: 真实答案（neural RM 不使用，但接口需要）
        extra_info: 附加信息，包含 prompt 等
        reward_router_address: 可选的奖励路由地址
        reward_model_tokenizer: 奖励模型的 tokenizer

    Returns:
        包含 'score' 键的字典，或直接返回浮点数分数
    """
    rm_path = os.environ.get("REWARD_MODEL_PATH", "models/internlm2-7b-reward")
    rm_dtype = os.environ.get("REWARD_MODEL_DTYPE", "bfloat16")

    _load_model(rm_path, rm_dtype)

    prompt_text = extra_info.get("prompt_text", "")

    prompt_template = (
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n{response}<|im_end|>"
    )
    text = prompt_template.format(prompt=prompt_text, response=solution_str)

    enc = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(_device)

    with torch.no_grad():
        score = _model(**enc).logits[0].item()

    return {"score": score}
