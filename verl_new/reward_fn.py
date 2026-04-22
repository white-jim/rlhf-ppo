"""
Neural reward function — 通过 verl 管理的 internlm2-7b-reward vllm 服务器打分。

verl 通过 `reward.custom_reward_function.path` 和 `name` 配置加载此函数。
当 `reward.reward_model.enable=true` 时，verl 会启动 reward model 的 vllm 服务，
并将服务地址通过 `reward_router_address` 传入。

函数签名必须为:
    def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict,
        reward_router_address: str = None,
        reward_model_tokenizer = None,
        **kwargs
    ) -> dict | float
"""

import os
from typing import Any, Optional


PROMPT_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n{response}<|im_end|>"
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Any = None,
    **kwargs,
) -> dict:
    """
    使用 internlm2-7b-reward 模型计算奖励分数。

    当 reward_router_address 不为 None 时，通过 verl 管理的 vllm 服务器打分；
    否则回退到本地加载模型（仅用于调试，生产环境应始终通过服务器）。
    """
    prompt_text = extra_info.get("prompt_text", "")
    text = PROMPT_TEMPLATE.format(prompt=prompt_text, response=solution_str)

    if reward_router_address is not None:
        score = _score_via_server(text, reward_router_address, reward_model_tokenizer)
    else:
        score = _score_locally(text)

    return {"score": score}


def _score_via_server(text: str, address: str, tokenizer) -> float:
    """通过 verl 管理的 reward model vllm 服务器获取分数。"""
    import requests

    url = address.rstrip("/") + "/v1/completions"
    payload = {
        "model": os.environ.get("REWARD_MODEL_PATH", "models/internlm2-7b-reward"),
        "prompt": text,
        "max_tokens": 1,
        "logprobs": 1,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # internlm2-reward 输出 logits[0] 作为分数
    score = data["choices"][0]["logprobs"]["token_logprobs"][0]
    return float(score)


_local_model = None
_local_tokenizer = None
_local_device = None


def _score_locally(text: str) -> float:
    """回退：本地加载 reward model 打分（仅调试用）。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    global _local_model, _local_tokenizer, _local_device

    if _local_model is None:
        rm_path = os.environ.get("REWARD_MODEL_PATH", "models/internlm2-7b-reward")
        rm_dtype = os.environ.get("REWARD_MODEL_DTYPE", "bfloat16")
        _local_device = "cuda" if torch.cuda.is_available() else "cpu"
        _local_tokenizer = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True)
        _local_model = AutoModel.from_pretrained(
            rm_path,
            trust_remote_code=True,
            torch_dtype=getattr(torch, rm_dtype),
        ).to(_local_device).eval()

    enc = _local_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(_local_device)

    import torch
    with torch.no_grad():
        score = _local_model(**enc).logits[0].item()

    return score
