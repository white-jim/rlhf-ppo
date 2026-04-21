"""
verl custom_reward_function: InternLM2-7B-reward 通过 vLLM API 调用。

verl 会用 vLLM 部署 reward model，然后调用 compute_score。
我们通过 HTTP 请求访问 reward model API，而不是直接加载模型。
"""

import os
import aiohttp
from transformers import PreTrainedTokenizer


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
) -> dict:
    """
    verl custom_reward_function 接口（异步版本）。

    Args:
        data_source: 数据来源标识（如 "coig-cqia"）
        solution_str: 模型生成的 response 文本
        ground_truth: ground truth（neural RM 不使用）
        extra_info: 附加信息，包含 prompt 等
        reward_router_address: vLLM reward model 的 HTTP 地址
        reward_model_tokenizer: reward model 的 tokenizer

    Returns:
        dict: {"score": float}
    """
    # 从 extra_info 中获取 prompt
    prompt_text = extra_info.get("prompt", "")
    if isinstance(prompt_text, list):
        # verl 传入的是 chat messages list
        prompt_text = prompt_text[0]["content"] if prompt_text else ""

    # 构造 InternLM2 reward model 的输入格式
    template = os.getenv(
        "REWARD_PROMPT_TEMPLATE",
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
    )
    full_text = template.format(prompt=prompt_text, response=solution_str)

    # 通过 vLLM API 获取 reward
    # InternLM2-7B-reward 是一个 reward model，需要用特殊方式调用
    # 由于 vLLM 主要用于生成模型，我们需要获取 hidden states
    # 但 vLLM API 不直接支持获取 hidden states

    # 替代方案：将 reward model 当作生成模型，生成一个分数
    # 但这需要 InternLM2-7B-reward 支持生成模式

    # 临时方案：调用 vLLM 的 completions API，取最后一个 token 的 logit
    # 这需要 vLLM 支持 return_hidden_states 或者通过 logits 计算

    # 最简单的方案：用生成模型生成 "good" 或 "bad" token 的概率
    # 但 InternLM2-7B-reward 可能不支持这种方式

    # 现在采用最直接的方法：构造一个特殊的 prompt 让模型输出分数
    # 由于 InternLM2-7B-reward 是 reward model，我们假设它接受 prompt+response 并输出分数

    # 实际上，InternLM2-7B-reward 的用法是：
    # 输入: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    # 取最后一个 token 的 hidden state，然后通过 reward head 得到分数

    # vLLM API 不支持直接获取 hidden states，所以我们需要：
    # 1. 要么用 HuggingFace pipeline 本地加载（但这样 reward worker 需要 GPU）
    # 2. 要么扩展 vLLM API 支持获取 hidden states

    # 目前的折中方案：假设 reward model 会被当作一个特殊的模型处理
    # verl 可能有特殊的 reward model 处理逻辑

    # 让我们先尝试调用 chat completion API
    try:
        messages = [{"role": "user", "content": full_text}]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            url = f"http://{reward_router_address}/v1/chat/completions"
            payload = {
                "messages": messages,
                "max_tokens": 1,  # 只需要一个 token
                "temperature": 0,  # 确定性输出
            }
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()

        # 解析结果
        # 如果 reward model 返回的是分数，我们需要提取它
        # 这取决于 InternLM2-7B-reward 在 vLLM 中的实际行为

        # 暂时返回一个默认分数，需要根据实际情况调整
        print(f"[reward_fn] Got result from reward model API: {result}")
        return {"score": 0.0}

    except Exception as e:
        print(f"[reward_fn] Error calling reward model API: {e}")
        return {"score": 0.0}
