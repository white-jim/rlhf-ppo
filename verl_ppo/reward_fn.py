"""
verl custom_reward_function: InternLM2-7B-reward 模型封装。

verl 的 reward manager 会调用 compute_score(data_source, solution_str, ground_truth, extra_info)。
我们在这里加载 InternLM2 reward model（单例模式），对 prompt+response 打分。

注意：
  - verl 会在多个 worker process 中并行调用此函数
  - 为节省内存，使用 num_workers=1 并在首次调用时加载模型（单例）
  - 模型加载到 GPU 0（假设 CUDA_VISIBLE_DEVICES 已设置）
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig


_REWARD_MODEL = None
_REWARD_TOKENIZER = None
_REWARD_CONFIG = None


def _load_reward_model():
    global _REWARD_MODEL, _REWARD_TOKENIZER, _REWARD_CONFIG
    if _REWARD_MODEL is not None:
        return _REWARD_MODEL, _REWARD_TOKENIZER

    model_path = os.getenv("REWARD_MODEL_PATH", "models/internlm2-7b-reward")
    dtype = getattr(torch, os.getenv("REWARD_MODEL_DTYPE", "bfloat16"))
    # Force cuda:0 since CUDA_VISIBLE_DEVICES already restricts GPU visibility
    device = "cuda:0"

    print(f"[reward_fn] Loading InternLM2 reward model from {model_path} (dtype={dtype}, device={device})")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    # 修复 rope_scaling 配置（与 trl_ppo/trainer.py 相同逻辑）
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        rs = dict(config.rope_scaling)
        if "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
        if "factor" not in rs:
            rs["factor"] = rs.get("scaling_factor") or rs.get("rope_scaling_factor") or 1.0
        valid = ["linear", "dynamic", "ntk-aware", "ntk_alpha", "yarn", "longrope"]
        config.rope_scaling = rs if rs.get("type", "").lower() in valid else None
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"

    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    _REWARD_MODEL = model
    _REWARD_TOKENIZER = tokenizer
    _REWARD_CONFIG = config
    print(f"[reward_fn] Reward model loaded successfully")
    return model, tokenizer


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> float:
    """
    verl custom_reward_function 接口。

    Args:
        data_source: 数据来源标识（如 "coig-cqia"）
        solution_str: 模型生成的 response 文本
        ground_truth: ground truth（neural RM 不使用）
        extra_info: 附加信息，包含 prompt 等

    Returns:
        float: reward score
    """
    model, tokenizer = _load_reward_model()

    # 从 extra_info 中获取 prompt（verl 会传入）
    # 如果没有，则使用空字符串（不应发生）
    prompt_text = extra_info.get("prompt", "")
    if isinstance(prompt_text, list):
        # verl 可能传入 chat messages list
        prompt_text = prompt_text[0]["content"] if prompt_text else ""

    # 构造 InternLM2 reward model 的输入格式
    template = os.getenv(
        "REWARD_PROMPT_TEMPLATE",
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
    )
    full_text = template.format(prompt=prompt_text, response=solution_str)

    # tokenize & forward
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        # 取最后一个非 pad token 的 hidden state
        attention_mask = inputs["attention_mask"]
        last_pos = attention_mask.sum(dim=1) - 1
        final_hidden = hidden_states[0, last_pos]  # [hidden_size]

        # InternLM2 reward head: linear(hidden_size, 1)
        # 尝试从模型中找到 reward_head / score / value_head / classifier
        score_head = None
        for attr in ("reward_head", "score", "value_head", "classifier"):
            head = getattr(model, attr, None)
            if head is not None:
                score_head = head
                break

        if score_head is None:
            # 如果找不到，手动创建一个 linear head（权重随机，仅用于测试）
            print("[reward_fn] WARNING: No reward head found, using random linear head")
            hidden_size = _REWARD_CONFIG.hidden_size
            score_head = torch.nn.Linear(hidden_size, 1, dtype=final_hidden.dtype, bias=False).to(model.device)

        reward_logits = score_head(final_hidden.unsqueeze(0))  # [1, 1]
        reward_score = reward_logits.item()

    return reward_score
