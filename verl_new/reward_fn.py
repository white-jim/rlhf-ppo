"""
Neural reward function — 使用 internlm2-7b-reward 对 (prompt, response) 打分。
verl 通过 reward_fn(data_source, solution_str, ground_truth, extra_info) 调用。
"""

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


class RewardManager:
    """
    verl 要求 reward_fn 是一个可调用对象，签名为:
        __call__(data: DataProto) -> torch.Tensor  (shape: [batch_size])
    """

    def __init__(self, tokenizer, num_examine: int = 0):
        # tokenizer 由 verl 传入(actor tokenizer),这里不用,但接口需要保留
        self.tokenizer = tokenizer
        # 延迟加载,路径从环境变量读取
        import os
        self.rm_path = os.environ.get("REWARD_MODEL_PATH", "models/internlm2-7b-reward")
        self.rm_dtype = os.environ.get("REWARD_MODEL_DTYPE", "bfloat16")
        self.prompt_template = (
            "<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n{response}<|im_end|>"
        )

    def __call__(self, data) -> torch.Tensor:
        """
        data: verl DataProto
          - data.non_tensor_batch["raw_prompt_ids_rmpad"] or decoded via tokenizer
          - data.batch["responses"] token ids
        """
        _load_model(self.rm_path, self.rm_dtype)

        # 解码 prompt + response
        input_ids = data.batch["input_ids"]          # [B, seq_len]
        attention_mask = data.batch["attention_mask"]
        responses = data.batch["responses"]           # [B, resp_len]

        prompts_text = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        responses_text = self.tokenizer.batch_decode(
            responses, skip_special_tokens=True
        )

        scores = []
        with torch.no_grad():
            for prompt, response in zip(prompts_text, responses_text):
                text = self.prompt_template.format(prompt=prompt, response=response)
                enc = _tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(_device)
                score = _model(**enc).logits[0].item()  # internlm2-reward 输出标量
                scores.append(score)

        return torch.tensor(scores, dtype=torch.float32)
