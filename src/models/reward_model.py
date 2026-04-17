import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from contextlib import contextmanager


class RewardModel(nn.Module):
    def __init__(self, reward_config):
        super().__init__()
        self.reward_config = reward_config
        self.prompt_template = reward_config["prompt_template"]

        # Load config and fix compatibility issues
        config = AutoConfig.from_pretrained(
            reward_config["path"],
            trust_remote_code=True,
            local_files_only=True
        )

        # Fix rope_scaling compatibility issues for InternLM2
        # The InternLM2 custom modeling code expects specific keys that may differ
        # from newer transformers versions
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            rope_scaling = dict(config.rope_scaling)  # Make a mutable copy

            # Ensure "type" key exists (map from "rope_type" if needed)
            if "rope_type" in rope_scaling and "type" not in rope_scaling:
                rope_scaling["type"] = rope_scaling["rope_type"]

            # Ensure "factor" key exists
            if "factor" not in rope_scaling:
                # Try common alternative key names
                factor = rope_scaling.get("scaling_factor") or rope_scaling.get("rope_scaling_factor") or 1.0
                rope_scaling["factor"] = factor

            # InternLM2 doesn't recognize "default" as a valid scaling type
            # Valid types are typically: "linear", "dynamic", "ntk-aware", "ntk_alpha", etc.
            # If type is "default" or unknown, disable rope_scaling
            scaling_type = rope_scaling.get("type", "")
            valid_scaling_types = ["linear", "dynamic", "ntk-aware", "ntk_alpha", "yarn", "longrope"]
            if scaling_type.lower() not in valid_scaling_types:
                # Disable rope_scaling - model will use default (no scaling) behavior
                config.rope_scaling = None
            else:
                config.rope_scaling = rope_scaling

        # Force eager attention implementation to avoid flash attention issues
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"

        self.model = AutoModel.from_pretrained(
            reward_config["path"],
            config=config,
            torch_dtype=getattr(torch, reward_config["dtype"]),
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        # Try loading tokenizer with use_fast=False first to avoid tiktoken/sentencepiece dependency
        # If that fails, try with use_fast=True (may require additional packages)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                reward_config["path"],
                padding_side="right",
                truncation_side="right",
                trust_remote_code=True,
                local_files_only=True,
                use_fast=False  # Use slow tokenizer to avoid conversion issues
            )
        except Exception as e:
            print(f"Warning: Failed to load slow tokenizer: {e}")
            print("Trying fast tokenizer (may require tiktoken/sentencepiece)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                reward_config["path"],
                padding_side="right",
                truncation_side="right",
                trust_remote_code=True,
                local_files_only=True
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.model.requires_grad_(False)

        # Determine output attribute name (could be 'logits' or 'scores' depending on model)
        # InternLM2-Reward typically outputs 'scores' for sequence classification
        self._output_attr = self._detect_output_attr()

    def _detect_output_attr(self):
        """Detect the output attribute name for the reward model."""
        # Try a dummy forward pass to check output structure
        with torch.no_grad():
            dummy_input = self.tokenizer("test", return_tensors="pt")
            dummy_input = {k: v.to(self._get_device()) for k, v in dummy_input.items()}
            try:
                outputs = self.model(**dummy_input)
                if hasattr(outputs, "logits"):
                    return "logits"
                elif hasattr(outputs, "scores"):
                    return "scores"
                elif hasattr(outputs, "end_scores"):
                    return "end_scores"
                else:
                    # Check if output is a tensor directly
                    if isinstance(outputs, torch.Tensor):
                        return "direct"
                    # Fallback to logits
                    return "logits"
            except Exception:
                # Default fallback
                return "logits"

    def _get_device(self):
        """Get the device where the model resides."""
        try:
            # For models with device_map="auto", get the first device
            if hasattr(self.model, "hf_device_map"):
                devices = list(self.model.hf_device_map.values())
                if devices:
                    return torch.device(devices[0])
            # Fallback: get device from first parameter
            return next(self.model.parameters()).device
        except (StopIteration, Exception):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @contextmanager
    def inference_mode(self):
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=self.model.dtype):
                yield

    def format_input(self, prompt, response):
        return self.prompt_template.format(prompt=prompt, response=response)

    def forward(self, input_ids, attention_mask=None):
        with self.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Handle different output formats
            if self._output_attr == "direct":
                rewards = outputs.squeeze(-1)
            elif self._output_attr == "end_scores":
                rewards = outputs.end_scores.squeeze(-1)
            else:
                rewards = getattr(outputs, self._output_attr).squeeze(-1)
        return rewards

    def compute_reward(self, prompts, responses):
        texts = [self.format_input(p, r) for p, r in zip(prompts, responses)]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        device = self._get_device()
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        rewards = self.forward(input_ids, attention_mask)
        return rewards
