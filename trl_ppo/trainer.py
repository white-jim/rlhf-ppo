import json
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
)
from peft import LoraConfig
from trl.experimental.ppo import PPOConfig, PPOTrainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# InternLM2 reward model wrapped to match TRL's get_reward() interface:
#   lm_backbone = getattr(model, model.base_model_prefix)
#   output = lm_backbone(input_ids, ..., output_hidden_states=True)
#   reward_logits = model.score(output.hidden_states[-1])
# ---------------------------------------------------------------------------

class _FakeOutput:
    """Minimal stand-in for a transformer output that only carries hidden_states."""
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _RetokenizingBackbone(nn.Module):
    """Decode policy token IDs → text → re-encode with reward tokenizer → run reward backbone.

    TRL's get_reward() indexes reward_logits at sequence_lengths computed from the *policy*
    sequence.  To make that work after re-tokenization (which changes sequence length), we
    take the final hidden state of the reward sequence and broadcast it to every position of
    the policy sequence — so the indexed value is always the correct reward regardless of
    which position sequence_lengths points to.
    """

    def __init__(self, backbone: nn.Module, policy_tokenizer, reward_tokenizer):
        super().__init__()
        self._backbone = backbone
        self._policy_tok = policy_tokenizer
        self._reward_tok = reward_tokenizer

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        model_device = next(self._backbone.parameters()).device
        batch_size, policy_seq_len = input_ids.shape
        src_device = input_ids.device

        # 1. Decode policy token IDs → plain text
        texts = self._policy_tok.batch_decode(input_ids.cpu(), skip_special_tokens=True)

        # 2. Re-encode with reward tokenizer
        enc = self._reward_tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=policy_seq_len,
        )
        reward_input_ids = enc["input_ids"].to(model_device)
        reward_attn_mask = enc["attention_mask"].to(model_device)

        # 3. Run reward backbone
        output = self._backbone(
            input_ids=reward_input_ids,
            attention_mask=reward_attn_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        # 4. Extract the last non-pad hidden state for each sequence → [batch, hidden_size]
        last_pos = reward_attn_mask.sum(dim=1) - 1
        final_hidden = output.hidden_states[-1][
            torch.arange(batch_size, device=model_device), last_pos
        ]

        # 5. Broadcast to policy sequence length so TRL's sequence_lengths indexing is safe
        expanded = final_hidden.unsqueeze(1).expand(-1, policy_seq_len, -1).to(src_device)

        return _FakeOutput(hidden_states=(expanded,))


class InternLM2RewardWrapper(nn.Module):
    base_model_prefix = "model"

    def __init__(self, reward_cfg: dict, policy_tokenizer):
        super().__init__()
        dtype = getattr(torch, reward_cfg.get("dtype", "bfloat16"))
        device = reward_cfg.get("device", "cuda")

        config = AutoConfig.from_pretrained(
            reward_cfg["path"], trust_remote_code=True, local_files_only=True
        )
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

        _backbone = AutoModel.from_pretrained(
            reward_cfg["path"],
            config=config,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
        _backbone.eval()
        _backbone.requires_grad_(False)

        # Build score head; try to copy weights from the original reward head
        hidden_size = config.hidden_size
        self._score = nn.Linear(hidden_size, 1, dtype=dtype, bias=False)
        for attr in ("reward_head", "score", "value_head", "classifier"):
            head = getattr(_backbone, attr, None)
            if head is not None and hasattr(head, "weight"):
                self._score.weight.data.copy_(head.weight.data)
                break
        rm_device = next(_backbone.parameters()).device
        self._score = self._score.to(rm_device)
        self._score.eval()
        self._score.requires_grad_(False)

        reward_tokenizer = AutoTokenizer.from_pretrained(
            reward_cfg["path"], trust_remote_code=True, local_files_only=True
        )
        self.model = _RetokenizingBackbone(_backbone, policy_tokenizer, reward_tokenizer)
        self.config = config

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states arrive on the caller's device (cuda:0); move to score head device for compute
        rm_device = next(self._score.parameters()).device
        src_device = hidden_states.device
        return self._score(hidden_states.to(rm_device)).to(src_device)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class TRLPPOTrainer:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.output_dir = Path(self.cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self, tokenizer) -> Dataset:
        cfg = self.cfg
        prompt_template = cfg["dataset"]["prompt_template"]
        max_len = cfg.get("max_prompt_length", 512)

        raw_prompts = []
        with open(cfg["dataset"]["train_path"], "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                raw_prompts.append(prompt_template.format(prompt=item["prompt"]))

        def tokenize(batch):
            return tokenizer(batch["text"], padding=False, truncation=True, max_length=max_len)

        ds = Dataset.from_dict({"text": raw_prompts})
        ds = ds.map(tokenize, batched=True, remove_columns=["text"])
        return ds

    def train(self):
        cfg = self.cfg
        ppo_cfg_dict = cfg["ppo"]
        gen_cfg = cfg["generation"]
        device_maps = cfg.get("device_maps", {})
        dtype = getattr(torch, cfg["dtype"])

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["tokenizer_path"],
            padding_side="left",
            truncation_side="right",
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA config
        lora_cfg = None
        if cfg["lora"]["enable"]:
            lora_cfg = LoraConfig(
                r=cfg["lora"]["r"],
                lora_alpha=cfg["lora"]["lora_alpha"],
                target_modules=cfg["lora"]["target_modules"],
                lora_dropout=cfg["lora"]["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
            )

        # Policy (actor)
        print("Loading policy model...")
        policy = AutoModelForCausalLM.from_pretrained(
            cfg["model_path"],
            torch_dtype=dtype,
            device_map=device_maps.get("actor", "auto"),
            local_files_only=True,
        )

        # Reference model — None when using LoRA (TRL disables adapter for ref)
        ref_policy = None
        if lora_cfg is None:
            print("Loading reference model...")
            ref_policy = AutoModelForCausalLM.from_pretrained(
                cfg["model_path"],
                torch_dtype=dtype,
                device_map=device_maps.get("ref_model", "auto"),
                local_files_only=True,
            )
            ref_policy.eval()
            ref_policy.requires_grad_(False)

        # Value model (separate sequence classifier head, num_labels=1)
        print("Loading value model...")
        value_model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_path"],
            torch_dtype=dtype,
            device_map=device_maps.get("value_model", device_maps.get("actor", "auto")),
            num_labels=1,
            local_files_only=True,
        )

        # Reward model
        print("Loading reward model...")
        reward_model = InternLM2RewardWrapper(cfg["reward_model"], policy_tokenizer=tokenizer)

        # Dataset
        print("Preparing dataset...")
        train_dataset = self._load_dataset(tokenizer)

        # PPOConfig
        ppo_config = PPOConfig(
            output_dir=str(self.output_dir),
            learning_rate=float(ppo_cfg_dict["learning_rate"]),
            per_device_train_batch_size=ppo_cfg_dict["batch_size"],
            num_mini_batches=ppo_cfg_dict.get("num_mini_batches", 1),
            local_rollout_forward_batch_size=ppo_cfg_dict.get("local_rollout_forward_batch_size", 2),
            num_ppo_epochs=ppo_cfg_dict["ppo_epochs"],
            gamma=ppo_cfg_dict["gamma"],
            lam=ppo_cfg_dict["lam"],
            cliprange=ppo_cfg_dict["clip_range"],
            cliprange_value=ppo_cfg_dict["clip_range"],
            vf_coef=ppo_cfg_dict["vf_coef"],
            kl_coef=ppo_cfg_dict.get("kl_coef", 0.05),
            max_grad_norm=ppo_cfg_dict["max_grad_norm"],
            response_length=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            stop_token="eos",
            num_train_epochs=cfg.get("num_train_epochs", 1),
            save_steps=cfg.get("save_freq", 10),
            logging_steps=cfg.get("log_freq", 1),
            bf16=(cfg["dtype"] == "bfloat16"),
            gradient_checkpointing=cfg.get("use_gradient_checkpointing", True),
            sft_model_path=cfg["model_path"],
            reward_model_path=cfg["reward_model"]["path"],
        )

        # Trainer
        print("Initializing PPOTrainer...")
        trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
            peft_config=lora_cfg,
        )

        print("Starting TRL PPO training...")
        trainer.train()

        print(f"Saving model to {self.output_dir}...")
        trainer.save_model(str(self.output_dir))
        print("Done.")
