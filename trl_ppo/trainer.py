import os
import sys
import json
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(train_path: str) -> List[str]:
    prompts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item["prompt"])
    return prompts


# ---------------------------------------------------------------------------
# Reward model wrapper (reuses the same InternLM2 reward model logic)
# ---------------------------------------------------------------------------

class RewardModel:
    def __init__(self, reward_cfg: dict):
        self.prompt_template = reward_cfg["prompt_template"]
        device = reward_cfg.get("device", "cuda")
        dtype = getattr(torch, reward_cfg.get("dtype", "bfloat16"))

        config = AutoConfig.from_pretrained(
            reward_cfg["path"], trust_remote_code=True, local_files_only=True
        )
        # Fix rope_scaling for InternLM2
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

        self.model = AutoModel.from_pretrained(
            reward_cfg["path"],
            config=config,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        tok_kwargs = dict(
            padding_side="right", truncation_side="right",
            trust_remote_code=True, local_files_only=True,
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                reward_cfg["path"], use_fast=True, **tok_kwargs
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                reward_cfg["path"], use_fast=False, **tok_kwargs
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._output_attr = self._detect_output_attr()

    def _detect_output_attr(self):
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt")
            dummy = {k: v.to(self._device()) for k, v in dummy.items()}
            try:
                out = self.model(**dummy)
                for attr in ("logits", "scores", "end_scores"):
                    if hasattr(out, attr):
                        return attr
                return "direct" if isinstance(out, torch.Tensor) else "logits"
            except Exception:
                return "logits"

    def _device(self):
        try:
            if hasattr(self.model, "hf_device_map"):
                d = list(self.model.hf_device_map.values())[0]
                return torch.device("cuda", d) if isinstance(d, int) else torch.device(d)
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        texts = [
            self.prompt_template.format(prompt=p, response=r)
            for p, r in zip(prompts, responses)
        ]
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        device = self._device()
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=self.model.dtype):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        if self._output_attr == "direct":
            rewards = out.squeeze(-1)
        elif self._output_attr == "end_scores":
            rewards = out.end_scores.squeeze(-1)
        else:
            rewards = getattr(out, self._output_attr).squeeze(-1)

        # Return per-sample scalar tensors (TRL expects a list of 0-d or 1-d tensors)
        return [rewards[i].float().cpu() for i in range(len(prompts))]


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class TRLPPOTrainer:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.output_dir = Path(self.cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_actor(self):
        cfg = self.cfg
        dtype = getattr(torch, cfg["dtype"])
        device_map = cfg.get("device_maps", {}).get("actor", "auto")

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

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg["model_path"],
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            peft_config=lora_cfg,
        )
        return model

    def _build_ref_model(self):
        cfg = self.cfg
        dtype = getattr(torch, cfg["dtype"])
        device_map = cfg.get("device_maps", {}).get("ref_model", "auto")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg["model_path"],
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        )
        ref_model.eval()
        ref_model.requires_grad_(False)
        return ref_model

    def train(self):
        cfg = self.cfg
        ppo_cfg_dict = cfg["ppo"]
        gen_cfg = cfg["generation"]

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["tokenizer_path"],
            padding_side="left",
            truncation_side="right",
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Models
        print("Loading actor model...")
        model = self._build_actor()
        print("Loading reference model...")
        ref_model = self._build_ref_model()
        print("Loading reward model...")
        reward_model = RewardModel(cfg["reward_model"])

        # PPOConfig
        ppo_config = PPOConfig(
            model_name=cfg["model_name"],
            learning_rate=float(ppo_cfg_dict["learning_rate"]),
            batch_size=ppo_cfg_dict["batch_size"],
            mini_batch_size=ppo_cfg_dict["mini_batch_size"],
            ppo_epochs=ppo_cfg_dict["ppo_epochs"],
            gamma=ppo_cfg_dict["gamma"],
            lam=ppo_cfg_dict["lam"],
            cliprange=ppo_cfg_dict["clip_range"],
            vf_coef=ppo_cfg_dict["vf_coef"],
            target_kl=ppo_cfg_dict["target_kl"],
            kl_penalty=ppo_cfg_dict["kl_penalty"],
            max_grad_norm=ppo_cfg_dict["max_grad_norm"],
            log_with=None,
            project_kwargs={"logging_dir": str(self.output_dir / "logs")},
        )

        trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
        )

        # Dataset
        prompts = load_prompts(cfg["dataset"]["train_path"])
        prompt_template = cfg["dataset"]["prompt_template"]
        batch_size = ppo_cfg_dict["batch_size"]
        num_epochs = cfg.get("num_train_epochs", 1)

        generation_kwargs = {
            "max_new_tokens": gen_cfg["max_new_tokens"],
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
            "top_k": gen_cfg["top_k"],
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        global_step = 0
        save_freq = cfg.get("save_freq", 10)
        log_freq = cfg.get("log_freq", 1)

        print(f"Starting TRL PPO training — {num_epochs} epoch(s), {len(prompts)} samples")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            indices = list(range(0, len(prompts), batch_size))

            for start in tqdm(indices, desc=f"Epoch {epoch + 1}"):
                batch_prompts = prompts[start: start + batch_size]
                formatted = [prompt_template.format(prompt=p) for p in batch_prompts]

                # Tokenize prompts → list of tensors
                query_tensors = [
                    tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
                    for text in formatted
                ]

                # Generate responses
                response_tensors = trainer.generate(
                    query_tensors, return_prompt=False, **generation_kwargs
                )
                responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

                # Score with reward model
                rewards = reward_model.score(batch_prompts, responses)

                # PPO step
                stats = trainer.step(query_tensors, response_tensors, rewards)

                global_step += 1

                if global_step % log_freq == 0:
                    mean_reward = sum(r.item() if hasattr(r, "item") else r for r in rewards) / len(rewards)
                    print(
                        f"  Step {global_step} | "
                        f"reward: {mean_reward:.4f} | "
                        f"policy_loss: {stats.get('ppo/loss/policy', 0):.4f} | "
                        f"value_loss: {stats.get('ppo/loss/value', 0):.4f} | "
                        f"mean_kl: {stats.get('ppo/mean_non_score_reward', 0):.4f}"
                    )

                if global_step % save_freq == 0:
                    save_path = self.output_dir / f"checkpoint-step-{global_step}"
                    trainer.save_pretrained(str(save_path))
                    print(f"  Saved checkpoint to {save_path}")

        # Final save
        final_path = self.output_dir / "final"
        trainer.save_pretrained(str(final_path))
        print(f"\nTraining complete. Model saved to {final_path}")
