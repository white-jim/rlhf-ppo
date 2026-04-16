import os
import torch
import torch.nn as nn
import yaml
import time
import random
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import RLHFDataset, PromptOnlyCollator
from ..models import ActorCritic, ReferenceModel, RewardModel
from .rollout import RolloutCollector
from .advantage import compute_gae_advantages, normalize_advantages
from .loss import compute_ppo_loss


class PPOTrainer:
    def __init__(self, model_config_path, ppo_config_path, eval_log_config_path):
        self.model_config = self._load_config(model_config_path)
        self.ppo_config = self._load_config(ppo_config_path)
        self.eval_log_config = self._load_config(eval_log_config_path)
        
        self.device = self.model_config["device"]
        self._setup_output_dirs()
        
        self.actor_critic = None
        self.reference_model = None
        self.reward_model = None
        self.rollout_collector = None
        self.optimizer = None
        
        self.kl_coef = 0.1
        self.target_kl = 0.01
        self.global_step = 0
        self.epoch = 0
    
    def _load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _setup_output_dirs(self):
        output_dir = Path(self.eval_log_config["output_dir"])
        save_dir = output_dir / self.eval_log_config["save_dir"]
        log_dir = output_dir / self.eval_log_config["log_dir"]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_dir = save_dir
        self.log_dir = log_dir
    
    def setup(self):
        print("正在初始化模型...")
        
        self.actor_critic = ActorCritic(self.model_config)
        self.reference_model = ReferenceModel(self.model_config)
        self.reward_model = RewardModel(self.ppo_config["reward_model"])
        
        self.rollout_collector = RolloutCollector(
            self.actor_critic,
            self.reference_model,
            self.reward_model,
            self.model_config,
            self.ppo_config
        )
        
        trainable_params = self.actor_critic.get_trainable_params(self.ppo_config["learning_rate"])
        self.optimizer = torch.optim.AdamW(trainable_params)
        
        print("模型初始化完成!")
    
    def prepare_dataloader(self):
        dataset_config = self.model_config["datasets"][self.model_config["datasets"]["current_dataset"]]
        
        train_dataset = RLHFDataset(dataset_config, self.actor_critic.tokenizer, split="train")
        
        collator = PromptOnlyCollator(
            self.actor_critic.tokenizer,
            self.model_config["max_seq_len"]
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.ppo_config["batch_size"],
            shuffle=True,
            collate_fn=collator
        )
        
        return train_dataloader
    
    def compute_loss_mask(self, input_ids, prompt_lengths):
        batch_size, seq_len = input_ids.shape
        loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.float, device=input_ids.device)
        
        for i, prompt_len in enumerate(prompt_lengths):
            loss_mask[i, prompt_len:] = 1.0
        
        return loss_mask
    
    def train_step(self, rollout_batch, normalized_advantages, returns, loss_mask):
        losses = compute_ppo_loss(
            self.actor_critic,
            rollout_batch.input_ids,
            rollout_batch.attention_mask,
            rollout_batch.action_log_probs,
            rollout_batch.values,
            normalized_advantages,
            returns,
            loss_mask,
            self.ppo_config["clip_range"],
            self.ppo_config["value_clip_range"],
            self.ppo_config["value_loss_coef"],
            self.ppo_config["entropy_coef"]
        )
        
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(),
            self.ppo_config["max_grad_norm"]
        )
        
        self.optimizer.step()
        
        return losses
    
    def update_kl_coef(self, mean_kl):
        if mean_kl > self.target_kl * 1.5:
            self.kl_coef = min(self.kl_coef * 1.5, 10.0)
        elif mean_kl < self.target_kl / 1.5:
            self.kl_coef = max(self.kl_coef / 1.5, 1e-4)
    
    def save_checkpoint(self, step=None):
        if step is None:
            step = self.global_step
        
        save_path = self.save_dir / f"checkpoint-step-{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.model_config["lora"]["enable"]:
            self.actor_critic.model.save_pretrained(save_path)
        else:
            self.actor_critic.model.save_pretrained(save_path)
        
        torch.save({
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "kl_coef": self.kl_coef,
        }, save_path / "training_state.pt")
        
        print(f"模型已保存到: {save_path}")
    
    def train(self, num_epochs=1):
        self.setup()
        train_dataloader = self.prepare_dataloader()
        
        print(f"开始训练，共 {num_epochs} 个 epoch...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                start_time = time.time()
                
                rollout_batch, kl_divergences = self.rollout_collector.collect_rollout(
                    [{"prompt": p} for p in batch["prompts"]],
                    self.kl_coef
                )
                
                loss_mask = self.compute_loss_mask(
                    rollout_batch.input_ids,
                    rollout_batch.prompt_lengths
                )
                
                advantages, returns = compute_gae_advantages(
                    rollout_batch.rewards,
                    rollout_batch.values,
                    self.ppo_config["gamma"],
                    self.ppo_config["lam"]
                )
                
                normalized_advantages = normalize_advantages(advantages, loss_mask)
                
                ppo_epochs = self.ppo_config["ppo_epochs"]
                batch_size = rollout_batch.input_ids.shape[0]
                mini_batch_size = self.ppo_config["mini_batch_size"]
                
                indices = list(range(batch_size))
                
                for ppo_epoch in range(ppo_epochs):
                    random.shuffle(indices)
                    
                    for start_idx in range(0, batch_size, mini_batch_size):
                        end_idx = min(start_idx + mini_batch_size, batch_size)
                        mb_indices = indices[start_idx:end_idx]
                        
                        mb_loss_mask = loss_mask[mb_indices]
                        
                        losses = self.train_step(
                            rollout_batch,
                            normalized_advantages[mb_indices],
                            returns[mb_indices],
                            mb_loss_mask
                        )
                
                mean_kl = (kl_divergences * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                self.update_kl_coef(mean_kl.item())
                
                self.global_step += 1
                
                step_time = time.time() - start_time
                
                if self.global_step % self.eval_log_config["log_freq"] == 0:
                    print(f"\nStep {self.global_step}")
                    print(f"  Policy Loss: {losses['policy_loss'].item():.4f}")
                    print(f"  Value Loss: {losses['value_loss'].item():.4f}")
                    print(f"  Entropy: {losses['entropy_bonus'].item():.4f}")
                    print(f"  KL Coef: {self.kl_coef:.4f}")
                    print(f"  Mean KL: {mean_kl.item():.4f}")
                    print(f"  Clip Fraction: {losses['clip_fraction'].item():.4f}")
                    print(f"  Step Time: {step_time:.2f}s")
                
                if self.global_step % self.eval_log_config["save_freq"] == 0:
                    self.save_checkpoint()
        
        self.save_checkpoint()
        print("\n训练完成!")
