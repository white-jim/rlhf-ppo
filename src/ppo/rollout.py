import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    prompts: List[str]
    responses: List[str]
    prompt_lengths: List[int]


class RolloutCollector:
    def __init__(self, actor_critic, reference_model, reward_model, model_config, ppo_config):
        self.actor_critic = actor_critic
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.model_config = model_config
        self.ppo_config = ppo_config
        
        self.tokenizer = actor_critic.tokenizer
        self.max_seq_len = model_config["max_seq_len"]
        self.max_new_tokens = model_config["max_new_tokens"]
    
    def generate_responses(self, prompt_batch):
        prompts = [item["prompt"] for item in prompt_batch]
        
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len - self.max_new_tokens,
            return_tensors="pt"
        )
        
        _actor_device = next(self.actor_critic.model.parameters()).device
        input_ids = prompt_encodings["input_ids"].to(_actor_device)
        attention_mask = prompt_encodings["attention_mask"].to(_actor_device)
        
        prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]
        
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.model_config["temperature"],
            "top_p": self.model_config["top_p"],
            "top_k": self.model_config["top_k"],
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }
        
        with torch.no_grad():
            full_output_ids = self.actor_critic.generate(
                input_ids=input_ids,
                **generate_kwargs
            )
        
        response_ids = full_output_ids[:, input_ids.shape[1]:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        return {
            "full_input_ids": full_output_ids,
            "prompt_lengths": prompt_lengths,
            "prompts": prompts,
            "responses": responses
        }
    
    def compute_log_probs_and_values(self, full_input_ids, prompt_lengths):
        batch_size, seq_len = full_input_ids.shape
        
        attention_mask = torch.ones_like(full_input_ids)
        
        outputs = self.actor_critic(
            input_ids=full_input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs["logits"]
        values = outputs["values"]
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_input_ids = full_input_ids[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        action_log_probs = log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
        
        padded_action_log_probs = torch.zeros((batch_size, seq_len), dtype=action_log_probs.dtype, device=action_log_probs.device)
        padded_action_log_probs[:, 1:] = action_log_probs
        
        padded_values = values[:, :-1]
        padded_values = torch.cat([padded_values, padded_values[:, -1:]], dim=-1)
        
        loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.float, device=full_input_ids.device)
        for i, prompt_len in enumerate(prompt_lengths):
            loss_mask[i, prompt_len:] = 1.0
        
        return padded_action_log_probs, padded_values, loss_mask
    
    def compute_ref_log_probs(self, full_input_ids, prompt_lengths):
        batch_size, seq_len = full_input_ids.shape
        
        attention_mask = torch.ones_like(full_input_ids)
        
        with torch.no_grad():
            outputs = self.reference_model(
                input_ids=full_input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_input_ids = full_input_ids[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        action_log_probs = log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
        
        padded_action_log_probs = torch.zeros((batch_size, seq_len), dtype=action_log_probs.dtype, device=action_log_probs.device)
        padded_action_log_probs[:, 1:] = action_log_probs
        
        return padded_action_log_probs
    
    def compute_rewards(self, prompts, responses, action_log_probs, ref_log_probs, prompt_lengths, kl_coef):
        batch_size, seq_len = action_log_probs.shape
        
        rm_rewards = self.reward_model.compute_reward(prompts, responses).view(-1)

        rewards = torch.zeros_like(action_log_probs)

        kl_divergences = action_log_probs - ref_log_probs.to(action_log_probs.device)
        
        for i, prompt_len in enumerate(prompt_lengths):
            rewards[i, prompt_len:-1] = -kl_coef * kl_divergences[i, prompt_len:-1]
            rewards[i, -1] = rm_rewards[i].to(rewards.device) - kl_coef * kl_divergences[i, -1]
        
        return rewards, kl_divergences
    
    def collect_rollout(self, prompt_batch, kl_coef):
        generate_result = self.generate_responses(prompt_batch)
        full_input_ids = generate_result["full_input_ids"]
        prompt_lengths = generate_result["prompt_lengths"]
        prompts = generate_result["prompts"]
        responses = generate_result["responses"]
        
        action_log_probs, values, loss_mask = self.compute_log_probs_and_values(
            full_input_ids, prompt_lengths
        )
        
        ref_log_probs = self.compute_ref_log_probs(full_input_ids, prompt_lengths)
        
        rewards, kl_divergences = self.compute_rewards(
            prompts, responses, action_log_probs, ref_log_probs, prompt_lengths, kl_coef
        )
        
        attention_mask = torch.ones_like(full_input_ids)
        
        return RolloutBatch(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            action_log_probs=action_log_probs,
            ref_log_probs=ref_log_probs,
            values=values,
            rewards=rewards,
            prompts=prompts,
            responses=responses,
            prompt_lengths=prompt_lengths
        ), kl_divergences
