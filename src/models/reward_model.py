import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from contextlib import contextmanager


class RewardModel(nn.Module):
    def __init__(self, reward_config):
        super().__init__()
        self.reward_config = reward_config
        self.prompt_template = reward_config["prompt_template"]
        
        self.model = AutoModel.from_pretrained(
            reward_config["path"],
            torch_dtype=getattr(torch, reward_config["dtype"]),
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
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
            rewards = outputs.logits.squeeze(-1)
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
        
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        rewards = self.forward(input_ids, attention_mask)
        return rewards
