import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


class ActorCritic(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["model_path"],
            torch_dtype=getattr(torch, model_config["dtype"]),
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["tokenizer_path"],
            padding_side="right",
            truncation_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(self.hidden_size, 1, dtype=self.model.dtype)
        
        if model_config["lora"]["enable"]:
            lora_config = LoraConfig(
                r=model_config["lora"]["r"],
                lora_alpha=model_config["lora"]["lora_alpha"],
                target_modules=model_config["lora"]["target_modules"],
                lora_dropout=model_config["lora"]["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
        
        if model_config.get("use_gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
    
    def forward(self, input_ids, attention_mask=None, return_dict=True):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict
        )
        
        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]
        
        values = self.value_head(last_hidden_state).squeeze(-1)
        
        return {
            "logits": logits,
            "values": values,
            "hidden_states": outputs.hidden_states
        }
    
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs
        )
    
    def get_trainable_params(self, lr_config):
        backbone_params = []
        value_head_params = []
        
        for name, param in self.named_parameters():
            if "value_head" in name:
                value_head_params.append(param)
            else:
                backbone_params.append(param)
        
        return [
            {"params": backbone_params, "lr": lr_config["backbone"]},
            {"params": value_head_params, "lr": lr_config["value_head"]}
        ]
