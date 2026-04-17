import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager


class ReferenceModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["model_path"],
            torch_dtype=getattr(torch, model_config["dtype"]),
            device_map="auto",
            local_files_only=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["tokenizer_path"],
            padding_side="left",
            truncation_side="right",
            local_files_only=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self.model.requires_grad_(False)
    
    @contextmanager
    def inference_mode(self):
        with torch.inference_mode():
            try:
                with torch.amp.autocast("cuda", dtype=self.model.dtype):
                    yield
            except TypeError:
                with torch.amp.autocast("cuda"):
                    yield

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        with self.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                use_cache=False,
            )
        return outputs
