import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer


class PromptOnlyCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "prompts": prompts
        }


class FullSequenceCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def _format_chat_template(self, prompt: str, response: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    def process_pair(self, prompt: str, response: str):
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False
        )
        prompt_len = len(prompt_encodings["input_ids"])
        
        full_text = self._format_chat_template(prompt, response)
        
        full_encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False
        )
        
        input_ids = full_encodings["input_ids"]
        attention_mask = full_encodings["attention_mask"]
        
        loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
        loss_mask = loss_mask[:len(input_ids)]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "loss_mask": torch.tensor(loss_mask),
            "prompt": prompt,
            "response": response
        }
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_attention_masks = []
        all_loss_masks = []
        all_prompts = []
        all_responses = []
        
        for item in batch:
            prompt = item["prompt"]
            if "response" in item:
                response = item["response"]
            else:
                response = item["chosen"]
            
            processed = self.process_pair(prompt, response)
            
            all_input_ids.append(processed["input_ids"])
            all_attention_masks.append(processed["attention_mask"])
            all_loss_masks.append(processed["loss_mask"])
            all_prompts.append(processed["prompt"])
            all_responses.append(processed["response"])
        
        max_len = max(len(seq) for seq in all_input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_loss_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        for input_ids, attention_mask, loss_mask in zip(all_input_ids, all_attention_masks, all_loss_masks):
            padding_len = max_len - len(input_ids)
            
            padded_input_ids.append(torch.cat([input_ids, torch.full((padding_len,), pad_token_id, dtype=torch.long)]))
            padded_attention_masks.append(torch.cat([attention_mask, torch.zeros(padding_len, dtype=torch.long)]))
            padded_loss_masks.append(torch.cat([loss_mask, torch.zeros(padding_len, dtype=torch.float)]))
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "loss_mask": torch.stack(padded_loss_masks),
            "prompts": all_prompts,
            "responses": all_responses
        }
