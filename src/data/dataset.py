import os
import json
from typing import List, Dict, Any
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset


class RLHFDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split="train"):
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        
        if split == "train":
            data_path = dataset_config["train_path"]
        else:
            data_path = dataset_config["val_path"]
        
        if os.path.exists(data_path):
            self.data = self._load_processed_data(data_path)
        else:
            raw_dir = dataset_config["raw_dir"]
            self.data = self._load_and_process_raw_data(raw_dir, split)
        
        self.prompt_template = dataset_config["prompt_template"]
    
    def _load_processed_data(self, data_path):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def _load_and_process_raw_data(self, raw_dir, split):
        dataset = load_from_disk(raw_dir)
        split_data = dataset[split]
        
        processed_data = []
        for item in split_data:
            prompt = item["prompt"]
            processed_item = {"prompt": prompt}
            if "chosen" in item:
                processed_item["chosen"] = item["chosen"]
            if "rejected" in item:
                processed_item["rejected"] = item["rejected"]
            processed_data.append(processed_item)
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        result = {"prompt": item["prompt"]}
        if "chosen" in item:
            result["chosen"] = item["chosen"]
        if "rejected" in item:
            result["rejected"] = item["rejected"]
        return result
