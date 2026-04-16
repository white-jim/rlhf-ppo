import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
from datasets import load_dataset
import yaml
import json


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_config.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_and_save_dataset(dataset, output_path):
    processed_data = []
    
    for item in dataset:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        processed_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return len(processed_data)


def main():
    config = load_config()
    current_dataset = config["datasets"]["current_dataset"]
    dataset_config = config["datasets"][current_dataset]
    
    print(f"正在下载数据集: {dataset_config['name']}")
    
    raw_dir = Path(dataset_config["raw_dir"])
    processed_dir = Path(dataset_config["processed_dir"])
    train_path = Path(dataset_config["train_path"])
    val_path = Path(dataset_config["val_path"])
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(dataset_config["name"])
    
    print(f"原始数据集保存到: {raw_dir.resolve()}")
    dataset.save_to_disk(raw_dir)
    
    print("\n正在处理数据集...")
    
    train_count = process_and_save_dataset(dataset["train"], str(train_path))
    val_count = process_and_save_dataset(dataset["test"], str(val_path))
    
    print(f"数据集处理完成!")
    print(f"训练集: {train_count} 条")
    print(f"验证集: {val_count} 条")
    print(f"原始数据保存到: {raw_dir.resolve()}")
    print(f"处理后数据保存到: {processed_dir.resolve()}")


if __name__ == "__main__":
    main()
