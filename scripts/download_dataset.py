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


def process_coig_cqia_dataset(dataset, output_path):
    processed_data = []
    
    for item in dataset:
        instruction = item.get("instruction", "").strip()
        input_context = item.get("input", "").strip()
        
        if input_context:
            prompt = f"{instruction}\n{input_context}".strip()
        else:
            prompt = instruction
        
        processed_data.append({
            "prompt": prompt
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return len(processed_data)


def extract_last_turn(dialogue):
    lines = dialogue.strip().split("\n\n")
    
    last_human = ""
    last_assistant = ""
    
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Assistant:") and not last_assistant:
            last_assistant = line[len("Assistant:"):].strip()
        elif line.startswith("Human:") and last_assistant:
            last_human = line[len("Human:"):].strip()
            break
    
    return last_human, last_assistant


def process_and_save_dataset(dataset, output_path, dataset_name):
    if dataset_name == "m-a-p/COIG-CQIA":
        return process_coig_cqia_dataset(dataset, output_path)
    
    processed_data = []
    
    for item in dataset:
        chosen_dialogue = item["chosen"]
        rejected_dialogue = item["rejected"]
        
        prompt, chosen_response = extract_last_turn(chosen_dialogue)
        _, rejected_response = extract_last_turn(rejected_dialogue)
        
        processed_data.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response
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
    
    train_count = process_and_save_dataset(dataset["train"], str(train_path), dataset_config["name"])
    if "test" in dataset:
        val_count = process_and_save_dataset(dataset["test"], str(val_path), dataset_config["name"])
    elif "validation" in dataset:
        val_count = process_and_save_dataset(dataset["validation"], str(val_path), dataset_config["name"])
    else:
        val_count = 0
    
    print(f"数据集处理完成!")
    print(f"训练集: {train_count} 条")
    print(f"验证集: {val_count} 条")
    print(f"原始数据保存到: {raw_dir.resolve()}")
    print(f"处理后数据保存到: {processed_dir.resolve()}")


if __name__ == "__main__":
    main()
