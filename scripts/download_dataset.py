import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
from datasets import load_dataset
import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_config.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    current_dataset = config["datasets"]["current_dataset"]
    dataset_config = config["datasets"][current_dataset]
    
    print(f"正在下载数据集: {dataset_config['name']}")
    
    raw_dir = Path(dataset_config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(dataset_config["name"])
    
    dataset.save_to_disk(raw_dir)
    
    print(f"数据集已下载到: {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
