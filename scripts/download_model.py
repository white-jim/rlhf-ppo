import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_config.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    model_name = config["model_name"]
    model_path = config["model_path"]
    
    print(f"正在下载模型: {model_name}")
    print(f"保存到: {model_path}")
    
    local_dir = Path(model_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print(f"模型已下载到: {local_dir.resolve()}")


if __name__ == "__main__":
    main()
