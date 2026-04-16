import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from huggingface_hub import hf_hub_download
import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "ppo_config.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


REQUIRED_FILES = [
    "config.json",
    "configuration_internlm2.py",
    "model.safetensors.index.json",
    "model-00001-of-00008.safetensors",
    "model-00002-of-00008.safetensors",
    "model-00003-of-00008.safetensors",
    "model-00004-of-00008.safetensors",
    "model-00005-of-00008.safetensors",
    "model-00006-of-00008.safetensors",
    "model-00007-of-00008.safetensors",
    "model-00008-of-00008.safetensors",
    "modeling_internlm2.py",
    "special_tokens_map.json",
    "tokenization_internlm2.py",
    "tokenization_internlm2_fast.py",
    "tokenizer.model",
    "tokenizer_config.json",
]


def main():
    config = load_config()
    reward_model_config = config["reward_model"]
    model_name = reward_model_config["name"]
    model_path = Path(reward_model_config["path"])
    model_path.mkdir(parents=True, exist_ok=True)

    print(f"正在下载奖励模型: {model_name}")
    print(f"保存到: {model_path.resolve()}")

    for filename in REQUIRED_FILES:
        target = model_path / filename
        if target.exists():
            print(f"已存在，跳过: {filename}")
            continue

        print(f"下载中: {filename}")
        hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )

    print(f"\n完成，奖励模型保存在: {model_path.resolve()}")


if __name__ == "__main__":
    main()
