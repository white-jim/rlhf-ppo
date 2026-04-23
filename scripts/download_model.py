# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# from pathlib import Path
# from huggingface_hub import snapshot_download
# import yaml


# def load_config():
#     config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_config.yml")
#     with open(config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def main():
#     config = load_config()
#     model_name = config["model_name"]
#     model_path = config["model_path"]
    
#     print(f"正在下载模型: {model_name}")
#     print(f"保存到: {model_path}")
    
#     local_dir = Path(model_path)
#     local_dir.mkdir(parents=True, exist_ok=True)
    
#     snapshot_download(
#         repo_id=model_name,
#         local_dir=str(local_dir),
#         local_dir_use_symlinks=False,
#         resume_download=True
#     )
    
#     print(f"模型已下载到: {local_dir.resolve()}")


# if __name__ == "__main__":
#     main()

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import list_repo_files
for f in list_repo_files("THU-KEG/WildReward-8B"):
    print(f)


# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from pathlib import Path
# from huggingface_hub import hf_hub_download
# import yaml


# def load_config():
#     config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_config.yml")
#     with open(config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# # 只下载这些文件
# REQUIRED_FILES = [
#     "config.json",
#     "tokenizer.json",
#     "tokenizer_config.json",
#     "tokenizer.json",
#     "added_tokens.json",
#     "special_tokens_map.json",
#     "vocab.json",
#     "merges.txt",
#     "chat_template.jinja",
#     "model.safetensors.index.json",
#     "model-00001-of-00002.safetensors",
#     "model-00002-of-00002.safetensors",
# ]


# def main():
#     config = load_config()
#     model_name = config["model_name"]
#     model_path = Path(config["model_path"])
#     model_path.mkdir(parents=True, exist_ok=True)

#     print(f"正在下载模型: {model_name}")
#     print(f"保存到: {model_path.resolve()}")

#     for filename in REQUIRED_FILES:
#         target = model_path / filename
#         if target.exists():
#             print(f"已存在，跳过: {filename}")
#             continue

#         print(f"下载中: {filename}")
#         hf_hub_download(
#             repo_id=model_name,
#             filename=filename,
#             local_dir=str(model_path),
#             local_dir_use_symlinks=False,
#         )

#     print(f"\n完成，模型保存在: {model_path.resolve()}")


# if __name__ == "__main__":
#     main()
