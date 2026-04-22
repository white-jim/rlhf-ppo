"""
已弃用：奖励计算现在通过 verl 官方模型型路径实现。
verl 会自动启动 internlm2-7b-reward 的 vLLM server 并调用 /classify 接口。
配置见 run_ppo.sh 中的 reward.reward_model.* 参数。
"""
