import torch


def compute_gae_advantages(rewards, values, gamma, lam):
    batch_size, seq_len = rewards.shape
    
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    last_advantage = 0.0
    last_return = values[:, -1]
    
    for t in range(seq_len - 1, -1, -1):
        if t == seq_len - 1:
            next_value = values[:, t]
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[:, t] = last_advantage
        
        last_return = rewards[:, t] + gamma * last_return
        returns[:, t] = last_return
    
    return advantages, returns


def normalize_advantages(advantages, loss_mask=None):
    if loss_mask is not None:
        masked_advantages = advantages * loss_mask
        mean = masked_advantages.sum() / (loss_mask.sum() + 1e-8)
        std = torch.sqrt(((masked_advantages - mean) ** 2 * loss_mask).sum() / (loss_mask.sum() + 1e-8))
    else:
        mean = advantages.mean()
        std = advantages.std()
    
    normalized_advantages = (advantages - mean) / (std + 1e-8)
    
    return normalized_advantages
