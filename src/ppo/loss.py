import torch
import torch.nn.functional as F


def compute_log_probs_from_logits(logits, input_ids):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_input_ids = input_ids[:, 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    action_log_probs = log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
    
    batch_size, seq_len = input_ids.shape
    padded_action_log_probs = torch.zeros((batch_size, seq_len), dtype=action_log_probs.dtype, device=action_log_probs.device)
    padded_action_log_probs[:, 1:] = action_log_probs
    
    return padded_action_log_probs


def compute_entropy_from_logits(logits):
    shift_logits = logits[:, :-1, :].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    batch_size, seq_len = logits.shape[0], logits.shape[1]
    padded_entropy = torch.zeros((batch_size, seq_len), dtype=entropy.dtype, device=entropy.device)
    padded_entropy[:, 1:] = entropy
    
    return padded_entropy


def compute_ppo_loss(
    actor_critic,
    input_ids,
    attention_mask,
    old_action_log_probs,
    old_values,
    advantages,
    returns,
    loss_mask,
    clip_range,
    value_clip_range,
    value_loss_coef,
    entropy_coef
):
    outputs = actor_critic(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    logits = outputs["logits"]
    values = outputs["values"]
    
    current_action_log_probs = compute_log_probs_from_logits(logits, input_ids)
    
    entropy = compute_entropy_from_logits(logits)
    
    log_ratio = current_action_log_probs - old_action_log_probs
    ratio = torch.exp(log_ratio)
    
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    policy_loss_1 = ratio * advantages
    policy_loss_2 = clipped_ratio * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2)
    
    values_pred = values
    values_pred_clipped = old_values + torch.clamp(values_pred - old_values, -value_clip_range, value_clip_range)
    
    value_loss_1 = (values_pred - returns) ** 2
    value_loss_2 = (values_pred_clipped - returns) ** 2
    value_loss = torch.max(value_loss_1, value_loss_2) * 0.5
    
    entropy_bonus = entropy
    
    policy_loss = (policy_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    value_loss = (value_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    entropy_bonus = (entropy_bonus * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    
    total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus
    
    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_bonus": entropy_bonus,
        "approx_kl": ((ratio - 1) - log_ratio).mean(),
        "clip_fraction": ((torch.abs(ratio - 1) > clip_range).float() * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    }
