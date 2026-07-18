"""可嵌入训练循环的 GSPO PyTorch 核心实现。

约定：
    new_log_probs: [B, T]，当前策略对已采样 completion token 的 log-prob
    old_log_probs: [B, T]，rollout/旧策略保存的 log-prob
    completion_mask: [B, T]，有效 completion token 为 1，padding/prompt 为 0
    advantages: [B]，每条响应的组内相对优势

此文件只实现损失，不包含 rollout、reward、分布式训练和优化器。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def group_relative_advantages(
    rewards: Tensor,
    numerical_eps: float = 1e-8,
) -> Tensor:
    """按 prompt 分组标准化 reward。

    Args:
        rewards: [num_prompts, group_size]。
        numerical_eps: 防止组内 reward 完全相同时除零。

    Returns:
        与 rewards 同形状的 advantage。
    """
    if rewards.ndim != 2:
        raise ValueError("rewards 必须是 [num_prompts, group_size]")

    means = rewards.mean(dim=1, keepdim=True)
    # unbiased=False 对应分母为 G 的总体标准差。
    stds = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - means) / (stds + numerical_eps)


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    """只对 mask=1 的位置求平均。"""
    mask = mask.to(dtype=values.dtype)
    counts = mask.sum(dim=dim)
    if torch.any(counts == 0):
        raise ValueError("每条响应至少需要一个有效 completion token")
    return (values * mask).sum(dim=dim) / counts


def gspo_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    completion_mask: Tensor,
    advantages: Tensor,
    eps_low: float = 3e-4,
    eps_high: float = 4e-4,
) -> tuple[Tensor, dict[str, Any]]:
    """计算最小化形式的 GSPO loss。

    论文写的是最大化目标 J；训练代码通常返回 loss = -J。
    eps_low/eps_high 默认采用 GSPO 论文实验中的非对称范围，仅作为起点，
    不代表对所有模型和训练配置都最优。
    """
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError("new_log_probs 与 old_log_probs 形状必须相同")
    if new_log_probs.shape != completion_mask.shape:
        raise ValueError("completion_mask 必须与 log-probs 同形状")
    if new_log_probs.ndim != 2:
        raise ValueError("log-probs 必须是 [batch, completion_length]")
    if advantages.shape != (new_log_probs.shape[0],):
        raise ValueError("advantages 必须是 [batch]")
    if eps_low < 0 or eps_high < 0:
        raise ValueError("裁剪阈值不能为负数")

    # old_log_probs 是 rollout 时旧策略的常量，不能让梯度流入它。
    token_log_ratios = new_log_probs - old_log_probs.detach()

    # log s_i = (1 / |y_i|) * sum_t(log pi_theta - log pi_old)
    sequence_log_ratios = masked_mean(
        token_log_ratios,
        completion_mask,
        dim=1,
    )
    sequence_ratios = torch.exp(sequence_log_ratios)

    advantages = advantages.detach()
    unclipped = sequence_ratios * advantages
    clipped_ratios = torch.clamp(
        sequence_ratios,
        min=1.0 - eps_low,
        max=1.0 + eps_high,
    )
    clipped = clipped_ratios * advantages

    # 论文最大化 min(unclipped, clipped)，代码最小化它的相反数。
    per_sequence_objective = torch.minimum(unclipped, clipped)
    loss = -per_sequence_objective.mean()

    lower = 1.0 - eps_low
    upper = 1.0 + eps_high
    clipped_mask = (sequence_ratios < lower) | (sequence_ratios > upper)
    metrics: dict[str, Any] = {
        "objective": per_sequence_objective.detach().mean(),
        "sequence_ratio_mean": sequence_ratios.detach().mean(),
        "sequence_ratio_min": sequence_ratios.detach().min(),
        "sequence_ratio_max": sequence_ratios.detach().max(),
        "sequence_clip_fraction": clipped_mask.float().detach().mean(),
    }
    return loss, metrics
