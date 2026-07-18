"""GSPO 的纯 Python 数值演示。

这个文件不训练语言模型，只把 GSPO 最容易混淆的数学量算清楚：

1. 组内相对优势；
2. token 级重要性比率；
3. 长度归一化后的序列级重要性比率；
4. GRPO 与 GSPO 的裁剪粒度差异。

运行：
    python gspo_note/gspo_demo.py
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def mean(values: Sequence[float]) -> float:
    """返回算术平均值。"""
    if not values:
        raise ValueError("values 不能为空")
    return sum(values) / len(values)


def population_std(values: Sequence[float]) -> float:
    """返回总体标准差，即分母使用 G 而不是 G-1。"""
    center = mean(values)
    return math.sqrt(mean([(value - center) ** 2 for value in values]))


def group_relative_advantages(
    rewards: Sequence[float],
    numerical_eps: float = 1e-8,
) -> list[float]:
    """对同一 prompt 的 G 个 reward 做组内标准化。"""
    reward_mean = mean(rewards)
    reward_std = population_std(rewards)
    return [
        (reward - reward_mean) / (reward_std + numerical_eps)
        for reward in rewards
    ]


def token_ratios_from_log_probs(
    new_log_probs: Sequence[float],
    old_log_probs: Sequence[float],
) -> list[float]:
    """由新旧策略的 token log-prob 计算 token 级概率比。"""
    if len(new_log_probs) != len(old_log_probs):
        raise ValueError("new_log_probs 与 old_log_probs 长度必须相同")
    if not new_log_probs:
        raise ValueError("响应至少要包含一个有效 token")
    return [
        math.exp(new_log_prob - old_log_prob)
        for new_log_prob, old_log_prob in zip(new_log_probs, old_log_probs)
    ]


def sequence_ratio(token_ratios: Sequence[float]) -> float:
    """计算 GSPO 的长度归一化序列比率，即 token 比率的几何平均。"""
    if not token_ratios:
        raise ValueError("token_ratios 不能为空")
    if any(ratio <= 0.0 for ratio in token_ratios):
        raise ValueError("概率比必须大于 0")
    mean_log_ratio = mean([math.log(ratio) for ratio in token_ratios])
    return math.exp(mean_log_ratio)


def clip(value: float, lower: float, upper: float) -> float:
    """把 value 限制到 [lower, upper]。"""
    return min(max(value, lower), upper)


def clipped_surrogate(
    ratio: float,
    advantage: float,
    eps_low: float,
    eps_high: float,
) -> float:
    """PPO 风格的 clipped surrogate（要最大化的量）。"""
    unclipped = ratio * advantage
    clipped = clip(ratio, 1.0 - eps_low, 1.0 + eps_high) * advantage
    return min(unclipped, clipped)


def grpo_response_objective(
    token_ratios: Sequence[float],
    advantage: float,
    eps_low: float,
    eps_high: float,
) -> float:
    """GRPO：逐 token 裁剪后，再对响应中的 token 求平均。"""
    token_objectives = [
        clipped_surrogate(ratio, advantage, eps_low, eps_high)
        for ratio in token_ratios
    ]
    return mean(token_objectives)


def gspo_response_objective(
    token_ratios: Sequence[float],
    advantage: float,
    eps_low: float,
    eps_high: float,
) -> float:
    """GSPO：先聚合成一个序列比率，再对整条响应裁剪一次。"""
    ratio = sequence_ratio(token_ratios)
    return clipped_surrogate(ratio, advantage, eps_low, eps_high)


def raw_sequence_ratio(token_ratios: Iterable[float]) -> float:
    """未做长度归一化的原始联合概率比，仅用于解释长度效应。"""
    result = 1.0
    for ratio in token_ratios:
        result *= ratio
    return result


def show_group_advantage_example() -> None:
    rewards = [1.0, 0.0, 0.0, 1.0]
    advantages = group_relative_advantages(rewards)

    print("1) 组内相对优势")
    print(f"   rewards    = {rewards}")
    print(f"   mean/std   = {mean(rewards):.4f}/{population_std(rewards):.4f}")
    print(f"   advantages = {[round(value, 4) for value in advantages]}")
    print()


def show_length_normalization_example() -> None:
    short_ratios = [1.01] * 10
    long_ratios = [1.01] * 100

    print("2) 为什么要做长度归一化")
    print(
        "   每个 token 都变化 1% 时，未归一化联合比率："
        f"T=10 -> {raw_sequence_ratio(short_ratios):.4f}，"
        f"T=100 -> {raw_sequence_ratio(long_ratios):.4f}"
    )
    print(
        "   几何平均后的 GSPO 比率："
        f"T=10 -> {sequence_ratio(short_ratios):.4f}，"
        f"T=100 -> {sequence_ratio(long_ratios):.4f}"
    )
    print()


def show_clipping_example() -> None:
    # 为了单独展示“裁剪粒度”，这里让 GRPO 与 GSPO 临时使用同一阈值。
    # 真实训练时，两者的阈值通常不应相同。
    eps_low = 0.2
    eps_high = 0.2
    positive_ratios = [1.30, 1.30, 0.90, 0.90]
    negative_ratios = [0.75, 0.75, 1.05, 1.05]

    print("3) token 级裁剪与序列级裁剪")
    print(f"   教学用裁剪区间 = [{1-eps_low:.2f}, {1+eps_high:.2f}]")

    for name, ratios, advantage in [
        ("好回答", positive_ratios, 1.0),
        ("差回答", negative_ratios, -1.0),
    ]:
        seq_ratio = sequence_ratio(ratios)
        grpo_value = grpo_response_objective(
            ratios, advantage, eps_low, eps_high
        )
        gspo_value = gspo_response_objective(
            ratios, advantage, eps_low, eps_high
        )
        print(f"   {name}:")
        print(f"     token ratios = {ratios}")
        print(f"     sequence ratio = {seq_ratio:.6f}")
        print(f"     GRPO objective = {grpo_value:.6f}")
        print(f"     GSPO objective = {gspo_value:.6f}")
    print()


def show_log_prob_example() -> None:
    old_log_probs = [-1.20, -0.70, -2.00, -0.50]
    new_log_probs = [-1.15, -0.73, -1.90, -0.54]
    ratios = token_ratios_from_log_probs(new_log_probs, old_log_probs)

    print("4) 从模型输出的 log-prob 得到 s_i")
    print(f"   old log-probs = {old_log_probs}")
    print(f"   new log-probs = {new_log_probs}")
    print(f"   token ratios  = {[round(value, 6) for value in ratios]}")
    print(f"   s_i           = {sequence_ratio(ratios):.6f}")


def main() -> None:
    show_group_advantage_example()
    show_length_normalization_example()
    show_clipping_example()
    show_log_prob_example()


if __name__ == "__main__":
    main()
