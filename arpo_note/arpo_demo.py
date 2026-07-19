"""ARPO 的最小可运行数值演示（仅依赖 Python 标准库）。

这个文件不负责真实地调用 LLM 或工具，而是把 ARPO 最容易混淆的三件事拆开：

1. 根据 logits 计算归一化 token entropy；
2. 用工具调用前后的 entropy delta 调整分支概率，并受总 rollout 预算约束；
3. 对带共享前缀的轨迹计算 hard advantage attribution。

运行：
    python arpo_note/arpo_demo.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """数值稳定的 softmax。"""
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    scaled = [x / temperature for x in logits]
    max_logit = max(scaled)
    exps = [math.exp(x - max_logit) for x in scaled]
    normalizer = sum(exps)
    return [x / normalizer for x in exps]


def token_entropy(
    logits: Sequence[float],
    temperature: float = 1.0,
    normalize: bool = True,
) -> float:
    """计算一个生成位置的 Shannon entropy。

    使用自然对数时，完整词表分布的最大熵是 log(V)，因此除以 log(V) 后，
    归一化熵位于 [0, 1]。
    """
    probabilities = softmax(logits, temperature)
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    if not normalize:
        return entropy
    if len(probabilities) <= 1:
        return 0.0
    return entropy / math.log(len(probabilities))


def mean_entropy(
    logits_by_position: Sequence[Sequence[float]],
    first_k: int = 20,
) -> float:
    """取前 first_k 个生成位置的归一化 entropy 均值。"""
    selected = logits_by_position[:first_k]
    if not selected:
        return 0.0
    return sum(token_entropy(logits) for logits in selected) / len(selected)


def branch_probability(
    entropy_initial: float,
    entropy_now: float,
    base_probability: float = 0.5,
    entropy_weight: float = 0.2,
) -> float:
    """论文 P_t = alpha + beta * Delta H_t 的概率形式。"""
    entropy_delta = entropy_now - entropy_initial
    probability = base_probability + entropy_weight * entropy_delta
    return max(0.0, min(1.0, probability))


@dataclass(frozen=True)
class BranchCandidate:
    path: str
    entropy_initial: float
    entropy_now: float

    @property
    def entropy_delta(self) -> float:
        return self.entropy_now - self.entropy_initial


def choose_branches(
    candidates: Iterable[BranchCandidate],
    *,
    group_size: int,
    init_sample_size: int,
    beam_size: int = 2,
    base_probability: float = 0.5,
    entropy_weight: float = 0.2,
    seed: int = 3,
) -> list[tuple[str, float, float, bool]]:
    """按 ARPO 的预算和熵概率，从当前路径创建额外分支。

    返回元素为：
        (路径名, 有效分支概率, 随机阈值 u, 是否成功创建分支)

    beam_size=2 表示每条源路径最多新建 beam_size - 1 = 1 条分支。
    """
    if group_size < init_sample_size:
        raise ValueError("group_size 不能小于 init_sample_size")
    if beam_size < 1:
        raise ValueError("beam_size 必须至少为 1")

    rng = random.Random(seed)
    remaining_slots = group_size - init_sample_size
    created = 0
    decisions: list[tuple[str, float, float, bool]] = []

    for candidate in candidates:
        branches_for_this_path = min(
            beam_size - 1,
            remaining_slots - created,
        )
        if branches_for_this_path <= 0:
            break

        for _ in range(branches_for_this_path):
            probability = branch_probability(
                candidate.entropy_initial,
                candidate.entropy_now,
                base_probability,
                entropy_weight,
            )
            threshold = rng.random()
            branched = threshold <= probability
            decisions.append(
                (candidate.path, probability, threshold, branched)
            )
            if branched:
                created += 1

    return decisions


def group_relative_advantages(
    rewards: Sequence[float],
    eps: float = 1e-8,
) -> list[float]:
    """用总体标准差计算 GRPO/ARPO 的组内相对优势。"""
    if not rewards:
        raise ValueError("rewards 不能为空")
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
    std_reward = math.sqrt(variance)
    if std_reward < eps:
        return [0.0 for _ in rewards]
    return [(r - mean_reward) / (std_reward + eps) for r in rewards]


def hard_shared_advantage(
    descendant_indices: Sequence[int],
    trajectory_advantages: Sequence[float],
) -> float:
    """共享前缀的 hard advantage：取所有后代轨迹优势的平均值。"""
    if not descendant_indices:
        raise ValueError("共享前缀至少要有一条后代轨迹")
    selected = [trajectory_advantages[i] for i in descendant_indices]
    return sum(selected) / len(selected)


def clipped_grpo_objective(
    ratios: Sequence[float],
    advantages: Sequence[float],
    clip_epsilon: float = 0.2,
) -> float:
    """教学用的 token 级 GRPO surrogate objective 均值。"""
    if len(ratios) != len(advantages):
        raise ValueError("ratios 和 advantages 长度必须相同")
    if not ratios:
        raise ValueError("ratios 不能为空")

    lower, upper = 1.0 - clip_epsilon, 1.0 + clip_epsilon
    terms = []
    for ratio, advantage in zip(ratios, advantages):
        clipped_ratio = max(lower, min(upper, ratio))
        terms.append(min(ratio * advantage, clipped_ratio * advantage))
    return sum(terms) / len(terms)


def demo_entropy() -> None:
    print("=== 1. token entropy ===")
    confident_logits = [5.0, 0.0, -1.0, -2.0]
    uncertain_logits = [1.0, 1.0, 1.0, 1.0]
    print(f"高置信分布的归一化熵: {token_entropy(confident_logits):.4f}")
    print(f"均匀分布的归一化熵:   {token_entropy(uncertain_logits):.4f}")


def demo_rollout() -> None:
    print("\n=== 2. entropy-adaptive rollout ===")
    candidates = [
        BranchCandidate("A", 0.30, 0.55),
        BranchCandidate("B", 0.30, 0.25),
        BranchCandidate("C", 0.30, 0.70),
        BranchCandidate("D", 0.30, 0.32),
        BranchCandidate("E", 0.30, 0.60),
        BranchCandidate("F", 0.30, 0.20),
    ]
    decisions = choose_branches(
        candidates,
        group_size=8,
        init_sample_size=6,
        beam_size=2,
        base_probability=0.5,
        entropy_weight=0.2,
        seed=3,
    )

    print("group_size=8, init_sample_size=6，所以最多补 2 条分支")
    for path, probability, threshold, branched in decisions:
        result = "创建分支" if branched else "继续原路径"
        print(
            f"路径 {path}: p_branch={probability:.3f}, "
            f"u={threshold:.3f} -> {result}"
        )
    print(f"实际创建分支数: {sum(x[3] for x in decisions)}")


def demo_advantage() -> None:
    print("\n=== 3. hard advantage attribution ===")
    names = ["A1", "A2", "B", "C1", "C2", "D", "E", "F"]
    rewards = [1.0, 0.0, 0.2, 0.9, 0.1, 0.4, 0.8, 0.2]
    advantages = group_relative_advantages(rewards)

    for name, reward, advantage in zip(names, rewards, advantages):
        print(f"{name:>2}: reward={reward:.1f}, advantage={advantage:+.3f}")

    shared_a = hard_shared_advantage([0, 1], advantages)
    shared_c = hard_shared_advantage([3, 4], advantages)
    print(f"A1/A2 的共享前缀 advantage: {shared_a:+.3f}")
    print(f"C1/C2 的共享前缀 advantage: {shared_c:+.3f}")
    print("分叉后的 token 仍分别使用各自完整轨迹的 advantage。")


def demo_soft_objective() -> None:
    print("\n=== 4. soft setting 的直觉 ===")
    # 两条后代轨迹共享某个前缀 token，因此该 token 的 ratio 相同。
    shared_ratio = 1.05
    child_advantages = [1.497, -1.225]
    objective = clipped_grpo_objective(
        [shared_ratio, shared_ratio],
        child_advantages,
    )
    print(
        "共享 token 的两个样本项使用同一 ratio，求平均后等价于 "
        f"ratio × 平均 advantage，目标值={objective:+.4f}"
    )


if __name__ == "__main__":
    demo_entropy()
    demo_rollout()
    demo_advantage()
    demo_soft_objective()
