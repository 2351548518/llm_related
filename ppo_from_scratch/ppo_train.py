"""用最少组件演示：如何用 PPO 对因果语言模型做 RLHF 微调。

这是一份教学实现，主流程与 ``ppo.png`` 对应：

1. Actor 根据 prompt 生成 response（采样轨迹）；
2. Actor、Reference、Critic、Reward Model 分别给轨迹打分；
3. 用 ``任务奖励 - KL 惩罚`` 构造逐 token 奖励；
4. 用 GAE 计算 advantage/return；
5. 用 PPO clipped objective 更新 Actor，用回归损失更新 Critic。

最重要的张量对齐（假设 prompt 有 3 个 token，response 有 2 个 token）：

    seqs                 = [p0, p1, p2, a0, a1]       # 长度 5
    logits[:, :-1] 预测  = [p1, p2, a0, a1]           # 长度 4
    取最后 num_actions=2 项后，对应 [a0, a1] 的 log_prob

说明：代码刻意保持简洁，适合理解算法，不是可直接用于大规模训练的生产实现。
例如它没有分布式训练、梯度累积、advantage 白化、保存 checkpoint 等机制。
"""

from dataclasses import dataclass
import random
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


class PromptDataset(Dataset):
    """把原始问题转换成 Actor 可以直接接收的 prompt 字符串。

    Args:
        prompts: 原始问题列表，例如 ``["1+1 等于多少？"]``。
        tokenizer: Actor 对应的 tokenizer。
        apply_chat_template: 是否包装成模型原生的聊天模板。

    例子（具体格式随模型而异）：
        输入 ``"1+1 等于多少？"``，Qwen 聊天模板可能生成类似
        ``<|im_start|>user\n1+1 等于多少？<|im_end|>...assistant\n``。
        若不使用聊天模板，则只在文本前添加 BOS token。
    """

    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.final_prompts = []

        for prompt in prompts:
            if apply_chat_template: # SFT 模型
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    content,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else: # 预训练模型
                prompt = self.tokenizer.bos_token + prompt
            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.final_prompts[index]

# 价值（评论家）模型，用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
# - 分类头：[batch, seq_len, vocab_size] → 每个位置一个 vocab 维向量
# - 价值头：[batch, seq_len, 1] → 每个位置一个 1 维向量（其实就是标量，但还包着一层）
class Critic(nn.Module):
    """价值模型：为 response 中的每一个动作 token 估计状态价值 V(s_t)。

    ``base_model`` 输出每个位置的 hidden state，再通过线性 ``value_head``
    将 ``hidden_size`` 映射成一个标量。

    形状示例：
        ``input_ids.shape == [2, 8]``，``num_actions == 3``；
        骨干输出 ``[2, 8, hidden_size]``，最终返回 ``values.shape == [2, 3]``。

    ``[:, :-1]`` 的原因与语言模型标签错位一致：位置 t 的状态用来评价下一步
    token；随后 ``[:, -num_actions:]`` 只保留 response 对应的位置。

    注意：本脚本传入 ``actor_model.base_model``，所以 Actor 与 Critic 共享骨干参数。
    这便于教学和节省显存，但两个 optimizer 都会更新共享参数，不是最常见的工程做法。
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model # 策略模型 初始化的 得到的
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1) # 只优化模型头

    def forward(self, input_ids, attention_mask, num_actions):
        # 以 B=2, L=8(=prompt 5 + response 3), hidden=H 为例贯穿本函数。
        # input_ids:      [B=2, L=8]
        # attention_mask: [B=2, L=8]
        hidden_state = self.base_model(
            input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        # hidden_state: [B=2, L=8, H]            # 每个位置一个 H 维向量
        value_model_output = self.value_head(hidden_state)
        # value_model_output: [B=2, L=8, 1]      # Linear(hidden->1) 给每位置一个标量，第3维=1 是残留
        values = value_model_output.squeeze(-1)[:, :-1][:, -num_actions:]
        # 1) squeeze(-1):  [B=2, L=8, 1] -> [B=2, L=8]   去掉大小为1的尾巴
        # 2) [:, :-1]:     [B=2, L=8] -> [B=2, 7]        丢最后一位，对齐"位置t的状态评价t+1 token"
        # 3) [:, -num_actions=3]: [B=2, 7] -> [B=2, 3]   只保留 response 的 3 个价值
        return values  # [B=2, 3]


def compute_policy_loss(
    log_probs,
    old_log_probs,
    advantages,
    action_mask=None,
    clip_eps=0.2,
):
    """计算 PPO 的 clipped policy loss。

    ``ratio = pi_new(a|s) / pi_old(a|s) = exp(new_logp - old_logp)``。
    clip 会限制一次更新相对旧策略偏移过远。

    数值例子：old_logp=-1.0，new_logp=-0.7，则 ratio≈1.35。
    当 ``clip_eps=0.2`` 且 advantage=2 时：
    ``surr1≈2.70``，``surr2=1.2*2=2.40``，取较小者，阻止过度增大概率。

    ``action_mask`` 为 0 的位置是补齐 token，不应参与 loss。
    """
    # 全程以 B=2, num_actions=3 为例。下列张量第1维都是 response 长度 3。
    # log_probs/old_log_probs/advantages/action_mask 形状均为 [B=2, 3]。
    ratio = (log_probs - old_log_probs).exp()
    # ratio: [B=2, 3]   exp(logp_new - logp_old) = π_new/π_old，逐 token 的概率比
    surr1 = ratio * advantages
    # surr1: [B=2, 3]   未裁剪的代理目标
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    # surr2: [B=2, 3]   把 ratio 限制在 [0.8, 1.2] 后再乘 advantage
    loss = -torch.min(surr1, surr2)
    # loss: [B=2, 3]   取较小者并取负（要最小化 loss = 最大化目标）
    if action_mask is None:
        return loss.mean(-1).mean()
    # 有 mask 时：先按 token 维加权求和再除以有效 token 数，得到每条样本的平均 loss，
    # 再对 batch 取平均，最终 loss 是一个标量。
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(
    values,
    old_values,
    returns,
    action_mask=None,
    clip_eps: float = None,
):
    """让 Critic 的 V(s_t) 拟合 GAE 得到的 return。

    若传入 ``clip_eps``，会像 PPO policy 一样限制新 value 相对 old value 的变化。
    本脚本调用时没有传 ``clip_eps``，因此实际使用普通均方误差。

    例：``values=0.4``、``returns=1.0``，该位置损失为 ``(0.4-1)^2=0.36``。
    """
    # 以 B=2, num_actions=3 为例：values/old_values/returns/action_mask 均为 [B=2, 3]。
    if clip_eps is not None:
        # 把 (values - old_values) 限制在 [-clip_eps, clip_eps]，防止 value 一步跳太远
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)  # [B=2,3]
        surr1 = (values_clipped - returns) ** 2   # [B=2,3]
        surr2 = (values - returns) ** 2           # [B=2,3]
        loss = torch.max(surr1, surr2)             # [B=2,3] 取较大者 = 悲观估计
    else:
        loss = (values - returns) ** 2            # [B=2,3] 普通 MSE，本脚本走这条分支

    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()  # 标量


class ExperienceBuffer:
    """存放 rollout 经验的简单内存缓冲区。

    ``append`` 会把一个 micro-batch 的 ``Experience`` 拆成 buffer item；
    DataLoader 随后将这些 item 打乱、重新组装成训练 batch。
    """

    def __init__(self, limit):
        self.limit = limit
        self.buffer = []

    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
            "seqs",
            "action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
            "num_actions",
        )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value

        self.buffer.extend(batch)
        # 只保留最近的 limit 个 micro-batch，避免 buffer 无限增长。
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer) - self.limit :]

    def get_batches(self, batch_size):
        """随机抽样接口；当前训练主流程使用 DataLoader，未调用此函数。"""
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]


@dataclass
class Samples:
    """
    策略模型的输出
    Actor rollout 的结果；张量第一维都是 micro rollout batch。
    以 B=2, prompt=5, response=3, L=8, num_actions=3 为例标注各字段 shape。
    """

    seqs: torch.Tensor
    """[B, L] prompt+response 拼接后的完整 token id 序列，左填充并补齐到 max_length+max_new_tokens=306"""

    attention_mask: Optional[torch.LongTensor]
    """[B, L] 整条序列的非 pad 位置为 1（prompt+response 都算 1）；用于告诉模型哪些位置是真实 token"""

    action_mask: Optional[torch.BoolTensor]
    """[B, num_actions] 只覆盖 response 段：真正生成的 token 为 1，生成结束后的 pad 为 0；用来只对 response 算 loss"""

    num_actions: Union[int, torch.Tensor]
    """response 的长度 = max_new_tokens=50，也就是"动作数"（每个生成的 token 是一个动作）"""

    packed_seq_lens: Optional[torch.Tensor]
    """预留字段，本脚本未使用（pack 模式下记录每条序列打包前的真实长度）；恒为 None"""

    response_length: torch.Tensor
    """[B] 每条样本 response 中真正生成（非 pad）的 token 数；= action_mask 沿 dim=-1 求和"""

    total_length: torch.Tensor
    """[B] 每条样本整条序列的有效 token 数（prompt+response）；= attention_mask 沿 dim=-1 求和"""


@dataclass
class Experience:
    """
    一次 rollout 计算完旧 log_prob、奖励、价值和 GAE 后的完整经验。
    以 B=2, prompt=5, response=3, L=8, num_actions=3 为例标注各字段 shape。
    所有训练目标量都已 detach，作为"旧策略"快照。
    """

    seqs: torch.Tensor
    """[B, L] prompt+response 完整序列，与 Samples.seqs 相同"""

    action_log_probs: torch.Tensor
    """[B, num_actions] Actor 在采样到的 response token 上的 OLD log π_θ(a|s)，detach；做 ratio 的分母"""

    values: torch.Tensor
    """[B, num_actions] Critic 在采样时刻预测的 V_old(s_t)，detach；value_loss 里做 clip 的基准"""

    returns: Optional[torch.Tensor]
    """[B, num_actions] GAE 目标回报 return_t = A_t + V(s_t)，detach；value_loss 的拟合目标"""

    advantages: Optional[torch.Tensor]
    """[B, num_actions] GAE 优势 A_t，detach；>0 鼓励该动作，<0 抑制；policy_loss 里乘进 ratio"""

    attention_mask: Optional[torch.LongTensor]
    """[B, L] 整条序列非 pad 位置为 1，与 Samples.attention_mask 相同"""

    action_mask: Optional[torch.BoolTensor]
    """[B, num_actions] response 段有效 token 为 1，与 Samples.action_mask 相同；loss 只算这些位置"""

    reward: torch.Tensor
    """[B, 1] Reward Model 给整条回复打的原始标量分数 r（未经 KL 罚、未经 clip 的原始值）"""

    response_length: torch.Tensor
    """[B] 每条 response 真实生成 token 数（沿用自 Samples）"""

    total_length: torch.Tensor
    """[B] 每条整条序列有效 token 数（沿用自 Samples）"""

    num_actions: Union[int, torch.Tensor]
    """response 长度 = max_new_tokens=50；切片 [:, -num_actions] 时用它定位 response 段"""

    kl: Optional[torch.Tensor] = None
    """[B, num_actions] 逐 token 的近似 KL = log π_θ(a|s) − log π_ref(a|s)；compute_rewards 据此算 KL 罚"""


def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):
    """返回采样 token 上的 ``log pi_actor - log pi_ref``。

    这不是对词表求和得到的精确 KL，而是只在实际采样动作上的 Monte Carlo 估计。
    例如 actor_logp=-1.0、ref_logp=-1.3，则结果为 0.3，说明 Actor 对该 token
    比 Reference 更自信；``compute_rewards`` 会给它 ``-kl_ctl * 0.3`` 的惩罚。
    """
    # log_probs/ref_log_probs 均为 [B=2, 3]（response 段的逐 token logp）。
    log_ratio = log_probs.float() - ref_log_probs.float()
    # log_ratio: [B=2, 3]   = log π_θ(a|s) - log π_ref(a|s)，逐 token 的对数概率比
    if action_mask is not None:
        # 把 padding 位置的 log_ratio 清零，避免无效 token 干扰
        log_ratio = log_ratio * action_mask  # [B=2, 3]
    return log_ratio

# δ(t) = R(t) + gam*V(t+1) - V(t)                          # TD 误差
# gae: A(t) = δ(t) + gam*lam*A(t+1)                          # = R(t)+gam*V(t+1)-V(t) + gam*lam*A(t+1)
# 最后一个时刻的未来优势和未来收益为0：A(T+1)=0, V(T+1)=0, 则 A(T)=R(T)-V(T)
# A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T)        # 知道 A(T) 可算 A(T-1)，依次类推
# returns(t) = A(t) + V(t) = R(t) + gam*(V(t+1) + lam*A(t+1))
def get_advantages_and_returns(
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    lambd: float,
):
    """从序列末端向前递推 GAE advantage，并得到 return。

    公式：
        ``delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)``
        ``A_t = delta_t + gamma * lambda * A_{t+1}``
        ``return_t = A_t + V(s_t)``

    两步数值例子（gamma=1，lambda=1）：
        rewards=[0, 1]，values=[0.2, 0.4]；
        A_1 = 1 - 0.4 = 0.6；
        A_0 = (0 + 0.4 - 0.2) + 0.6 = 0.8；
        returns=[1.0, 1.0]。

    ``detach`` advantage，确保更新 Actor 时梯度不会反向流入 Critic。
    """
    # 以 B=2, num_actions=3 为例：values/rewards/action_mask 均为 [B=2, 3]。
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)   # = 3，response 长度（= num_actions）

    if action_mask is not None:
        values = action_mask * values     # [B=2, 3] padding 位置清零
        rewards = action_mask * rewards   # [B=2, 3]

    for t in reversed(range(response_length)):  # 倒序 t=2,1,0
        # 终点之后没有未来状态，因此最后一步 nextvalues=0。
        # t<2 时取 values[:, t+1]: [B=2]；最后一步 nextvalues=0.0（标量，广播）
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]   # [B=2] TD 误差
        lastgaelam = delta + gamma * lambd * lastgaelam              # [B=2] GAE 递推
        advantages_reversed.append(lastgaelam)  # 列表里 3 个 [B=2] 张量，顺序是 t=2,1,0

    # advantages_reversed[::-1] 把顺序翻回 t=0,1,2；stack 到 dim=1 -> [B=2, 3]
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values   # [B=2, 3]   return_t = A_t + V(s_t)
    return advantages.detach(), returns


def generate_samples(
    prompts,
    model,
    max_length,
    max_new_tokens,
    n_samples_per_prompt,
    micro_rollout_batch_size,
):
    """用当前 Actor 为每个 prompt 生成多个 response。

    例：2 个 prompt、``n_samples_per_prompt=3`` 会展开成 6 条 rollout；若
    ``micro_rollout_batch_size=2``，则分 3 次生成，以降低峰值显存。

    为便于后续 ``torch.cat``，所有序列都补齐到
    ``max_length + max_new_tokens``。``action_mask`` 只覆盖 response 区域：
    真正生成的 token 为 1，生成结束后的 padding 为 0。
    """
    samples_list = []
    model.eval()
    all_prompts = sum(
        [[prompt] * n_samples_per_prompt for prompt in prompts],
        [],
    )

    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i : i + micro_rollout_batch_size]
        inputs = actor_tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        # inputs['input_ids']: [B=micro_rollout, max_length=256]，左填充到等长
        # inputs['attention_mask']: [B, 256]
        # 策略模型
        input_ids = inputs["input_ids"]
        seqs = model.generate(
            **inputs.to(device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        # generate 后 seqs: [B, 256 + max_new_tokens]（prompt 256 + 生成 ≤50）
        # 若提前遇到 eos，后半段用 pad_token_id 补齐。

        target_length = max_length + max_new_tokens   # 256+50 = 306
        if seqs.size(1) >= target_length:
            seqs = seqs[:, :target_length]            # [B, 306] 截断
        else:
            # 不足 306 时右侧补 pad 到 306（生成不足 max_new_tokens 的情况）
            padding = torch.full(
                (seqs.size(0), target_length - seqs.size(1)),
                fill_value=pad_token_id,
                device=seqs.device,
            )
            seqs = torch.cat([seqs, padding], dim=1)  # [B, 306]

        # attention_mask: 整条序列（prompt+response）非 pad 处为 1 -> [B, 306]
        attention_mask = seqs.ne(pad_token_id).to(dtype=torch.long)


        # 模型的输出部分
        # 输入固定补到 max_length，故从 input_ids.size(1) 开始都是 response。
        ans = seqs[:, input_ids.size(1) :]   # [B, 50] 只取 response 段
        # action_mask: response 中真正生成的 token 为 1，生成结束后的 pad 为 0 -> [B, 50]
        action_mask = ans.ne(pad_token_id).to(dtype=torch.long)

        samples = Samples(
            seqs=seqs,                                   # [B, 306]
            attention_mask=attention_mask,               # [B, 306]
            action_mask=action_mask,                     # [B, 50]
            num_actions=action_mask.size(1),             # 50 (= max_new_tokens)
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),  # [B] 每条实际生成的 token 数
            total_length=attention_mask.float().sum(dim=-1),   # [B] 每条总有效 token 数
        )
        samples_list.append(samples)

    return samples_list


def compute_rewards(
    kl,                # [B, num_actions] 逐 token 的近似 KL = log π_θ(a|s) - log π_ref(a|s)，来自 compute_approx_kl，pad 位已被 action_mask 清零
    r,                 # [B, 1] Reward Model 对整条回复打的标量分数（结果奖励，一个标量/条）
    action_mask,       # [B, num_actions] response 段有效 token 为 1、pad 为 0；用于定位"最后一个真实生成 token"把 RM 分加过去
    kl_ctl,            # float，KL 惩罚系数 β（本代码 0.1）：每个 token 背的罚 = -kl_ctl * kl，控制偏离参考模型的强度
    clip_reward_value,  # float，RM 分数的 clip 范围 c（本代码 0.2）：r 被 clip 到 [-c, c]，防止极端 reward 主导训练
):
    """
    把序列级 Reward Model 分数分配成逐 token 奖励。

    每个有效 response token 都收到 KL 惩罚 ``-kl_ctl * log_ratio``；
    只有最后一个有效 token 额外收到裁剪后的任务奖励 ``r``。

    例：KL=[0.1, 0.2, 0.0]，r=0.8，kl_ctl=0.1，reward clip=0.2，
    action_mask=[1,1,0]，则奖励约为 [-0.01, -0.02+0.2, 0]。
    """
    # 以 B=2, num_actions=3 为例。kl/action_mask: [B=2,3]；r: [B=2,1]（RM 输出标量）。
    kl_divergence_estimate = -kl_ctl * kl
    # [B=2, 3]  每个 response token 先背一个 KL 罚：-kl_ctl * log_ratio
    rewards = kl_divergence_estimate
    ends = action_mask.sum(1)   # [B=2] 每条样本最后一个有效 token 的位置（1-based 计数）

    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)

    reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)
    # reward_clip: [B=2, 1]  把 RM 标量分数 clip 到 [-0.2, 0.2]
    batch_size = r.size(0)
    for j in range(batch_size):
        # [:ends[j]][-1] 定位到该样本最后一个非 padding 的动作 token。
        # rewards[j]: [3]；[:ends[j]] 截到有效段；[-1] 取末位 -> 标量
        rewards[j, : ends[j]][-1] += reward_clip[j, 0]

    return rewards   # [B=2, 3]  逐 token 奖励


def generate_experiences(samples_list):
    """为 rollout 补齐 PPO 更新所需的所有统计量。

    Actor log_prob 会作为 ``old_log_probs`` 固定下来；同一批经验训练多个 epoch 时，
    train_step 重新计算的 log_prob 才是 ``new_log_probs``。这正是 PPO ratio 的来源。
    """
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []

    for samples in samples_list:
        # 从 Samples 取出本 micro-batch 的四个核心量（沿用采样阶段已构造好的结果）。
        # 以 B=2, prompt=5, response=3, L=8, num_actions=3 为例标注 shape。
        seqs = samples.seqs # seqs 是 tokenizer 之后的 token id 序列（一个整数张量 torch.LongTensor），不是字符串
        # [B, L] prompt+response 完整序列（左填充 prompt 在前、response 在后，补齐到 max_length+max_new_tokens=306）。Actor/Ref/Critic 都吃它做前向。
        attention_mask = samples.attention_mask
        # [B, L] 整条序列非 pad 位置为 1（prompt 真实 token + response 真实 token 都为 1，两处 pad 都为 0）。喂给模型前向，告诉模型哪些位置是真实 token。
        action_mask = samples.action_mask
        # [B, num_actions] 只覆盖 response 段：真实生成的 token（含 EOS）为 1，生成结束后的 pad 为 0。喂给 loss，只在 response 真实 token 上算损失。
        num_actions = samples.num_actions
        # int，response 区域宽度 = max_new_tokens=50（不是真实生成数！）。用于 [:, -num_actions:] 切片，把 logits/value 的最后 50 列切出来对齐到 response 段。


        # rollout 统计量都作为固定训练目标，不需要保留计算图。
        # 以 B=2, prompt=5, response=3, L=8, num_actions=3, hidden=H, vocab=V 为例。
        with torch.no_grad():

            """
            计算 策略模型 输出 token 的概率
            """
            # Actor 对实际出现的“下一个 token”取 log probability。
            output = actor_model(seqs, attention_mask=attention_mask)
            # seqs: [B=2, L=8]；output.logits: [B=2, L=8, V]  V=词表大小
            logits = output.logits
            # logits[:, :-1]: [B=2, 7, V]  丢最后一位，位置 t 预测 t+1 token
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            # log_probs: [B=2, 7, V]  在词表维做 log_softmax
            log_probs_labels = log_probs.gather(
                dim=-1,
                index=seqs[:, 1:].unsqueeze(-1),
            )
            # seqs[:, 1:]: [B=2, 7]  丢第一位，作为"被预测的下一个 token"标签
            #   .unsqueeze(-1): [B=2, 7, 1]  作为 gather 的 index（在 V 维上取）
            # gather 后: [B=2, 7, 1]  取出每个位置实际下一个 token 的 logp
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
            # squeeze(-1): [B=2, 7, 1] -> [B=2, 7]
            # [:, -num_actions=3]: [B=2, 7] -> [B=2, 3]  只留 response 的 3 个 logp（OLD）

            """
            计算 参考模型 输出 token 的概率
            """
            # Reference 始终不更新，用它约束 Actor 不要偏离初始语言模型太远。
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits   # [B=2, 8, V]
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)   # [B=2, 7, V]
            ref_log_probs_labels = ref_log_probs.gather(
                dim=-1,
                index=seqs[:, 1:].unsqueeze(-1),   # [B=2, 7, 1]
            )   # [B=2, 7, 1]
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[
                :, -num_actions:
            ]   # squeeze -> [B=2,7] -> [:, -3] -> [B=2, 3]
            

            """
            计算价值,只计算生成token的价值
            """
            # Critic 估计 response 每一步的 V(s_t)。
            value = critic_model(seqs, attention_mask, num_actions).to(device)
            # value: [B=2, 3]  response 段每个 token 一个标量价值


            """
            转换成 文本
            """
            # Reward Model 是序列分类器：整段 prompt+response 只输出一个标量分数。
            seq_texts = actor_tokenizer.batch_decode(
                seqs,
                skip_special_tokens=True,
            )
            # seq_texts: list[str]，长度 B=2；把 token id 解码回文本

            """
            计算 奖励模型的 奖励值,每条文本一个标量奖励分数 奖励模型的输出，相当于生成 最后一个 token的奖励（结果奖励模型）
            """
            reward_model_inputs = reward_tokenizer(
                seq_texts,
                return_tensors="pt",
                padding=True,
            )
            # reward_model_inputs['input_ids']: [B=2, T]  T=该 batch 最长文本长度
            r = reward_model(**reward_model_inputs.to(device)).logits
            # r: [B=2, 1]  


            """
            计算KL散度(近似KL散度)
            """
            kl = compute_approx_kl(
                action_log_probs,
                ref_action_log_probs,
                action_mask=action_mask,
            ).to(device)
            # kl: [B=2, 3]  逐 token 的 log(π_θ/π_ref)

            
            """
            计算实际奖励
            """
            rewards = compute_rewards(
                kl,
                r,
                action_mask,
                kl_ctl=0.1,
                clip_reward_value=0.2,
            )
            # rewards: [B=2, 3]  逐 token 奖励（KL 罚 + 末 token 的 RM 分数）

            """
            计算 优势 和 回报
            """
            advantages, returns = get_advantages_and_returns(
                value,
                rewards,
                action_mask,
                gamma=0.1,
                lambd=0.2,
            )
            # advantages: [B=2, 3]，returns: [B=2, 3]  GAE 递推结果（已 detach）

        experiences.append(
            Experience(
                seqs,
                action_log_probs.detach(),
                value.detach(),
                returns.detach(),
                advantages.detach(),
                attention_mask,
                action_mask,
                r.detach(),
                samples.response_length,
                samples.total_length,
                num_actions,
                kl.detach(),
            )
        )

    return experiences


@dataclass
class BufferItem:
    """DataLoader 合并后，直接传给 ``train_step`` 的训练 batch。"""

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]


def collate_fn(batch):
    """把多个 buffer item 在 batch 维（dim=0）拼接。

    例：每个 item 来自 2 条 rollout，DataLoader batch_size=2，拼接后实际训练
    batch 含 4 条序列。这里的 ``micro_train_batch_size`` 数的是 item，不一定等于
    最终的序列条数。
    """
    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []

    for x in batch:
        seqs.append(x["seqs"])
        action_log_probs.append(x["action_log_probs"])
        values.append(x["values"])
        returns.append(x["returns"])
        advantages.append(x["advantages"])
        attention_mask.append(x["attention_mask"])
        action_mask.append(x["action_mask"])
    # 每个 append 进来的张量第0维是 1（一条样本）。
    # 例：batch 有 N 个 item，每个 seqs 形状 [1, 306] -> 列表里 N 个 [1,306]。

    seqs = torch.cat(seqs, dim=0)
    # cat(dim=0): N×[1,306] -> [N, 306]；其余同理。
    action_log_probs = torch.cat(action_log_probs, dim=0)   # [N, num_actions]
    values = torch.cat(values, dim=0)                       # [N, num_actions]
    returns = torch.cat(returns, dim=0)                     # [N, num_actions]
    advantages = torch.cat(advantages, dim=0)              # [N, num_actions]
    attention_mask = torch.cat(attention_mask, dim=0)      # [N, L]
    action_mask = torch.cat(action_mask, dim=0)            # [N, num_actions]

    return BufferItem(
        seqs,                       # [N, L]
        action_log_probs,           # [N, num_actions]  = OLD log_probs
        values,                     # [N, num_actions]  = OLD values
        returns,                    # [N, num_actions]
        advantages,                 # [N, num_actions]
        attention_mask,             # [N, L]
        action_mask,                # [N, num_actions]
        action_mask.size(1),        # num_actions（int）
    )


def train_step(experience, steps):
    """在同一批旧经验上，分别更新一次 Actor 和 Critic。

    核心是 OLD（采样时刻快照，已 detach）vs NEW（当前参数重新前向，带梯度）的对比：
      - OLD 量来自 generate_experiences、在采样时刻固定下来：old_action_log_probs、
        old_values、advantages、returns。
      - NEW 量是本函数里重新前向算出来的：action_log_probs、values，会随每个 epoch
        的参数更新而变化。
      - PPO 的 ratio = exp(NEW_logp - OLD_logp) 就来自这一对；advantages/returns 是固定靶。

    两阶段更新（Actor 先、Critic 后，分开 backward+step，不合成 total_loss）：
      ① Actor: NEW log_probs 与 OLD 算 ratio、乘 advantages 走 PPO clip 损失。
      ② Critic: NEW values 与 returns 算 MSE，让 V(s_t) 逼近 G_t。

    例子（N=1, num_actions=2 便于手算）：
        采样快照: old_logp=[-1.0,-0.5], advantages=[0.8,0.6],
                  old_values=[0.2,0.4], returns=[1.0,1.0]
        重新前向: NEW logp=[-0.7,-0.4]  (adv 为正→训练在提高概率)
        ratio = exp(NEW-OLD) ≈ [1.35, 1.105]
        token0: ratio=1.35 超过 1.2 → clip 砍到 1.2，限制概率一次涨幅 ≤20%
        policy_loss = -min(ratio*adv, clip(ratio)*adv) 的均值（负数，最小化它=最大化目标）
        Critic: NEW values=[0.3,0.45], value_loss=((0.3-1)^2+(0.45-1)^2)/2≈0.396
    """
    # ===== ① Actor 更新 =====
    actor_model.train()
    optimizer_actor.zero_grad()

    # 从经验里取 OLD 快照与固定目标（都已 detach，不回传梯度）。
    # 以 N=4, L=306, num_actions=50 为例（N = micro_train_batch_size × micro_rollout_batch_size）。
    sequences = experience.seqs                       # [N, L] prompt+response 完整序列
    old_action_log_probs = experience.action_log_probs  # [N, num_actions] OLD log π_old(a|s)，ratio 分母
    advantages = experience.advantages                # [N, num_actions] GAE 优势，>0 鼓励、<0 抑制
    num_actions = experience.num_actions              # int，response 区域宽度，用于切片
    attention_mask = experience.attention_mask        # [N, L] 整条非 pad，喂模型前向
    action_mask = experience.action_mask              # [N, num_actions] response 段有效 token，喂 loss
    old_values = experience.values                    # [N, num_actions] OLD V_old(s)，value clip 基准
    returns = experience.returns                      # [N, num_actions] G_t = A_t + V_t，Critic 目标

    # 重新前向得到当前策略的 NEW log_prob（带梯度，会随参数更新变化）。
    logits = actor_model(sequences, attention_mask=attention_mask).logits   # [N, L, V]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)                     # [N, L-1, V] 丢最后一位对齐
    log_probs_labels = log_probs.gather(
        dim=-1,
        index=sequences[:, 1:].unsqueeze(-1),   # [N, L-1, 1]  用真实下一个 token 当 index
    )   # [N, L-1, 1]  取出每个位置实际下一个 token 的 logp
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]       # [N, num_actions] = NEW log_probs

    # PPO clip 损失：ratio=exp(NEW-OLD)，乘 advantages，clip 到 [0.8,1.2] 取 min，再取负最小化。
    policy_loss = compute_policy_loss(
        action_log_probs,        # [N, num_actions]  NEW（分子）
        old_action_log_probs,    # [N, num_actions]  OLD（分母，快照）
        advantages,              # [N, num_actions]  固定信号
        action_mask=action_mask, # [N, num_actions]  只在 response 真实 token 上算
    )   # 标量（通常为负，最小化它 = 最大化 PPO 目标）
    policy_loss.backward()       # 只回传到 Actor（OLD/advantages 已 detach，不流入 Critic）
    optimizer_actor.step()
    writer.add_scalar("policy_loss", policy_loss.item(), steps)   # 记曲线（对应 ppo.png 的 policy_loss）

    # ===== ② Critic 更新 =====
    critic_model.train()
    optimizer_critic.zero_grad()
    # 重新前向得到当前 Critic 的 NEW values（带梯度）。
    values = critic_model(sequences, attention_mask, num_actions)   # [N, num_actions] = NEW V(s)
    # MSE 回归损失：让 NEW values 逼近固定的 returns。本代码未传 clip_eps，走普通 MSE 分支。
    value_loss = compute_value_loss(
        values,        # [N, num_actions]  NEW（带梯度）
        old_values,    # [N, num_actions]  OLD（快照，本分支未用到，传了是为 clip 分支预留）
        returns,       # [N, num_actions]  固定目标 G_t
        action_mask,   # [N, num_actions]  只在 response 真实 token 上算
    )   # 标量（正数）
    value_loss.backward()        # 回传到 Critic（含共享的 base_model backbone）
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)   # 记曲线（对应 ppo.png 的 value_loss）

    print(
        f"step: {steps}  policy_loss: {policy_loss.item():.4f}  "
        f"value_loss: {value_loss.item():.4f}"
    )


def train():
    """外层采样、内层优化的 PPO 训练循环。"""
    buffer = ExperienceBuffer(limit=100)
    steps = 0

    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            # 先用当前 Actor 采样，得到固定的 on-policy 训练数据。
            """
            生成样本(获取模型推理结果)
            """
            samples = generate_samples(
                rand_prompts,
                actor_model,
                max_length,
                max_new_tokens,
                n_samples_per_prompt,
                micro_rollout_batch_size,
            )
            """
            生成经验(获取优势、奖励、回报等)
            """
            # 计算 old log_prob、reward、value、advantage 和 return。
            experiences = generate_experiences(samples)
            buffer.append(experiences)

            dataloader = DataLoader(
                buffer,
                batch_size=micro_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            torch.cuda.empty_cache()

            # PPO 允许一批 rollout 重复训练若干轮；轮数太多会使策略离旧策略过远。
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1

            # 清空后重新采样，保证下一批经验来自更新后的 Actor。
            buffer.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 完整遍历 prompt 数据集的轮数。
    episodes = 3
    # 每批 rollout 被 PPO 重复优化的轮数。
    max_epochs = 5
    # 每次从 prompt 数据集取多少个问题。
    rollout_batch_size = 8
    # 每次送进多个推理模型的 rollout 数；调小可降低峰值显存。
    micro_rollout_batch_size = 2
    # 每个 prompt 生成几个不同回答，用于增加探索。
    n_samples_per_prompt = 2
    # response 最多生成的 token 数，也就是最多动作数。
    max_new_tokens = 50
    # prompt 分词后的固定长度；超长截断，不足则左侧 padding。
    max_length = 256
    # DataLoader 一次读取多少个 buffer item，详见 collate_fn 的批大小说明。
    micro_train_batch_size = 2

    writer = SummaryWriter("./runs")

    # Actor：需要被 PPO 更新的策略模型。
    actor_model = AutoModelForCausalLM.from_pretrained(
        "/home/user/Downloads/Qwen2.5-0.5B-Instruct"
    ).to(device)
    # Reference：初始策略的冻结参照；它只在 torch.no_grad() 中前向。
    ref_model = AutoModelForCausalLM.from_pretrained(
        "/home/user/Downloads/Qwen2.5-0.5B-Instruct"
    ).to(device)
    # Reward Model：对完整回答输出偏好分数。
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "/home/user/Downloads/reward-model-deberta-v3-large-v2"
    ).to(device)

    actor_tokenizer = AutoTokenizer.from_pretrained(
        "/home/user/Downloads/Qwen2.5-0.5B-Instruct"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "/home/user/Downloads/reward-model-deberta-v3-large-v2"
    )

    # 教学简化：Critic 复用 Actor 的 base_model，只新增 value head。
    critic_model = Critic(actor_model.base_model).to(device)

    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)

    # decoder-only 模型批量生成通常使用左 padding，确保真实 prompt 末尾对齐。
    actor_tokenizer.padding_side = "left"
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id

    # 小型演示数据。正式 RLHF 应使用规模更大、分布更合理的 prompt 集合。
    prompt_list = [
        "请问 1+1 等于多少？",
        "在 PowerShell 中，如何知道 BIOS 中的虚拟化是否已禁用？",
        "为什么人们喜欢在水族馆里游泳，而不是在游泳池里？",
        "你是一位营销专家。为 Instagram Reels 写 10 个带有营销技巧的脚本。",
        "你是一位营销专家。为 Instagram Reels 写 10 个带有营销技巧的脚本。",
        "你是一位营销专家。为 Instagram Reels 写 10 个带有营销技巧的脚本。",
        "为什么所有的镜子都是矩形的？",
        "在受感染的植物根部可以找到哪一种：臭氧还是金子？",
    ]

    prompts_dataset = PromptDataset(
        prompt_list,
        actor_tokenizer,
        apply_chat_template=True,
    )
    prompts_dataloader = DataLoader(
        prompts_dataset,
        batch_size=rollout_batch_size,
        shuffle=True,
    )

    train()
