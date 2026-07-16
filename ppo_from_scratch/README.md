# PPO from scratch：代码与训练曲线导读

这份代码用四个模型展示语言模型 RLHF 中的 PPO：

- **Actor**：生成回答，也是最终要优化的模型。
- **Reference**：Actor 的初始副本，不训练，用于计算 KL 惩罚。
- **Reward Model**：给完整回答一个偏好分数。
- **Critic**：给回答中的每个 token 估计价值 `V(s_t)`。

## 一条样本如何流过代码

假设 prompt 是“`1+1=`”，Actor 生成“`2。`”，把 token 简写成：

```text
完整序列 seqs       [请, 计算, 1+1, 2, 。]
response/action                    [2, 。]
action_mask                        [1,  1]
Actor log_prob                     [-0.3, -0.1]
Reference log_prob                 [-0.4, -0.2]
log_ratio/KL estimate              [ 0.1,  0.1]
```

若 `kl_ctl=0.1`，两个 token 先分别得到 `-0.01` 的 KL 惩罚。假设 Reward
Model 的分数裁剪后是 `0.2`，它只加到最后一个有效 token，最终逐 token 奖励为
`[-0.01, 0.18]`。随后 `get_advantages_and_returns` 从右向左计算 GAE，得到 Actor
使用的 advantage 和 Critic 拟合的 return。

## PPO clip 为什么有效

`train_step` 在 rollout 后重新计算当前 Actor 的 log probability，并与采样时保存的
`old_action_log_probs` 比较：

```text
ratio = exp(new_log_prob - old_log_prob)
```

若某个正 advantage 动作的概率从旧策略到新策略增加了 35%，`ratio=1.35`；默认
`clip_eps=0.2` 只按 `1.2` 计算收益，避免单次更新走得过远。负 advantage 的动作同理，
clip 会限制概率下降幅度。

## 如何阅读 ppo.png

![TensorBoard 中的 policy loss 与 value loss](./ppo.png)

- `value_loss` 开始接近 100，前十几个 step 快速下降到约 10，随后逐渐接近 0：说明
  随机初始化的 value head 很快开始拟合当前批次的 return。
- `policy_loss` 前半段整体下降，但约在 step 40 从接近 0 跳到约 0.25；同一位置
  `value_loss` 也出现小峰值。这与代码在新 episode/新 rollout 上切换数据的时点相符：
  新回答、奖励和 return 的分布变化后，两个损失都可能突然变大。
- 后半段 policy loss 保持在约 0.25～0.35，**不能单凭这张图判断策略变好或变坏**。
  PPO loss 依赖 advantage 的尺度和符号，不像监督学习交叉熵那样要求持续降到 0。

更可靠的诊断还应记录：平均 Reward Model 分数、平均 KL、clip fraction、response
长度、advantage 均值/标准差以及验证集人工偏好胜率。

## 这份教学实现的关键限制

1. Actor 与 Critic 共享 `actor_model.base_model`，且两个 optimizer 都包含共享参数。
   Actor 更新一次后，Critic 更新还会再次改变同一骨干；生产实现通常会明确控制共享、
   冻结或使用独立 Critic。
2. Reference 虽然只在 `torch.no_grad()` 中使用，但没有显式将参数
   `requires_grad_(False)`；结果正确，但会多占一些优化相关的心智与维护成本。
3. 没有 advantage 白化、动态 KL controller、梯度裁剪、混合精度或分布式训练。
4. `gamma=0.1`、`lambda=0.2`、reward clip `0.2` 都是演示参数，不应直接当作通用配置。
5. 图中只有 loss，无法证明奖励真的提升；训练时应同时增加上述 PPO 诊断指标。

运行前需将脚本底部的本地模型路径替换为实际路径，然后执行：

```powershell
python .\ppo_from_scratch\ppo_train.py
tensorboard --logdir .\runs
```
