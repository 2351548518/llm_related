# RLOO（REINFORCE Leave\-One\-Out）\& ReMax

### 策略梯度

基于策略梯度的强化学习算法优化目标可统一表示为如下形式：

$\begin{aligned}\nabla J(\pi_{\theta}) &= \mathbb{E}_{\tau \sim \pi_{\theta}} \big[ \sum_{t=0}^{T} \Psi_{t} \nabla \log \pi_{\theta}(a_{t} | s_{t}) \big] \\ \end{aligned}$

其中$\Psi_{t}$有如下不同的实现方式：

$\begin{aligned} 1.\quad & \sum_{t=0}^{\infty} r_{t} &\text{轨迹的累积奖励} \\ 2.\quad & \sum_{t'=t}^{\infty} \gamma^{t'} r_{t'} &\text{轨迹的折扣奖励} \\ 3.\quad & \sum_{t'=t}^{\infty} \gamma^{t'} r_{t'} - b(s_{t}) &\text{引入基线} \\ 4.\quad & Q^{\pi}(s_{t}, a_{t}) &\text{动作价值函数} \\ 5.\quad & A^{\pi}(s_{t}, a_{t}) &\text{优势函数} \\ 6.\quad & r_{t} + \lambda V^{\pi}(s_{t+1}) - V^{\pi}(s_{t}) &\text{时序差分残差}\end{aligned}$

### REINFORCE

使用方式1和方式2会存在高方差的问题，所以引入了方式3，通过减去一个baseline来达到降低方差的目的，方式3代表的方法即REINFORCE方法。

在REINFORCE中，使用一个batch内奖励的移动平均值作为baseline。

### RLOO

RLOO在REINFORCE的基础上进行改进，使用另一种方法作为baseline（留一法）：

$b(s, a_k) = \frac{1}{K - 1} \sum_{i=1, i \ne k}^{K} R(s, a_i)$

通俗来讲，RLOO对于同一prompt采样k次，得到k个样本，当前样本的baseline为其他k\-1个样本的平均奖励

RLOO优势估计方式如下：

$A(s, a_k) = R(s, a_k) - b(s, a_k)$

$R(s, a_k)$为当前样本的奖励，$b(s, a_k)$为当前样本对应的baseline



RLOO去掉了价值模型，和其他不使用价值模型的强化学习算法一样，会将句子的优势或奖励分配给句子中的每个token（所有token的优势相等）

此外，RLOO将一个序列当作一个action，而不是像PPO中将每个token作为一个action

在计算奖励时，将整个序列的KL散度之和与序列的奖励结合作为最终的奖励:

```Python
sequence_kl = kl.sum(1)
non_score_reward = -args.kl_coef * sequence_kl
rlhf_reward = non_score_reward + scores
```

在计算新旧策略的概率比时，使用整个序列每个token的log\_prob之和计算比值：

```Python
new_logprobs = new_logprobs.sum(1)
mb_logprobs = mb_logprobs.sum(1)
logprobs_diff = new_logprobs - mb_logprobs
ratio = torch.exp(logprobs_diff)
```

在损失计算中，trl中实现的RLOO没有使用REINFORCE loss，而是使用的PPO loss

PPO loss

```Python
pg_losses = -mb_advantage * ratio
pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
pg_loss_max = torch.max(pg_losses, pg_losses2)
pg_loss = pg_loss_max.mean()
```

REINFORCE loss

```Python
pg_losses = -new_logprobs * mb_advantage 
pg_loss = pg_losses.mean()
```

作者声明REINFORCE loss是PPO loss的一个特例，在on policy场景下，两种损失在反向传播计算梯度时是等价的，可参考如下代码：

```Python
import torch.nn.functional as F
from torch import LongTensor, Tensor, gather, no_grad

action = LongTensor([1])
advantage = Tensor([1.0])
logits = Tensor([[1.0, 2.0, 1.0, 1.0]])
logits.requires_grad = True
all_logprob = F.log_softmax(logits, dim=-1)
with no_grad():
    old_logprob = gather(all_logprob, 1, action.unsqueeze(-1)).squeeze(-1)
logprob = gather(all_logprob, 1, action.unsqueeze(-1)).squeeze(-1)
ratio = (logprob - old_logprob).exp()
ppo_loss = (ratio * advantage).mean() # [πθ(at | st) / πθ_old(at | st)* At]
# when the πθ and πθ_old are the same, the ratio is 1, and PPO's clipping has no effect
ppo_loss.backward()
print(f"{logits.grad=}") # tensor([[-0.1749, 0.5246, -0.1749, -0.1749]])
logits2 = Tensor([[1.0, 2.0, 1.0, 1.0]])
logits2.requires_grad = True
all_logprob2 = F.log_softmax(logits2, dim=-1)
logprob2 = gather(all_logprob2, 1, action.unsqueeze(-1)).squeeze(-1)
reinforce_loss = logprob2 * advantage # [log πθ(at | st)* At]
reinforce_loss.mean().backward()
print(f"{logits2.grad=}") # tensor([[-0.1749, 0.5246, -0.1749, -0.1749]])
```

## ReMax

REINFORCE方法衍生的另外一种方法ReMax原理和RLOO类似，不同的地方在于计算baseline和奖励的方法。

baseline

RLOO对同一输入采样k个输出，使用其他k\-1的输出的平均奖励作为baseline

ReMax使用模型贪婪采样(temp = 0.0)（do sample=False）的回答得到的奖励作为baseline

reward

对于奖励的计算，ReMax使用如下方法计算奖励（token粒度）：

```Python
def compute_returns(self, kl, reward_score, sequence_lengths):
    returns = torch.zeros_like(kl)
    
    batch_size = kl.shape[0]
    
    for j in range(batch_size):
        cumulative_reward = reward_score[j]
        cumulative_kl = 0
        for i in reversed(range(sequence_lengths[j])):
            cumulative_kl = kl[j, i]
    
            cumulative_reward *= self.args.gamma
            returns[j, i] += cumulative_kl + cumulative_reward   
    return returns
```

可以看到，ReMax将当前token对应的KL散度和当前token对应的奖励结合起来作为最终的回报或者奖励。但是，其奖励分数是针对一个完整句子给出的，是句子粒度的奖励，如何分配到每个token上呢？

在PPO中，只给最后一个token（\<eos\>）赋予完整奖励（将奖励归因于最后一个token），其他位置token的奖励为0

在ReMax中，越接近最后一个token位置的token获得更多的奖励，并往前逐渐衰减。

在RLOO中每个token的奖励和句子的奖励是相等的。


### 当前 ReMax 代码实现解读

#### 1. ReMax baseline 的计算

对于同一个 prompt，当前策略会生成两份回答：

1. 使用随机采样（`do_sample=True`）生成训练回答；
2. 使用贪心解码（`do_sample=False`）生成 baseline 回答。

二者分别经过奖励函数打分，最终的序列级相对奖励为：

$$
A_{\text{ReMax}}
=R(\text{随机回答})-R(\text{贪心回答})
$$

对应代码：

```Python
reward_scores = scores - baseline_scores
```

例如，正确性奖励规定回答正确为 `+1`、错误为 `-1`：

| 随机回答 | 贪心回答 | ReMax 相对奖励 |
| --- | --- | ---: |
| 正确（+1） | 错误（-1） | +2 |
| 错误（-1） | 正确（+1） | -2 |
| 正确（+1） | 正确（+1） | 0 |
| 错误（-1） | 错误（-1） | 0 |

贪心回答的奖励相当于一个与问题相关的 baseline：如果一个问题本身比较简单，随机回答必须优于模型的贪心回答，才能得到正的相对奖励。

#### 2. 参考模型 KL 惩罚

代码先计算随机回答中的每个 token 在当前策略和参考策略下的 log probability：

```Python
kl = logprobs - ref_logprobs
kl_reward = -args.kl_coef * kl
```

这里传给 `compute_returns()` 的 `kl` 实际上是 `kl_reward`。它更准确地说是**逐 token 的有符号 KL 惩罚估计**，而不是对完整词表概率分布单独计算出的 KL 散度。

例如：

```text
policy logp = -0.2
reference logp = -0.4
kl_coef = 0.05

kl = -0.2 - (-0.4) = 0.2
kl_reward = -0.05 × 0.2 = -0.01
```

当前策略给这个 token 的概率高于参考策略，因此获得 `-0.01` 的惩罚，防止训练后的策略偏离初始模型过远。

#### 3. 将句子奖励分配给 token

`reward_score` 是整条回答的序列级奖励，但语言模型的策略梯度需要落实到生成的 token。当前实现从回答末尾向前遍历，并按照 `gamma` 对句子奖励进行折扣：

$$
G_i=r_i^{\text{KL}}+\gamma^{T-i}A_{\text{ReMax}}
$$

其中：

- $G_i$ 是位置 $i$ 的 return/advantage；
- $r_i^{\text{KL}}$ 是位置 $i$ 的 KL reward；
- $A_{\text{ReMax}}$ 是整条回答的相对奖励；
- $\gamma$ 是折扣因子；
- 越靠近回答末尾，折扣次数越少，获得的序列奖励越大。

假设：

```text
reward_score = 2
gamma = 0.95
三个 token 的 kl_reward = [-0.01, -0.02, -0.03]
```

严格按照当前 `compute_returns()` 的更新顺序，从后向前得到：

| token 位置 | 计算过程 | return |
| ---: | --- | ---: |
| 2 | $-0.03 + 2 \times 0.95$ | 1.87 |
| 1 | $-0.02 + 2 \times 0.95^2$ | 1.785 |
| 0 | $-0.01 + 2 \times 0.95^3$ | 1.70475 |

因此，越接近答案末尾的 token 获得越强的序列级训练信号，越靠近答案开头则衰减越多。

#### 4. 与 RLOO、PPO 的对比需要区分 reward 和 advantage

上面的对比是一种简化描述。阅读实现时，需要区分以下三个概念：

- **原始序列奖励（score reward）**：奖励模型对完整回答给出的标量；
- **逐 token reward**：将序列奖励和每个位置的 KL 惩罚组合后的奖励；
- **return/advantage**：真正乘到 token log probability 上、用于策略梯度的训练信号。

PPO 可以只把原始序列奖励加到最后一个有效 token，但经过 return 或 GAE 计算后，前面的 token 仍可能得到非零 advantage。因此，“PPO 只有最后一个 token 有奖励”通常指原始 score reward 的放置位置，并不代表其他 token 最终没有梯度。

RLOO 通常先在序列级计算 leave-one-out advantage，再将同一个序列 advantage 用于该回答中的 token；逐 token KL 项仍可因位置不同而不同。因此，“每个 token 奖励完全相等”也是对序列级 advantage 的简化描述。

#### 5. 当前实现中的边界和命名问题

1. `cumulative_kl` 并没有真正累计 KL：

   ```Python
   cumulative_kl = kl[j, i]
   ```

   每一步只取当前位置的 KL reward，不会把后续 token 的 KL reward 累加进来。这个变量更适合命名为 `token_kl_reward`。

2. 奖励在写入最后位置前先乘了一次 `gamma`：

   ```Python
   cumulative_reward *= self.args.gamma
   returns[j, i] += cumulative_kl + cumulative_reward
   ```

   因此末端位置得到的是 `gamma * reward_score`，不是未经折扣的完整 `reward_score`。

3. 需要检查 `sequence_lengths` 的含义：训练代码将它构造为“最后一个有效 token 的下标”，但循环使用：

   ```Python
   range(sequence_lengths[j])
   ```

   `range` 不包含终点，因此最后一个有效 token 可能没有参与 return 计算。如果目标是遍历下标 `0` 到 `sequence_lengths[j]` 的全部有效位置，需要复核是否应写成：

   ```Python
   range(sequence_lengths[j] + 1)
   ```

综上，这份代码不是只使用标准 REINFORCE 目标的纯 ReMax，而是组合了：

```text
ReMax 贪心 baseline
        + 参考模型逐 token KL 约束
        + 折扣后的序列奖励分配
        + PPO clipped policy loss
```

理解和复现时，应以当前代码的实际张量计算为准，并将它与论文或其他框架中的标准 ReMax/RLOO 定义区分开。
