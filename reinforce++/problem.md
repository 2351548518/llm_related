重要性采样 是什么？

在kl散度的计算上，REINFORCE\+\+\-baseline使用k2，GRPO中使用的是k3
k1 k2 k3 分别是 什么？


# 重要性采样与 PPO 笔记

## 一、为什么需要重要性权重

核心原因是：我们想优化当前策略 $\pi_\theta$ 的表现，但手里的回答是旧策略 $\pi_{\text{old}}$ 生成的。两种策略产生数据的概率不同，直接把旧数据当作当前策略的数据会产生偏差。重要性权重就是用来修正这种“采样分布不一致”的。

强化学习希望最大化当前策略的期望奖励：

$$
J(\theta)
=
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[f(s,a)\right]
$$

离散情况下：

$$
J(\theta)
=
\sum_a \pi_\theta(a\mid s)f(s,a)
$$

这个期望要求样本来自当前策略 $\pi_\theta$。但是 PPO 已经用旧策略生成了一批回答：

$$
a\sim\pi_{\text{old}}(\cdot\mid s)
$$

如果每次更新一次参数都重新生成回答，训练成本会非常高。因此 PPO 希望一批 rollout 能进行多轮更新。问题是：更新一次以后，模型已经变成了 $\pi_\theta$，而手里的数据仍然来自 $\pi_{\text{old}}$。

## 二、重要性采样如何修正分布

当前策略的期望可以改写为：

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi_\theta}[f(a)]
&=
\sum_a \pi_\theta(a)f(a)\\
&=
\sum_a
\pi_{\text{old}}(a)
\frac{\pi_\theta(a)}{\pi_{\text{old}}(a)}
f(a)\\
&=
\mathbb{E}_{a\sim\pi_{\text{old}}}
\left[
\frac{\pi_\theta(a)}{\pi_{\text{old}}(a)}f(a)
\right].
\end{aligned}
$$

因此，即使样本来自旧策略，也可以乘上重要性权重：

$$
r(a)=
\frac{\pi_\theta(a)}{\pi_{\text{old}}(a)}
$$

重要性权重的本质是告诉优化器：这个由旧策略生成的样本，在当前策略下应该占多大比重。它不是额外奖励，而是一个概率分布修正系数。

## 三、重要性采样例子

假设模型在某个状态下只能选择 A 或 B。旧策略的概率为：

$$
\pi_{\text{old}}(A)=0.8,\qquad
\pi_{\text{old}}(B)=0.2
$$

因此旧策略生成 100 个样本时，大约会得到 80 个 A 和 20 个 B。经过一次更新，当前策略变成：

$$
\pi_\theta(A)=0.5,\qquad
\pi_\theta(B)=0.5
$$

假设：

$$
f(A)=1,\qquad f(B)=3
$$

真正想计算的当前策略期望是：

$$
\mathbb{E}_{\pi_\theta}[f]
=0.5\times1+0.5\times3=2
$$

但直接使用旧策略分布求平均会得到：

$$
0.8\times1+0.2\times3=1.4
$$

结果不是当前策略下的正确期望，因为旧数据中的 A 太多、B 太少。

对于 A：

$$
r(A)=
\frac{\pi_\theta(A)}{\pi_{\text{old}}(A)}
=\frac{0.5}{0.8}=0.625
$$

A 在旧数据中出现得太多，所以每个 A 样本的权重应该降低。

对于 B：

$$
r(B)=
\frac{\pi_\theta(B)}{\pi_{\text{old}}(B)}
=\frac{0.5}{0.2}=2.5
$$

B 在旧数据中出现得太少，所以每个 B 样本的权重应该提高。

重新计算：

$$
\begin{aligned}
\mathbb{E}_{\pi_{\text{old}}}[r(a)f(a)]
&=0.8\times0.625\times1
+0.2\times2.5\times3\\
&=0.5+1.5\\
&=2.
\end{aligned}
$$

这就用旧策略的数据恢复出了当前策略的期望。

## 四、放到 PPO 中是什么意思

PPO 中通常不是直接使用奖励 $f(a)$，而是使用优势 $A_t$：

$$
L(\theta)=
\mathbb{E}_{a_t\sim\pi_{\text{old}}}
\left[r_t(\theta)A_t\right]
$$

其中：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
     {\pi_{\text{old}}(a_t\mid s_t)}
$$

可以这样理解：

- $A_t>0$：这个 token 比预期好，希望提高它的概率；
- $A_t<0$：这个 token 比预期差，希望降低它的概率；
- $r_t$：修正这个旧样本在当前策略下应有的代表性。

代码中的：

```python
logprobs_diff = new_logprobs - mb_logprobs
ratio = torch.exp(logprobs_diff)
```

正是在计算：

$$
\begin{aligned}
\text{ratio}
&=\exp\left(\log\pi_\theta-\log\pi_{\text{old}}\right)\\
&=\frac{\pi_\theta}{\pi_{\text{old}}}.
\end{aligned}
$$

其中：

- `mb_logprobs`：生成 rollout 时保存的旧策略概率；
- `new_logprobs`：当前模型重新计算出的概率；
- `ratio`：当前概率除以旧概率。

代码没有单独保存一个 `old_policy` 模型，保存下来的 `mb_logprobs` 就相当于旧策略的快照。对应位置见 [train_reinforce++.py:367](</C:/q00520820/Learn_MLLMs/llm_related/reinforce++/train_reinforce++.py:367>)。

## 五、为什么还需要 PPO clipping

重要性采样虽然能够修正分布，但可能产生非常大的权重。例如前面的 B 有：

$$
r(B)=2.5
$$

更极端地，如果：

$$
\pi_{\text{old}}(B)=0.001,\qquad
\pi_\theta(B)=0.5
$$

则：

$$
r(B)=\frac{0.5}{0.001}=500
$$

一个样本就可能主导整次训练更新。

因此 PPO 使用：

$$
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)
$$

例如 $\epsilon=0.2$ 时，希望有效比率大致限制在 $[0.8,1.2]$：

$$
L^{CLIP}
=
\min\left(
r_tA_t,
\operatorname{clip}(r_t,0.8,1.2)A_t
\right)
$$

假设某个样本：

$$
A_t=3,\qquad r_t=2.5
$$

不 clipping 时：

$$
r_tA_t=2.5\times3=7.5
$$

clipping 后：

$$
1.2\times3=3.6
$$

PPO 会选择较保守的 $3.6$，避免当前策略因为少数旧样本而发生过大的变化。

因此：

- 重要性采样负责修正 old/new 之间的采样分布差异；
- clipping 负责防止重要性权重过大导致训练不稳定；
- clipping 会引入一定偏差，但换来了稳定性。

## 六、为什么第一轮更新时 ratio 接近 1

rollout 刚生成时，当前模型还没有更新：

$$
\pi_\theta=\pi_{\text{old}}
$$

因此：

$$
r_t=\frac{\pi_\theta}{\pi_{\text{old}}}=1
$$

完成第一轮参数更新后：

$$
\pi_\theta\neq\pi_{\text{old}}
$$

第二轮、第三轮继续复用同一批 rollout 时，ratio 才会逐渐偏离 1。这正是重要性采样主要发挥作用的地方。

完整过程是：

```text
当前模型生成 rollout
        ↓
保存生成时的 old log-prob
        ↓
第 1 轮 PPO 更新：new ≈ old，ratio ≈ 1
        ↓
模型参数发生变化
        ↓
第 2 轮 PPO 更新：new ≠ old，使用 ratio 修正
        ↓
继续复用同一批 rollout
        ↓
clipping 防止模型离 old policy 太远
```

## 七、LLM 中的具体例子

假设 prompt 是：

```text
2 + 3 等于多少？
```

旧策略生成答案时，在某个位置有：

$$
\pi_{\text{old}}(\text{“5”})=0.6
$$

训练一轮以后，当前策略变成：

$$
\pi_\theta(\text{“5”})=0.75
$$

那么这个 token 的重要性比率为：

$$
r_t=\frac{0.75}{0.6}=1.25
$$

说明当前策略生成“5”的概率比 rollout 时高了 25%。如果“5”的优势为正，普通重要性采样会继续按 $1.25A_t$ 推高它；但若 $\epsilon=0.2$，PPO 最多采用约 $1.2A_t$，防止模型在同一批数据上过度更新。

如果另一个错误 token “6” 的概率从 $0.4$ 降到 $0.25$：

$$
r_t=\frac{0.25}{0.4}=0.625
$$

说明当前策略已经大幅降低“6”的概率。如果“6”的优势是负数，PPO clipping 会限制继续降低它的幅度，避免一次更新走得过远。

## 八、公式中的 $s$、$a$ 和 $f(s,a)$

前面的通用公式中：

$$
J(\theta)=
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[f(s,a)\right]
$$

这里的符号含义是：

- $s$：state，状态；
- $a$：action，动作；
- $\pi_\theta(a\mid s)$：模型在状态 $s$ 下选择动作 $a$ 的概率；
- $f(s,a)$：选择动作 $a$ 之后，我们关心的某个数值，例如奖励、未来累计奖励或优势。

前面使用 $f(s,a)$ 只是为了介绍重要性采样的通用公式，它不是某个固定的强化学习变量。

### 1. $s$ 是什么

在传统强化学习中，$s$ 表示智能体当前看到的环境状态，例如下棋时的当前棋盘局面。

在 LLM 强化学习中，状态通常是：

$$
s_t=(x,y_{<t})
$$

其中：

- $x$：用户输入的 prompt；
- $y_{<t}=y_1,\ldots,y_{t-1}$：模型已经生成的 token；
- $s_t$：prompt 加上当前已经生成的文本前缀。

例如 prompt 是：

```text
计算 2 + 3。
```

模型已经生成：

```text
2 + 3 =
```

那么当前状态 $s_t$ 可以理解为：

```text
用户：计算 2 + 3。
助手：2 + 3 =
```

也就是模型此时所看到的全部上下文。

### 2. $a$ 是什么

$a$ 表示在状态 $s$ 下采取的动作。在 LLM 中，一个动作通常就是模型生成的下一个 token：

$$
a_t=y_t
$$

例如模型可以选择 `5`，也可以选择 `6`：

$$
\pi_\theta(\text{“5”}\mid s)=0.8,
\qquad
\pi_\theta(\text{“6”}\mid s)=0.1
$$

其中 $\pi_\theta(\cdot\mid s)$ 表示状态 $s$ 下所有可能动作的概率分布，公式中的“$\cdot$”是“所有可能动作”的占位符。

### 3. $f(s,a)$ 是什么

$f(s,a)$ 是一个通用函数，表示在状态 $s$ 下采取动作 $a$ 后，这个选择有多好。它具体是什么，取决于正在推导的目标。

#### 情况一：即时奖励 $r(s,a)$

如果动作做完马上就能得到奖励，那么：

$$
f(s,a)=r(s,a)
$$

例如：

$$
r(s,\text{“5”})=1,
\qquad
r(s,\text{“6”})=-1
$$

若：

$$
\pi_\theta(\text{“5”}\mid s)=0.8,
\qquad
\pi_\theta(\text{“6”}\mid s)=0.2
$$

则：

$$
\begin{aligned}
J(\theta)
&=0.8\times1+0.2\times(-1)\\
&=0.6.
\end{aligned}
$$

训练的目标就是调整 $\theta$，让正确动作的概率上升，从而让 $J(\theta)$ 变大。

#### 情况二：未来累计奖励 $Q(s,a)$

LLM 每次只生成一个 token，但通常要等完整回答生成完以后才能判断答案是否正确。例如模型生成第一个 token `<think>` 时，仅凭这个 token 不能判断最终答案是否正确。

这时可以使用：

$$
f(s_t,a_t)=Q(s_t,a_t)
$$

其中：

$$
Q(s_t,a_t)=
\mathbb{E}\left[
\text{从 }s_t\text{ 选择 }a_t\text{ 后的未来累计奖励}
\right]
$$

它表示：在当前文本前缀下生成这个 token，之后继续生成，最终预计能够获得多少奖励。

例如：

```text
状态 s_t：计算 2 + 3。我们可以把两个数……
动作 a_t：相加
```

虽然“相加”本身没有立即得到奖励，但它可能使模型更有可能在最后回答 5，因此这个动作具有较高的未来价值。

#### 情况三：优势 $A(s,a)$

PPO 通常不直接使用 $Q(s,a)$，而是使用优势：

$$
f(s,a)=A(s,a)
$$

优势定义为：

$$
A(s,a)=Q(s,a)-V(s)
$$

其中：

- $Q(s,a)$：选择动作 $a$ 后预计获得的未来奖励；
- $V(s)$：在状态 $s$ 下按照当前策略行动时的平均预期奖励；
- $A(s,a)$：动作 $a$ 比平均水平好多少。

因此：

- $A(s,a)>0$：动作比平均选择好，应该提高概率；
- $A(s,a)<0$：动作比平均选择差，应该降低概率；
- $A(s,a)=0$：动作和平均水平差不多。

PPO 的目标通常写成：

$$
L(\theta)=
\mathbb{E}_{s_t,a_t\sim\pi_{\text{old}}}
\left[r_t(\theta)A_t\right]
$$

所以在 PPO 语境下，通用公式中的 $f(s,a)$ 具体就是 $A_t$。

### 4. 与当前代码的对应关系

| 数学符号 | `train_reinforce++.py` 中的含义 |
| --- | --- |
| $s_t$ | prompt 加已经生成的 response 前缀 |
| $a_t$ | `mb_responses` 中第 $t$ 个 token |
| $\pi_{\text{old}}(a_t\mid s_t)$ | `mb_logprobs.exp()` |
| $\pi_\theta(a_t\mid s_t)$ | `new_logprobs.exp()` |
| $r_t(\theta)$ | `ratio` |
| $A_t$ | `mb_advantage` |
| 最终任务奖励 | `scores` |
| KL 修正后的奖励 | `rlhf_reward` |

对应代码：

```python
mb_advantage = advantages[micro_batch_inds]
mb_responses = responses[micro_batch_inds]
mb_logprobs = logprobs[micro_batch_inds]

new_logprobs = selective_log_softmax(logits, mb_responses)

logprobs_diff = new_logprobs - mb_logprobs
ratio = torch.exp(logprobs_diff)

pg_losses = -mb_advantage * ratio
```

这里：

- `mb_responses` 是动作 $a_t$；
- 模型输入的 prompt 和 response 前缀是状态 $s_t$；
- `ratio` 是重要性权重；
- `mb_advantage` 是代码中的 $f(s_t,a_t)$。

### 5. 当前代码的优势是怎么来的

当前代码先计算最终正确性奖励：

```python
score = +1  # 回答正确
score = -1  # 回答错误
```

然后加入 reference KL penalty：

$$
R_{\text{RLHF}}
=
R_{\text{task}}
-\beta\,\mathrm{KL}
$$

得到 `rlhf_reward`，接着：

```python
advantages = rlhf_reward
```

如果启用 baseline，则先减去同一 prompt 多个回答的平均奖励：

$$
A_i=R_i-\operatorname{mean}_{\text{group}}(R)
$$

最后再做 batch 标准化：

$$
A_i^{\text{norm}}
=
\frac{A_i-\operatorname{mean}(A)}
     {\operatorname{std}(A)+\epsilon}
$$

因此，当前代码中的 `mb_advantage` 大致表示：这个完整回答的奖励相对于 batch 或同组其他回答有多好。

需要注意：当前代码计算的是一个回答级标量优势，然后试图将它应用到回答里的每个 token，而不是为每个 token 单独估计不同的 $Q(s_t,a_t)$ 或 $A_t$。

### 6. 一个完整的 LLM 例子

假设：

$$
s_1=\text{“用户：2+3 等于多少？助手：”}
$$

模型选择：

$$
a_1=\text{“<think>”}
$$

于是下一个状态变成：

$$
s_2=\text{“用户：2+3 等于多少？助手：<think>”}
$$

接着选择：

$$
a_2=\text{“2+3=5”}
$$

状态继续变化：

$$
s_3=\text{“用户：2+3 等于多少？助手：<think>2+3=5”}
$$

最后生成完整回答：

```text
<think>2+3=5</think>
<answer>5</answer>
```

得到最终奖励：

$$
R=1
$$

整个回答可以表示为轨迹：

$$
\tau=(s_1,a_1,s_2,a_2,\ldots,s_T,a_T)
$$

更完整的 LLM 强化学习目标是：

$$
J(\theta)=
\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

或者把 prompt 和完整回答分开写成：

$$
J(\theta)=
\mathbb{E}_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
[R(x,y)]
$$

其中：

- $x$：prompt；
- $y=(y_1,\ldots,y_T)$：完整回答；
- $s_t=(x,y_{<t})$：生成第 $t$ 个 token 时的状态；
- $a_t=y_t$：第 $t$ 个 token；
- $R(x,y)$：完整回答的最终奖励。

最直观的理解是：$s$ 是模型已经看到了什么，$a$ 是模型接下来生成什么，$f(s,a)$ 是这个选择带来的价值；在 PPO 代码中，$f(s,a)$ 具体对应优势 $A_t$。

## 九、重要性采样与 clipping 的一句话总结

PPO 想优化当前策略，但为了节约 rollout 成本，手里复用的是旧策略生成的数据。重要性权重 $\pi_\theta/\pi_{\text{old}}$ 用来修正两者的采样概率差异；clipping 再限制这种修正不要过大，从而在“重复利用数据”和“保持训练稳定”之间取得平衡。

严格来说，PPO 的 token-level ratio 是一个局部 surrogate：状态和前缀本身仍来自旧策略，且 clipping 会故意引入偏差。因此它不是对完整当前策略期望的完全精确估计，而是在新旧策略足够接近时使用的稳定近似。
