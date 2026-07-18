# GRPO 参数更新笔记：从 Reward 到 `optimizer.step()`

> 本文只讨论当前仓库采用的 **Outcome Supervision GRPO**：每条完整回答得到一个
> 序列级奖励，组内标准化后形成回答优势；同一回答的所有有效 token 共享这个优势。

这篇笔记回答三个核心问题：

1. 为什么同时需要旧策略和当前策略？
2. 重要性比率究竟有什么作用？
3. GRPO 的梯度如何从 token loss 传回 Transformer，并更新模型参数？

对应教学代码：

- [`train.py`](./train.py)：采样、奖励、优势、GRPO loss 和参数更新。
- [`reward_func.py`](./reward_func.py)：规则奖励函数。
- [`GRPO_NOTE.md`](./GRPO_NOTE.md)：PPO、GRPO 和 KL 的完整背景笔记。

---

## 1. 一张图看懂整个更新流程

```text
当前模型生成一组回答
        ↓
将生成时的模型视为旧策略，保存 old log-prob
        ↓
奖励函数给每条完整回答打分
        ↓
在同一问题的回答组内计算相对优势
        ↓
当前模型重新计算回答 token 的 current log-prob
        ↓
计算 current / old 重要性比率
        ↓
构造逐 token PPO-clip loss
        ↓
使用 action_mask 去掉 padding，再按回答长度求平均
        ↓
loss.backward() 计算参数梯度
        ↓
optimizer.step() 更新当前模型参数
```

可以把 GRPO 的核心更新浓缩为：

$$
\boxed{
\text{回答优势决定更新方向}
\;+\;
\text{重要性比率决定 token 梯度权重}
\;+\;
\text{clip 限制有利方向上的过度更新}
}
$$

---

## 2. 符号约定

| 符号 | 含义 |
| --- | --- |
| $q$ | 输入问题或 prompt |
| $o_i$ | 同一问题采样得到的第 $i$ 条完整回答 |
| $o_{i,t}$ | 第 $i$ 条回答的第 $t$ 个 token |
| $o_{i,<t}$ | 生成第 $t$ 个 token 之前的回答前缀 |
| $T_i=\lvert o_i\rvert$ | 第 $i$ 条回答的有效 completion token 数 |
| $G$ | 同一问题采样的回答数量 |
| $R_i$ | 第 $i$ 条回答获得的完整序列奖励 |
| $\widehat A_i$ | 第 $i$ 条回答的组内相对优势 |
| $\pi_{\mathrm{old}}$ | 生成当前 rollout 数据时的旧策略 |
| $\pi_\theta$ | 当前正在训练的策略 |
| $\pi_{\mathrm{ref}}$ | 可选的冻结参考策略，用于 KL 约束 |
| $\rho_{i,t}$ | 当前策略与旧策略对已采样 token 的概率比 |
| $\epsilon$ | PPO/GRPO 的裁剪半径 |
| $m_{i,t}$ | action mask；有效 token 为 1，padding 为 0 |

---

## 3. 为什么需要旧策略和当前策略

### 3.1 旧策略负责产生数据

对同一个问题，旧策略采样 $G$ 条回答：

$$
o_i
\overset{\mathrm{i.i.d.}}{\sim}
\pi_{\mathrm{old}}(\cdot\mid q),
\qquad i=1,\ldots,G.
$$

生成回答时的 token 选择是离散采样，不能对“为什么采到了这个 token”直接进行
普通反向传播。采样完成以后，回答的 token IDs 已经固定，后续训练只能重新计算
模型对这些已生成 token 的概率。

在本仓库中，采样阶段会保存旧策略对每个回答 token 的 log-prob：

```python
with torch.no_grad():
    old_action_log_probs = self.get_action_log_probs(
        self.model,
        prompt_response_ids,
        attention_mask,
        num_actions,
    )
```

数学上：

$$
\ell_{i,t}^{\mathrm{old}}
=
\log
\pi_{\mathrm{old}}
(o_{i,t}\mid q,o_{i,<t}).
$$

`torch.no_grad()` 表示旧 log-prob 只是缓存数据，不建立反向传播计算图。

### 3.2 当前策略负责接收梯度

训练阶段，当前策略对同一批固定回答重新 forward：

$$
\ell_{i,t}
=
\log
\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

代码为：

```python
action_log_probs = self.get_action_log_probs(
    model,
    prompt_response_ids,
    attention_mask,
    num_actions,
)
```

当前 log-prob 保留计算图，因此梯度可以依次传回：

```text
token log-prob
→ log-softmax
→ logits
→ LM Head
→ Transformer 各层
→ 模型参数 θ
```

### 3.3 为什么旧策略和当前策略一开始相同

rollout 刚生成时，旧策略就是当前模型在采样时的快照：

$$
\pi_\theta=\pi_{\mathrm{old}}.
$$

执行一次参数更新后：

$$
\pi_\theta\ne\pi_{\mathrm{old}}.
$$

但这批回答仍由旧策略生成。如果同一批经验被拆成多个 mini-batch，或者被重复训练
多轮，就必须保留“采样时的概率”作为比较基准。

### 3.4 旧策略不一定需要保存完整模型

为了计算新旧概率比，只需要：

- 已生成的 token IDs；
- 旧策略对这些 token 的 log-prob；
- 有效 token mask；
- reward 和 advantage。

因此很多实现不长期保留一个完整 `old_model`，而是缓存
`old_action_log_probs`。旧策略与参考策略不是同一个角色：

- `old` 记录“这批数据是谁生成的”；
- `ref` 约束“当前模型不要偏离谁太远”。

---

## 4. 从完整回答奖励得到组内相对优势

### 4.1 奖励是一条回答一个标量

奖励模型或规则函数对每条完整回答打分：

$$
R_1,R_2,\ldots,R_G.
$$

例如：

$$
R=[3.5,\ 1.5,\ 1.0,\ 0.0].
$$

这些 reward 在 `torch.no_grad()` 内计算，只作为策略梯度的外部训练信号。

### 4.2 在同一问题的回答组内标准化

定义：

$$
\mu_R
=
\frac{1}{G}\sum_{i=1}^{G}R_i,
$$

$$
\sigma_R
=
\operatorname{std}(R_1,\ldots,R_G),
$$

$$
\widehat A_i
=
\frac{R_i-\mu_R}
{\sigma_R+\varepsilon_{\mathrm{num}}}.
$$

本仓库调用 `torch.std()` 的默认样本标准差。对于：

$$
R=[3.5,\ 1.5,\ 1.0,\ 0.0],
$$

有：

$$
\mu_R=1.5,
\qquad
\sigma_R\approx1.472,
$$

所以：

$$
\widehat A
\approx
[1.359,\ 0,\ -0.340,\ -1.019].
$$

解释：

- 第一条回答明显高于组内平均，应该提高其概率；
- 第二条回答等于组内平均，几乎没有策略梯度；
- 第三、第四条回答低于组内平均，应该降低其概率。

对应代码：

```python
mean_group_rewards = rewards.mean()
std_group_rewards = rewards.std()

advantages = (
    rewards - mean_group_rewards
) / (std_group_rewards + 1e-8)
```

### 4.3 同一回答的所有有效 token 共享优势

在 Outcome Supervision GRPO 中：

$$
\widehat A_{i,t}
=
\widehat A_i,
\qquad
t=1,\ldots,T_i.
$$

实现时不需要真的复制 $T_i$ 份优势。假设：

- `ratios` 形状为 `[B, T]`；
- `advantages` 形状为 `[B]`。

代码使用：

```python
token_objective = ratios * advantages.unsqueeze(1)
```

`advantages.unsqueeze(1)` 的形状为 `[B, 1]`，PyTorch 会自动广播到 `[B, T]`。

所有 token 共享优势不代表参数梯度完全相同，因为：

$$
\nabla_\theta
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

仍随 token 和前缀变化。

---

## 5. 如何得到已生成 token 的 log-prob

语言模型输出：

$$
\mathrm{logits}\in
\mathbb R^{B\times L\times V},
$$

其中 $V$ 是词表大小。

首先转换为 log-prob：

```python
log_probs = F.log_softmax(
    logits[:, :-1, :],
    dim=-1,
)
```

自回归模型当前位置的 logits 预测下一个 token，因此：

- logits 删除最后一个位置；
- labels 删除第一个 token；
- 两者在时间维度上对齐。

然后用 `gather` 取出实际生成 token 的 log-prob：

```python
log_probs_labels = log_probs.gather(
    dim=-1,
    index=input_ids[:, 1:].unsqueeze(-1),
)
```

最后只保留 completion 部分：

```python
action_log_probs = (
    log_probs_labels
    .squeeze(-1)[:, -num_actions:]
)
```

注意，GRPO 不是对整个词表的所有 token 都计算策略 loss，而是取出这次 rollout
实际生成的 token 对应的 log-prob。

---

## 6. 重要性比率是什么

### 6.1 定义

对第 $i$ 条回答的第 $t$ 个 token：

$$
\rho_{i,t}
=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})
}.
$$

在 log 空间计算：

$$
\rho_{i,t}
=
\exp\left(
\ell_{i,t}-\ell_{i,t}^{\mathrm{old}}
\right).
$$

对应代码：

```python
ratio = torch.exp(
    action_log_probs - old_action_log_probs
)
```

### 6.2 ratio 的直观意义

它回答的是：

> 对这个由旧策略生成的 token，当前策略给出的概率变成了旧概率的多少倍？

| 旧概率 | 当前概率 | ratio | 含义 |
| ---: | ---: | ---: | --- |
| $0.2$ | $0.2$ | $1$ | 概率没有变化 |
| $0.2$ | $0.3$ | $1.5$ | 当前概率是原来的 $1.5$ 倍 |
| $0.2$ | $0.1$ | $0.5$ | 当前概率降为原来的一半 |

因此：

- $\rho=1$：当前策略和旧策略对该 token 的概率相同；
- $\rho>1$：当前策略提高了该 token 的概率；
- $\rho<1$：当前策略降低了该 token 的概率。

ratio 不是 reward，也不是概率本身，而是无量纲的相对变化倍数。

### 6.3 第一个作用：对旧策略数据进行重加权

标准重要性采样恒等式为：

$$
\mathbb E_{o\sim\pi_\theta}[f(o)]
=
\mathbb E_{o\sim\pi_{\mathrm{old}}}
\left[
\frac{\pi_\theta(o)}
{\pi_{\mathrm{old}}(o)}
f(o)
\right].
$$

它表达的是：

> 虽然数据由旧策略产生，但可以根据新旧策略概率比重新调整样本贡献。

PPO/GRPO 把这个思想用于逐 token surrogate objective。需要注意，GSPO 论文质疑的
是 Outcome Supervision GRPO 使用单 token ratio 做离策略校正的稳定性，不是否认
重要性采样本身。

### 6.4 第二个作用：衡量当前策略偏离旧策略多少

如果：

$$
\rho_{i,t}\approx1,
$$

当前策略没有明显偏离旧策略。

如果：

$$
\rho_{i,t}=3,
$$

当前策略把该 token 的概率提高到了原来的三倍。

PPO/GRPO 使用 ratio 判断更新是否过大，并通过裁剪限制有利方向上的过度变化。

---

## 7. ratio 等于 1，为什么仍然有梯度

这是理解策略梯度最关键的一步。

只训练一轮时，本仓库使用：

```python
old_action_log_probs = action_log_probs.detach()
```

设：

$$
\ell=\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

那么：

$$
\rho
=
\exp\left(
\ell-\operatorname{sg}(\ell)
\right)
=1,
$$

其中 $\operatorname{sg}$ 表示 stop-gradient，也就是 PyTorch 的 `detach()`。

虽然前向数值为 1，但：

$$
\nabla_\theta\operatorname{sg}(\ell)=0.
$$

因此：

$$
\begin{aligned}
\nabla_\theta\rho
&=
\rho
\nabla_\theta
\left(
\ell-\operatorname{sg}(\ell)
\right)\\
&=
\rho\nabla_\theta\ell\\
&=
\nabla_\theta\ell.
\end{aligned}
$$

所以：

> ratio 数值等于 1，不代表 ratio 对模型参数的梯度等于 0。

如果错误地让分子和分母都保留同一个计算图：

$$
\rho=\exp(\ell-\ell)=1,
$$

那么两边梯度会抵消：

$$
\nabla_\theta(\ell-\ell)=0.
$$

模型就无法从策略项获得梯度。

---

## 8. PPO-clip 风格的 GRPO 目标

### 8.1 最大化目标与最小化 loss

GRPO 对每个 token 最大化：

$$
J_{i,t}
=
\min\left(
\rho_{i,t}\widehat A_i,\,
\operatorname{clip}
(\rho_{i,t},1-\epsilon,1+\epsilon)
\widehat A_i
\right).
$$

PyTorch 优化器最小化 loss，所以代码取负号：

$$
L_{i,t}^{\mathrm{policy}}
=
-J_{i,t}.
$$

对应代码：

```python
ratio_clipped = torch.clamp(
    ratio,
    1 - clip_eps,
    1 + clip_eps,
)

per_token_loss1 = (
    ratio * advantages.unsqueeze(1)
)
per_token_loss2 = (
    ratio_clipped * advantages.unsqueeze(1)
)

per_token_loss = -torch.min(
    per_token_loss1,
    per_token_loss2,
)
```

### 8.2 优势、ratio 和 clip 各自负责什么

$$
\boxed{
\begin{aligned}
\widehat A_i
&:\ \text{决定应该提高还是降低概率},\\
\rho_{i,t}
&:\ \text{衡量当前概率相对旧概率改变多少},\\
\operatorname{clip}
&:\ \text{限制有利方向上的过度改变}.
\end{aligned}
}
$$

当 $\widehat A_i>0$ 时，希望提高该回答 token 的概率；如果
$\rho_{i,t}>1+\epsilon$，继续提高的额外收益被截断。

当 $\widehat A_i<0$ 时，希望降低该回答 token 的概率；如果
$\rho_{i,t}<1-\epsilon$，继续降低的额外收益被截断。

---

## 9. GRPO 的梯度到底是什么

### 9.1 先忽略裁剪和 KL

单个 token 的 loss 为：

$$
L_{i,t}
=
-\rho_{i,t}\widehat A_i.
$$

由于旧 log-prob 是常量：

$$
\nabla_\theta\rho_{i,t}
=
\rho_{i,t}
\nabla_\theta
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

所以：

$$
\boxed{
\nabla_\theta L_{i,t}
=
-\widehat A_i
\rho_{i,t}
\nabla_\theta
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}
$$

这就是策略项真正传回 Transformer 的梯度。

### 9.2 正优势如何更新模型

当：

$$
\widehat A_i>0,
$$

梯度下降会提高已生成 token 的 log-prob：

$$
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
\uparrow.
$$

也就是提高好回答中 token 的生成概率。

### 9.3 负优势如何更新模型

当：

$$
\widehat A_i<0,
$$

梯度下降会降低已生成 token 的 log-prob：

$$
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
\downarrow.
$$

也就是降低差回答中 token 的生成概率。

模型不是直接执行：

```python
token_probability = new_probability
```

而是修改共享的 Transformer 参数。参数变化后，再次 forward 才会得到新的 token
概率。

---

## 10. 裁剪以后哪些 token 还有梯度

令：

$$
l=1-\epsilon,
\qquad
u=1+\epsilon.
$$

忽略裁剪边界处的次梯度选择，最大化目标的梯度系数为：

$$
c_{i,t}
=
\begin{cases}
\widehat A_i\rho_{i,t},
& \widehat A_i>0,\ \rho_{i,t}<u,\\
0,
& \widehat A_i>0,\ \rho_{i,t}>u,\\
0,
& \widehat A_i<0,\ \rho_{i,t}<l,\\
\widehat A_i\rho_{i,t},
& \widehat A_i<0,\ \rho_{i,t}>l.
\end{cases}
$$

于是：

$$
\nabla_\theta J_{i,t}
=
c_{i,t}
\nabla_\theta
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

程序最小化 $-J$，所以 loss 梯度多一个负号。

这意味着：

- 正优势且 ratio 超过上界：策略梯度归零；
- 负优势且 ratio 低于下界：策略梯度归零；
- 负优势且 ratio 很大：保留未裁剪梯度，且被大 ratio 放大。

### 10.1 为什么负优势的有效权重没有上界

当 $\widehat A<0$ 时：

$$
\min\left(
\rho\widehat A,\,
\operatorname{clip}(\rho,l,u)\widehat A
\right)
=
\widehat A\max(\rho,l).
$$

因此有效权重范围是：

$$
[1-\epsilon,+\infty).
$$

这里不是 `clip` 的输出超过了 $1+\epsilon$，而是 `min` 在负优势、大 ratio 时重新
选择了未裁剪项。

例如：

$$
\pi_{\mathrm{old}}=10^{-6},
\qquad
\pi_\theta=10^{-3},
$$

则：

$$
\rho
=
\frac{10^{-3}}{10^{-6}}
=
1000.
$$

若 $\widehat A=-1$、$\epsilon=0.2$：

$$
\rho\widehat A=-1000,
$$

$$
\operatorname{clip}(1000,0.8,1.2)\widehat A=-1.2,
$$

$$
\min(-1000,-1.2)=-1000.
$$

所以负优势下，极大的 token ratio 可能产生极大的梯度权重。

---

## 11. mask、长度归一化和总 loss

### 11.1 屏蔽 padding

代码先使用：

```python
per_token_loss = per_token_loss * action_mask
```

其中：

$$
m_{i,t}
=
\begin{cases}
1, & \text{有效 completion token},\\
0, & \text{padding 或无效位置}.
\end{cases}
$$

### 11.2 每条回答内部求平均

$$
L_i
=
\frac{
\sum_t m_{i,t}L_{i,t}
}{
\sum_t m_{i,t}
}.
$$

这使长回答和短回答原则上各贡献一个回答级平均 loss，不会仅因长回答 token 更多而
自然获得更大总权重。

### 11.3 batch 内回答求平均

$$
L
=
\frac{1}{B}
\sum_{i=1}^{B}L_i.
$$

对应代码：

```python
loss = (
    per_token_loss.sum(dim=1)
    / action_mask.sum(dim=1)
)
loss = loss.mean()
```

如果启用参考模型 KL，代码还会在每个有效 token 加上：

$$
\beta k_{3,i,t}.
$$

---

## 12. `loss.backward()` 做了什么

代码：

```python
loss.backward()
```

不会立刻修改参数。它只沿计算图计算：

$$
g
=
\nabla_\theta L,
$$

并把结果累积到每个参数的：

```python
parameter.grad
```

反向传播路径为：

```text
loss
→ token clipped objective
→ current / old ratio
→ current token log-prob
→ log-softmax
→ logits
→ Transformer 参数
```

没有梯度的量包括：

- reward；
- group advantage；
- old log-prob；
- token IDs；
- action mask；
- reference log-prob。

真正把梯度传回策略模型的是：

$$
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

---

## 13. `optimizer.step()` 如何更新模型

本仓库使用：

```python
self.optimizer = torch.optim.Adam(
    self.model.parameters(),
    lr=self.args.lr,
)
```

因此，所有 `requires_grad=True` 且参与前向计算的模型参数都可能被更新，包括：

- token embedding；
- Attention 的 Q、K、V、O 投影；
- MLP/FFN；
- LayerNorm；
- LM Head。

Adam 的简化更新为：

$$
m_k
=
\beta_1m_{k-1}
+
(1-\beta_1)g_k,
$$

$$
v_k
=
\beta_2v_{k-1}
+
(1-\beta_2)g_k^2,
$$

$$
\theta_{k+1}
=
\theta_k
-
\alpha
\frac{\widehat m_k}
{\sqrt{\widehat v_k}+\varepsilon_{\mathrm{Adam}}}.
$$

代码：

```python
loss.backward()

optimizer.step()
optimizer.zero_grad()
```

三个操作的区别是：

1. `backward()`：计算并累积梯度，参数尚未改变；
2. `step()`：根据梯度真正修改参数；
3. `zero_grad()`：清空旧梯度，准备下一轮。

更新的是整个共享模型，不是某个独立的“token 参数”。因此，一批回答产生的梯度也
可能改变模型在其他问题和其他 token 上的概率。

---

## 14. 梯度累积

当前配置：

```python
gradient_accumulation_steps = 2
```

每个 micro-batch 先执行：

```python
loss = loss / gradient_accumulation_steps
loss.backward()
```

连续两个 micro-batch 的梯度累积到 `.grad` 中，第二个 micro-batch 之后才执行：

```python
optimizer.step()
optimizer.zero_grad()
```

因此，一次真正的参数更新近似使用两个 micro-batch 的平均梯度。

---

## 15. 一次更新的完整时间线

假设 rollout 时模型参数为 $\theta_0$。

### 15.1 生成 rollout

使用：

$$
\pi_{\theta_0}
$$

生成回答，并保存：

$$
\ell_{i,t}^{\mathrm{old}}
=
\log\pi_{\theta_0}(o_{i,t}\mid q,o_{i,<t}).
$$

对这批数据而言：

$$
\pi_{\mathrm{old}}=\pi_{\theta_0}.
$$

### 15.2 第一次计算 loss

当前模型还未改变：

$$
\pi_\theta=\pi_{\theta_0}.
$$

因此 ratio 数值为：

$$
\rho_{i,t}=1.
$$

由于分母已经 stop-gradient，ratio 仍对当前模型有梯度。

### 15.3 第一次参数更新

执行：

```python
loss.backward()
optimizer.step()
```

得到：

$$
\theta_0\longrightarrow\theta_1.
$$

当前策略变为：

$$
\pi_\theta=\pi_{\theta_1},
$$

但缓存的旧 log-prob 仍来自 $\pi_{\theta_0}$。

### 15.4 重复使用同一批经验

重新 forward 后：

$$
\rho_{i,t}
=
\frac{
\pi_{\theta_1}(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_0}(o_{i,t}\mid q,o_{i,<t})
}.
$$

再次更新：

$$
\theta_1\longrightarrow\theta_2.
$$

同一批经验的旧策略分母仍然固定在 $\theta_0$。

### 15.5 生成下一批 rollout

使用更新后的 $\pi_{\theta_2}$ 生成下一批回答。对于下一批数据：

$$
\pi_{\mathrm{old}}
\leftarrow
\pi_{\theta_2}.
$$

旧策略不是永久固定的初始模型，而是每批 rollout 生成时的策略快照。

---

## 16. 一个完整的数值例子

### 16.1 组内奖励和优势

同一道题有 4 条回答：

$$
R=[3.5,\ 1.5,\ 1.0,\ 0.0].
$$

得到：

$$
\widehat A
\approx
[1.359,\ 0,\ -0.340,\ -1.019].
$$

选择第一条好回答：

$$
\widehat A_1=1.359.
$$

假设它有 3 个有效 token，旧策略概率为：

$$
p_{\mathrm{old}}
=[0.2,\ 0.5,\ 0.1],
$$

当前策略概率为：

$$
p_\theta
=[0.22,\ 0.45,\ 0.13].
$$

因此 token ratio 为：

$$
\rho
=
\left[
\frac{0.22}{0.2},\,
\frac{0.45}{0.5},\,
\frac{0.13}{0.1}
\right]
=
[1.1,\ 0.9,\ 1.3].
$$

令：

$$
\epsilon=0.2,
$$

裁剪后的 ratio 为：

$$
\operatorname{clip}(\rho,0.8,1.2)
=[1.1,\ 0.9,\ 1.2].
$$

逐 token 最大化目标为：

$$
J_t
=
\min\left(
\rho_t\widehat A_1,\,
\operatorname{clip}(\rho_t)\widehat A_1
\right).
$$

三个 token 分别得到：

$$
J
\approx
[1.495,\ 1.223,\ 1.631].
$$

第三个 token 的 ratio 为 $1.3$，正优势方向超过上界，所以只采用：

$$
1.2\times1.359\approx1.631.
$$

回答级平均目标：

$$
J_1
\approx
\frac{1.495+1.223+1.631}{3}
\approx
1.450.
$$

程序最小化：

$$
L_1=-J_1\approx-1.450.
$$

反向传播会总体提高这条好回答中 token 的概率，但第三个 token 已进入裁剪平坦区，
不会再从策略项获得继续提高概率的梯度。

### 16.2 一条负优势回答

第四条回答：

$$
\widehat A_4=-1.019.
$$

假设三个 token ratio 为：

$$
\rho=[0.5,\ 1.0,\ 1000].
$$

当 $\epsilon=0.2$ 时：

- ratio 为 $0.5$：低于下界 $0.8$，负优势方向被裁剪，策略梯度为 0；
- ratio 为 $1.0$：正常保留负优势梯度；
- ratio 为 $1000$：上侧不被负优势的 `min` 截断，保留极大的未裁剪梯度。

这说明 GRPO 的实际更新不是简单地“好回答全部增加、差回答全部减少”，而是：

> 每个 token 的梯度还会被自己的新旧概率比和裁剪状态进一步调节。

---

## 17. 与本仓库代码逐行对应

| 阶段 | `train.py` 关键位置 | 作用 |
| --- | ---: | --- |
| 创建优化器 | 179 | Adam 接收 `self.model.parameters()` |
| 生成经验 | 269 | 组织回答、概率、奖励和优势 |
| 保存旧 log-prob | 302–310 | `no_grad()` 下缓存旧策略概率 |
| 组内优势 | 376–384 | reward 标准化为回答优势 |
| 当前 log-prob | 413 | 当前模型重新 forward |
| 重要性比率 | 430–432 | `exp(current - old)` |
| 优势广播与 clip loss | 437–440 | 构造逐 token 策略 loss |
| mask 与长度平均 | 440–447 | 排除 padding，得到 batch loss |
| 反向传播 | 497 | 计算并累积 `.grad` |
| 更新参数 | 500 | `optimizer.step()` |
| 清空梯度 | 501 | `optimizer.zero_grad()` |
| 经验重复训练 | 527–530 | 多次使用缓存的 rollout |

---

## 18. 常见误区

### 18.1 “ratio 等于 1，所以没有梯度”

错误。旧 log-prob 已被 `detach`，因此前向值可以为 1，但梯度不会抵消。

### 18.2 “clip 会把所有 ratio 永远限制在裁剪区间”

错误。`clip` 的输出在区间内，但最终目标还会在未裁剪项和裁剪项之间取 `min`。
负优势、大 ratio 时会保留未裁剪项。

### 18.3 “优化器直接修改 token 概率”

错误。优化器修改 Transformer 参数；概率是参数更新后重新 forward 得到的结果。

### 18.4 “旧策略就是参考模型”

错误：

- 旧策略是 rollout 时的策略快照；
- 参考模型是可选的 KL 锚点。

### 18.5 “所有 GRPO 都必须一条回答共享一个优势”

本文明确限定为 Outcome Supervision GRPO。在这一设置中所有有效 token 共享回答
优势；不要把这一结论无条件推广到其他监督粒度。

### 18.6 “reward 会直接反向传播”

错误。reward 和 advantage 在本实现中都被当作常量系数。梯度通过当前策略的
token log-prob 传回模型。

---

## 19. 最后总结

GRPO 的参数更新可以写成：

$$
\boxed{
\theta
\leftarrow
\theta
+
\alpha
\frac{1}{B}
\sum_i
\frac{1}{T_i}
\sum_t
c_{i,t}
\nabla_\theta
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-
\text{KL 修正}
}
$$

其中 $c_{i,t}$ 由三个因素共同决定：

$$
c_{i,t}
=
f\left(
\widehat A_i,\,
\rho_{i,t},\,
\text{clip state}
\right).
$$

最值得记住的是：

1. 旧策略生成回答并提供旧概率基准；
2. 当前策略重新计算概率并接收梯度；
3. 回答优势决定提高还是降低概率；
4. 重要性比率连接旧数据与当前模型；
5. clip 限制有利方向上的过度更新；
6. `backward()` 只计算梯度；
7. `optimizer.step()` 才真正修改模型参数；
8. 下一批 rollout 会把更新后的当前策略作为新的旧策略。

---

## 参考资料

1. Zhihong Shao et al.,
   [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300),
   2024。
2. John Schulman et al.,
   [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347),
   2017。
3. Chujie Zheng et al.,
   [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071),
   2025。
