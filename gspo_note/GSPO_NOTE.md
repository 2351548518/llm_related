# GSPO 学习笔记：从 token 级 GRPO 到序列级策略优化

> **一句话理解：GSPO 保留 GRPO 的“同一问题生成一组回答、按组内奖励估计优势”，
> 但把重要性比率、裁剪和优化单元从单个 token 提升到整条响应序列。**

GSPO 的全称是 **Group Sequence Policy Optimization**。Qwen 团队在 2025 年提出
它，主要目标是解决 GRPO 在长回答、**大规模离策略更新**和 MoE 训练中出现的高方差
与不稳定问题。

本笔记先复习 GRPO，再逐步推导 GSPO，并用一个可手算的例子和两份代码说明如何
实现。文中的“论文认为/论文观察到”特指 GSPO 原论文的论点或实验结果，不把单篇
论文的经验结论写成对所有模型都成立的定理。

> **讨论范围：**本文只讨论 **Outcome Supervision（outcome-level）GRPO**：
> 每条完整响应得到一个最终奖励，再将组内归一化后的响应优势用于该响应的全部有效
> token。后文所说的“GRPO”均指这一设置。

---

## 1. 先看全局：GSPO 到底改了什么

同一个问题 $x$ 由旧策略生成 $G$ 条回答：

$$
\{y_i\}_{i=1}^{G}\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid x).
$$

每条完整回答得到一个序列级奖励 $r(x,y_i)$，再在组内标准化为优势
$\widehat A_i$。GRPO 和 GSPO 都使用这套采样与优势估计；两者主要区别是：

| 项目 | GRPO | GSPO |
|---|---|---|
| reward 粒度 | 通常是完整响应 | 完整响应 |
| advantage 粒度 | 一条响应一个标量，所有有效 token 共享 | 一条响应一个标量 |
| 重要性比率 | 每个 token 一个 $w_{i,t}$ | 每条响应一个 $s_i$ |
| 裁剪粒度 | token 级 | 序列级 |
| 一个响应是否会部分被裁剪 | 会 | 不会，整条响应共同裁剪 |
| 长度处理 | token loss 再求平均 | 对 log-ratio 求长度平均 |

最值得记住的结构是：

$$
\boxed{
\text{GSPO}
=
\text{GRPO 的组内相对优势}
+
\text{序列级重要性比率}
+
\text{序列级裁剪}
}
$$

---

## 2. 符号约定

| 符号 | 含义 |
|---|---|
| $x$ | prompt / query |
| $y_i=(y_{i,1},\ldots,y_{i,T_i})$ | 第 $i$ 条完整响应 |
| $T_i=\lvert y_i\rvert$ | 第 $i$ 条响应的有效 completion token 数 |
| $G$ | 同一个 prompt 采样的响应数，即 group size |
| $\pi_{\theta_{\mathrm{old}}}$ | 生成 rollout 数据的旧策略 |
| $\pi_\theta$ | 当前要更新的策略 |
| $r(x,y_i)$ | 对完整响应的奖励 |
| $\widehat A_i$ | 第 $i$ 条响应的组内相对优势 |
| $w_{i,t}$ | GRPO 的 token 级重要性比率 |
| $s_i$ | GSPO 的长度归一化序列级重要性比率 |
| $\epsilon$ | 对称裁剪半径 |
| $\epsilon_{\text{low}},\epsilon_{\text{high}}$ | 工程上常用的非对称左右裁剪半径 |

这里的 $T_i$ 只统计 **completion 的有效 token**。prompt token、padding token
和被丢弃的无效位置都不能进入长度平均。

---

## 3. 从 GRPO 出发

### 3.1 Outcome Supervision 的组内相对优势

对同一个问题的 $G$ 条回答得到奖励 $R_i=r(x,y_i)$，定义：

$$
\mu_R=\frac{1}{G}\sum_{j=1}^{G}R_j,
$$

$$
\sigma_R=
\sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j-\mu_R)^2},
$$

$$
\widehat A_i=
\frac{R_i-\mu_R}{\sigma_R+\varepsilon_{\mathrm{num}}}.
$$

它不需要单独训练 value/critic 模型：

- 奖励高于组内均值，$\widehat A_i>0$，应该提高该回答的概率；
- 奖励低于组内均值，$\widehat A_i<0$，应该降低该回答的概率；
- 组内奖励完全相同，所有 $\widehat A_i\approx 0$，这一组几乎没有策略梯度。

> 注意：不同框架对标准差使用总体估计（分母 $G$）还是样本估计（分母
> $G-1$）并不完全一致。两者只会改变缩放，不改变优势的正负和排序；复现实验时
> 必须与目标代码保持一致。

### 3.2 GRPO 的 token 级重要性比率

对第 $i$ 条响应的第 $t$ 个 token：

$$
w_{i,t}(\theta)=
\frac{
\pi_\theta(y_{i,t}\mid x,y_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})
}.
$$

在本文讨论的 Outcome Supervision GRPO 中，DeepSeekMath 原论文将一条响应中
所有 token 的优势都设为该响应的组内归一化奖励。因此，在数学上：

$$
\widehat A_{i,t}
=
\widehat A_i
=
\frac{R_i-\mu_R}{\sigma_R+\varepsilon_{\mathrm{num}}},
\qquad
t=1,\ldots,T_i.
$$

这不是说训练前必须在内存中真的复制 $T_i$ 份数据。实现时通常保存一个形状为
`[B]` 的响应优势，再扩展为 `[B, 1]`，由 PyTorch 在计算 token loss 时自动广播：

```python
# ratios: [B, T]，每个有效 token 一个重要性比率
# advantages: [B]，每条响应一个 outcome-level 优势
token_objective = ratios * advantages.unsqueeze(1)
```

`advantages.unsqueeze(1)` 的形状是 `[B, 1]`。它与 `[B, T]` 的 `ratios`
运算时表现得像每个 token 都取得同一个 $\widehat A_i$，但通常不需要物理复制
$T_i$ 份存储。

“优势相同”也不表示所有 token 产生完全相同的参数梯度，因为每个位置的：

$$
\nabla_\theta
\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
$$

仍取决于不同的 token 和前缀。

然后逐 token 计算 PPO 风格的裁剪目标：

$$
\begin{aligned}
\mathcal J_{\mathrm{GRPO}}(\theta)
=
\mathbb E\Bigg[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\min\Big(
&w_{i,t}(\theta)\widehat A_i,\\
&\operatorname{clip}(w_{i,t}(\theta),1-\epsilon,1+\epsilon)
\widehat A_i
\Big)
\Bigg].
\end{aligned}
$$

### 3.3 这里为什么会出现粒度错位

一个典型的 RLVR 任务只给完整答案一个奖励。例如数学题最终答案正确得 1 分，
错误得 0 分。也就是说：

$$
\underbrace{\text{reward}}_{\text{序列级}}
\quad\longleftrightarrow\quad
\underbrace{w_{i,t}}_{\text{token 级}}
$$

GRPO 对每个前缀下实际采到的那个 token 单独计算比率，却把同一个序列奖励广播给
所有 token。GSPO 论文认为，在大 rollout batch 被拆成多个 mini-batch、同一批数据
被更新多次的离策略场景中，这种 token 级校正会产生高方差噪声；响应越长，噪声
越容易累积，裁剪还可能进一步放大训练的不连续性。

具体过程是：

1. rollout 数据由 $\pi_{\theta_{\mathrm{old}}}$ 生成，并保存旧策略 log-prob；
2. 第一个 mini-batch 更新后，当前策略 $\pi_\theta$ 已经变化；
3. 后续 mini-batch 仍来自同一个旧策略，因而相对新的 $\pi_\theta$ 更加 off-policy；
4. 如果一批 rollout 还被重复训练多个 epoch，新旧策略差距会继续增大。

忽略裁剪时，GSPO 论文将 Outcome Supervision GRPO 的梯度写成：

$$
\nabla_\theta\mathcal J_{\mathrm{GRPO}}
=
\mathbb E\left[
\frac{1}{G}\sum_{i=1}^{G}
\widehat A_i
\frac{1}{T_i}\sum_{t=1}^{T_i}
w_{i,t}(\theta)
\nabla_\theta
\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
\right].
$$

虽然一条响应的 $\widehat A_i$ 相同，但每个 token 仍有各自的
$w_{i,t}$。例如：

$$
w_i=[1.01,\ 0.98,\ 1.40,\ 1.03,\ 0.95].
$$

这意味着同一响应中的 token 梯度被不等权地缩放。响应越长，出现极端
$w_{i,t}$ 或靠近裁剪边界的 token 的机会越多。这里不能简单套用“独立噪声求平均
后方差下降”的结论，因为 token 共享模型参数、依赖自回归前缀，而且裁剪是非线性
操作。

裁剪还给每个 token 增加了一个随 ratio 变化的梯度开关。以正优势为例：

$$
\frac{\partial}{\partial w}
\min\left(w\widehat A,\,
\operatorname{clip}(w,1-\epsilon,1+\epsilon)\widehat A\right)
=
\begin{cases}
\widehat A, & w<1+\epsilon,\\
0, & w>1+\epsilon,
\end{cases}
\qquad \widehat A>0.
$$

当一个 token 的 ratio 从 $1.19$ 变成 $1.21$，且 $1+\epsilon=1.2$ 时，它会从
“有梯度”突然变为“被裁剪”。目标值连续，但梯度在边界处切换。长响应包含大量这样
的 token 级开关，一次小更新就可能改变同一响应中参与梯度估计的 token 子集。

更严谨地说：

- **token 级概率比** 本身不是“数学上永远错误”；
- PPO 在逐状态、逐动作 advantage 可靠时本来就使用 token/action 级比率；
- GSPO 论文针对的是“**序列级 reward/advantage 被广播到 token，同时用每个 token
  的新旧概率比做离策略校正**”这套 GRPO 组合，并报告它在长序列与 MoE 训练中会
  不稳定。

### 3.4 负优势时，为什么有效权重没有上界

令：

$$
l=1-\epsilon,
\qquad
u=1+\epsilon.
$$

GRPO 对单个 token 最大化：

$$
L^{\mathrm{clip}}(w,\widehat A)
=
\min\left(
w\widehat A,\,
\operatorname{clip}(w,l,u)\widehat A
\right).
$$

当 $\widehat A<0$ 时，乘以负数会反转大小关系。分段计算得到：

$$
L^{\mathrm{clip}}(w,\widehat A)
=
\begin{cases}
l\widehat A, & 0<w<l,\\
w\widehat A, & w\ge l,
\end{cases}
\qquad \widehat A<0.
$$

等价地：

$$
\boxed{
L^{\mathrm{clip}}(w,\widehat A)
=
\widehat A\max(w,1-\epsilon),
\qquad \widehat A<0
}
$$

所以要严格区分：

- `clip` 函数本身的输出始终在 $[1-\epsilon,1+\epsilon]$；
- 但 `min` 在 $\widehat A<0$ 且 $w>1+\epsilon$ 时，会选择更加负的未裁剪项
  $w\widehat A$；
- 因此最终 surrogate objective 中的**有效梯度权重**是
  $\max(w,1-\epsilon)$，范围为 $[1-\epsilon,+\infty)$。

也就是说，正无穷确实不是 `clip` 的输出，而是“原始 ratio 可无界、负优势反转
大小关系、`min` 重新选择未裁剪分支”共同造成的。

#### 一个 ratio 等于 1000 的例子

GRPO 的 token 级重要性比率定义为：

$$
w_{i,t}
=
\frac{
\pi_\theta(y_{i,t}\mid x,y_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})
}.
$$

它比较的是：对于同一个已经由旧策略采样出来的 token，当前策略给出的概率是旧策略
概率的多少倍。

假设旧策略对这个 token 的概率非常接近 0：

$$
\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid\cdot)
=
10^{-6}
=
0.000001.
$$

经过一个或多个 mini-batch 更新后，当前策略对同一个 token 给出的概率变成：

$$
\pi_\theta(y_{i,t}\mid\cdot)
=
10^{-3}
=
0.001.
$$

从绝对值看，$0.001$ 仍然是一个很小的概率，两者只相差：

$$
0.001-0.000001=0.000999.
$$

但是 importance ratio 衡量的不是绝对差，而是**相对倍数**。利用指数除法：

$$
\frac{10^a}{10^b}=10^{a-b},
$$

所以：

$$
\begin{aligned}
w_{i,t}
&=
\frac{0.001}{0.000001}\\
&=
\frac{10^{-3}}{10^{-6}}\\
&=
10^{-3-(-6)}\\
&=
10^3\\
&=
1000.
\end{aligned}
$$

也就是说，当前概率虽然在绝对数值上仍然很小，却已经是旧概率的 1000 倍：

$$
\pi_\theta(y_{i,t}\mid\cdot)
=
1000\,
\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid\cdot).
$$

这正是“分母非常接近 0 时，概率只增加一个很小的绝对量，ratio 却可能非常大”的
含义。更准确地说，$0.001$ 相比 $0.000001$ 只是**绝对增加量很小**，但它的
**相对增幅并不小**，而是增长到了原来的 1000 倍。

在大词表语言模型中，单个已采样 token 的旧策略概率可能很低；同一批 rollout
数据经过多个 mini-batch 更新后，当前策略概率又可能发生变化，因此这种大 ratio
在长序列和大批量训练中不能简单忽略。

> 边界条件：分母必须满足
> $\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid\cdot)>0$。如果旧策略概率严格等于 0，
> 这个 token 不可能从该旧策略分布中被采样出来，ratio 也没有定义。标准 softmax
> 在精确数学意义下会给每个词表 token 正概率，但有限精度、截断采样和实现细节仍需
> 单独检查。

现在再设：

$$
\widehat A_i=-1,
\qquad
\epsilon=0.2.
$$

此时未裁剪项为：

$$
w_{i,t}\widehat A_i=-1000,
$$

裁剪项为：

$$
\operatorname{clip}(1000,0.8,1.2)\widehat A_i
=
1.2\times(-1)
=-1.2.
$$

最终：

$$
\min(-1000,-1.2)=-1000.
$$

尽管 `clip` 已把 ratio 从 $1000$ 变成 $1.2$，`min` 仍选择未裁剪的 $-1000$。
对应梯度中的缩放系数也包含这个大 ratio：

$$
\nabla_\theta(w\widehat A)
=
w\widehat A
\nabla_\theta\log\pi_\theta(y_{i,t}\mid\cdot).
$$

PPO 这样设计的直觉是：$\widehat A<0$ 时，减小坏动作概率是有利方向，低于
$1-\epsilon$ 后停止继续奖励；反过来，若 $w$ 很大，说明当前策略反而大幅提高了
坏动作的概率，这是有害方向，所以保留惩罚以提供纠错梯度。

但在 Outcome Supervision GRPO 中，这个 token 的负优势来自整条响应，且 token
ratio 只基于该前缀下实际采到的一个 token。GSPO 论文认为，无上界的单 token
权重会成为高方差来源，并可能在长响应中累积。

--- 

## 4. 从 token 概率推导序列级比率

### 4.1 自回归模型的序列似然

语言模型对完整响应 $y_i$ 的条件概率是每个 token 条件概率的乘积：

$$
\pi_\theta(y_i\mid x)
=
\prod_{t=1}^{T_i}
\pi_\theta(y_{i,t}\mid x,y_{i,<t}),
$$

$$
\pi_{\theta_{\mathrm{old}}}(y_i\mid x)
=
\prod_{t=1}^{T_i}
\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t}).
$$

所以原始序列概率比为：

$$
\frac{\pi_\theta(y_i\mid x)}
{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)}
=
\prod_{t=1}^{T_i}
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})}
=
\prod_{t=1}^{T_i}w_{i,t}.
$$

### 4.2 为什么不能直接使用上面的连乘

假设每个 token 的新策略概率都只比旧策略高 $1\%$，即 $w_{i,t}=1.01$：

$$
\prod_{t=1}^{10}w_{i,t}=1.01^{10}\approx1.105,
$$

$$
\prod_{t=1}^{100}w_{i,t}=1.01^{100}\approx2.705.
$$

逐 token 变化明明相同，长响应的联合比率却会因为连乘远离 1。这样不同长度的
回答需要不同裁剪范围，而且极易出现数值下溢或上溢。

### 4.3 GSPO 的长度归一化

GSPO 对原始序列比率开 $T_i$ 次方根：

$$
s_i(\theta)
=
\left(
\frac{\pi_\theta(y_i\mid x)}
{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)}
\right)^{1/T_i}.
$$

代入自回归分解：

$$
s_i(\theta)
=
\left(\prod_{t=1}^{T_i}w_{i,t}\right)^{1/T_i}.
$$

因此，$s_i$ 就是这条响应所有 token 概率比的**几何平均**。为避免直接连乘，
实际计算使用 log 空间：

$$
\boxed{
s_i(\theta)
=
\exp\left[
\frac{1}{T_i}
\sum_{t=1}^{T_i}
\left(
\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
-
\log\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})
\right)
\right]
}
$$

如果每个 token 都有 $w_{i,t}=1.01$，无论 $T_i=10$ 还是 $100$：

$$
s_i=(1.01^{T_i})^{1/T_i}=1.01.
$$

> **概念校正：**未经归一化的
> $\pi_\theta(y_i\mid x)/\pi_{\mathrm{old}}(y_i\mid x)$ 是标准的原始序列
> importance ratio；GSPO 实际用于优化的 $s_i$ 是它的长度归一化版本。它保留
> 序列似然变化的方向，并统一不同长度的数值尺度，但不应把开 $T_i$ 次方根后的
> 量再机械解释为完全不变形的经典重要性采样权重。

---

## 5. GSPO 目标函数

GSPO 仍然使用组内相对优势：

$$
\widehat A_i=
\frac{
r(x,y_i)-\operatorname{mean}(\{r(x,y_j)\}_{j=1}^{G})
}{
\operatorname{std}(\{r(x,y_j)\}_{j=1}^{G})
}.
$$

对称裁剪形式为：

$$
\boxed{
\mathcal J_{\mathrm{GSPO}}(\theta)
=
\mathbb E\left[
\frac{1}{G}\sum_{i=1}^{G}
\min\left(
s_i(\theta)\widehat A_i,\,
\operatorname{clip}(s_i(\theta),1-\epsilon,1+\epsilon)\widehat A_i
\right)
\right]
}
$$

训练代码一般最小化它的相反数：

$$
\mathcal L_{\mathrm{GSPO}}=-\mathcal J_{\mathrm{GSPO}}.
$$

工程上也可使用非对称范围：

$$
\operatorname{clip}
\left(
s_i,\,
1-\epsilon_{\mathrm{low}},\,
1+\epsilon_{\mathrm{high}}
\right).
$$

### 5.1 裁剪到底在限制什么

对一条好回答 $\widehat A_i>0$：

- $s_i\le 1+\epsilon_{\mathrm{high}}$：继续鼓励提高整条响应的似然；
- $s_i>1+\epsilon_{\mathrm{high}}$：正向收益被封顶，不再鼓励走得更远。

对一条差回答 $\widehat A_i<0$：

- $s_i\ge 1-\epsilon_{\mathrm{low}}$：继续降低整条响应的似然；
- $s_i<1-\epsilon_{\mathrm{low}}$：负向更新被封顶，不再鼓励降得更低。

需要注意，GSPO 沿用了同一个 PPO `min` 结构。因此当 $\widehat A_i<0$ 时，
序列 ratio 的上侧同样不会被截断：

$$
L_{\mathrm{GSPO},i}^{\mathrm{clip}}
=
\widehat A_i
\max\left(s_i,1-\epsilon_{\mathrm{low}}\right),
\qquad \widehat A_i<0.
$$

GSPO 的稳定性并不是来自把 ratio 变成双侧有界，而是来自两个改变：

1. 先把 token log-ratio 按有效长度平均，再得到一个序列 ratio，孤立 token 的极端
   波动不会原样成为独立梯度权重；
2. 每条响应只进行一次裁剪判断，不再有大量相互独立切换的 token 级裁剪开关。

关键是：**GSPO 对一条响应只做一次判断。**一条回答不会像 GRPO 那样出现“部分
token 被裁剪，部分 token 仍有梯度”的情况。

### 5.2 GSPO 的梯度为何仍能更新每个 token

“序列级目标”不等于“模型只产生一个梯度”。在未触发裁剪时：

$$
\mathcal J_i=s_i\widehat A_i.
$$

由于

$$
\log s_i
=
\frac{1}{T_i}\sum_{t=1}^{T_i}
\left(\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
-\log\pi_{\mathrm{old}}(\cdot)\right),
$$

且旧策略是常量，所以：

$$
\nabla_\theta s_i
=
s_i\frac{1}{T_i}
\sum_{t=1}^{T_i}
\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}).
$$

最终：

$$
\boxed{
\nabla_\theta\mathcal J_i
=
\frac{s_i\widehat A_i}{T_i}
\sum_{t=1}^{T_i}
\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
}
$$

因此所有有效 token 都参与反向传播，只是它们共享同一个序列级缩放系数
$s_i\widehat A_i/T_i$。各 token 的
$\nabla_\theta\log\pi_\theta(y_{i,t}\mid\cdot)$ 仍不同，所以不能理解成“句子中
每个 token 的参数梯度完全相同”。

---

## 6. 一个完整的数值例子

### 6.1 第一步：计算组内优势

假设同一道数学题生成 4 条回答，reward 为：

$$
R=[1,\ 0,\ 0,\ 1].
$$

使用总体标准差：

$$
\mu_R=0.5,\qquad \sigma_R=0.5,
$$

$$
\widehat A=[1,\ -1,\ -1,\ 1].
$$

第一、第四条回答会被鼓励，第二、第三条会被抑制。

### 6.2 第二步：从 token 比率得到序列比率

为了只观察“裁剪粒度”的区别，先让 GRPO 与 GSPO 在这个教学例子里都使用
$[0.8,1.2]$。真实训练不能据此把两者超参数设成一样。

一条好回答有 4 个 token：

$$
w=[1.30,\ 1.30,\ 0.90,\ 0.90],\qquad \widehat A=1.
$$

GSPO 的序列比率：

$$
\begin{aligned}
s
&=(1.30\times1.30\times0.90\times0.90)^{1/4}\\
&\approx1.081665.
\end{aligned}
$$

它在 $[0.8,1.2]$ 内，所以整条响应不被裁剪：

$$
\mathcal J_{\mathrm{GSPO},i}\approx1.081665.
$$

GRPO 则分别裁剪前两个 token：

$$
\mathcal J_{\mathrm{GRPO},i}
=
\frac{1.2+1.2+0.9+0.9}{4}
=1.05.
$$

这个例子展示的不是“1.081665 一定优于 1.05”，而是两种目标看到的数据结构不同：

- GRPO 认为前两个 token 已走得太远；
- GSPO 认为整条序列的平均似然变化仍在信赖域内。

### 6.3 反过来看一条差回答

设：

$$
w=[0.75,\ 0.75,\ 1.05,\ 1.05],\qquad \widehat A=-1.
$$

序列比率：

$$
s=(0.75^2\times1.05^2)^{1/4}\approx0.887412.
$$

GSPO 认为整条序列仍在区间内：

$$
\mathcal J_{\mathrm{GSPO},i}\approx-0.887412.
$$

GRPO 对两个 $0.75$ 触发负优势方向的下界裁剪：

$$
\mathcal J_{\mathrm{GRPO},i}
=
\frac{-0.8-0.8-1.05-1.05}{4}
=-0.925.
$$

再次看到：GRPO 会在一条响应内部混合“被裁剪 token”和“未裁剪 token”，GSPO
则以整条响应为一个裁剪单元。

运行 [gspo_demo.py](./gspo_demo.py) 可以复现这些数字：

```bash
python gspo_note/gspo_demo.py
```

---

## 7. 代码实现：从模型 log-prob 到 loss

核心实现位于 [gspo_loss.py](./gspo_loss.py)。训练阶段通常已经有：

- `new_log_probs[B, T]`：当前策略对已采样 completion token 的 log-prob；
- `old_log_probs[B, T]`：rollout 时保存的旧策略 log-prob；
- `completion_mask[B, T]`：有效 completion token 为 1；
- `advantages[B]`：每条响应的组内相对优势。

最关键的代码只有三步。

### 7.1 对 token log-ratio 做 masked mean

```python
token_log_ratios = new_log_probs - old_log_probs.detach()

sequence_log_ratios = (
    (token_log_ratios * completion_mask).sum(dim=1)
    / completion_mask.sum(dim=1)
)
```

这一步对应：

$$
\log s_i=
\frac{1}{T_i}\sum_t
\left(\log\pi_\theta-\log\pi_{\mathrm{old}}\right).
$$

### 7.2 回到序列比率

```python
sequence_ratios = torch.exp(sequence_log_ratios)
```

不要先对每个 token `exp`，再做算术平均：

```python
# 错误：这是 token ratio 的算术平均，不是 GSPO 的几何平均
wrong = torch.exp(token_log_ratios).mean(dim=1)
```

正确的 $s_i$ 是：

$$
\exp(\operatorname{mean}(\log w_{i,t})),
$$

而不是：

$$
\operatorname{mean}(w_{i,t}).
$$

### 7.3 序列级裁剪

```python
unclipped = sequence_ratios * advantages
clipped_ratios = torch.clamp(
    sequence_ratios,
    min=1.0 - eps_low,
    max=1.0 + eps_high,
)
clipped = clipped_ratios * advantages

per_sequence_objective = torch.minimum(unclipped, clipped)
loss = -per_sequence_objective.mean()
```

`torch.minimum` 不能错误地换成 `torch.maximum`。论文给的是要**最大化**的
`min` 目标，而训练器返回的是这个目标的负数。

### 7.4 组内优势代码

reward 应按 prompt 组织为 `[num_prompts, group_size]`：

```python
means = rewards.mean(dim=1, keepdim=True)
stds = rewards.std(dim=1, keepdim=True, unbiased=False)
advantages = (rewards - means) / (stds + 1e-8)
advantages = advantages.reshape(-1)
```

不能把整个全局 batch 的不同 prompt 混在一起标准化；GSPO/GRPO 比较的是同一道
题的一组回答。

---

## 8. 为什么 GSPO 对 MoE 更友好

MoE 模型每个 token 只激活少数专家。参数更新后，同一个 token 在新旧策略中可能
走到不同专家：

$$
\text{old policy: experts }(1,2)
\qquad\longrightarrow\qquad
\text{new policy: experts }(1,3).
$$

对 GRPO 来说，每个 token 的

$$
w_{i,t}=
\frac{\pi_\theta(y_{i,t}\mid\cdot)}
{\pi_{\mathrm{old}}(y_{i,t}\mid\cdot)}
$$

可能因路由变化产生尖锐波动。Routing Replay 的做法是缓存 rollout 时激活的专家，
训练时强制重放同样的路由，以便比较更一致的新旧 token 概率；代价是额外的存储、
通信和工程复杂度，还限制当前策略自由选择专家。

GSPO 把所有 token 的 log-ratio 聚合成一个长度归一化序列统计量，对少量 token 的
局部尖峰不那么敏感。GSPO 论文在 Qwen3-30B-A3B-Base 的实验中报告：GSPO 不依赖
Routing Replay 也能稳定收敛。

这里应保留实验边界：

- “不需要 Routing Replay”是该论文在其 MoE 架构与训练设置中的结果；
- 它不等于任何 MoE、任何精度或任何分布式系统都不再需要一致性检查；
- 若 rollout 与训练引擎的 tokenization、mask、权重版本或采样配置不一致，序列级
  聚合也无法修复这些语义错误。
---

## 9. 裁剪阈值：最容易踩的坑

GSPO 与 GRPO 的 ratio 定义不同，裁剪范围不能直接复用。

GSPO 论文的公式为便于表达使用对称 $\epsilon$，但论文实验使用：

$$
\epsilon_{\mathrm{low}}^{\mathrm{GSPO}}=3\times10^{-4},
\qquad
\epsilon_{\mathrm{high}}^{\mathrm{GSPO}}=4\times10^{-4}.
$$

对照 GRPO 基线则使用：

$$
\epsilon_{\mathrm{low}}^{\mathrm{GRPO}}=0.2,
\qquad
\epsilon_{\mathrm{high}}^{\mathrm{GRPO}}=0.27.
$$

差异大的原因不是 GSPO “天生必须是这两个固定值”，而是：

- GRPO 裁剪单个 token 的比率；
- GSPO 裁剪整条序列上 log-ratio 的平均所对应的几何比率；
- 两者统计分布不同，超参数的数量级自然不同。

正确做法是记录并监控：

```text
sequence_ratio_mean
sequence_ratio_min / max
sequence_clip_fraction
reward_mean / std
response_length
```

再根据具体模型、mini-batch 更新次数、学习率和 rollout 新鲜度调参。

---

## 10. 训练流程伪代码

```text
for prompts in dataloader:
    # 1. 用旧策略为每个 prompt 采样 G 条完整响应
    responses, old_log_probs = rollout(old_policy, prompts, group_size=G)

    # 2. 对完整响应打分
    rewards = verifier(prompts, responses)       # [num_prompts, G]

    # 3. 每个 prompt 内部计算相对优势
    advantages = group_normalize(rewards)        # [num_prompts, G]

    # 4. 可把大 rollout batch 拆成 mini-batch，更新当前策略
    for mini_batch in split(responses):
        new_log_probs = policy.log_probs(mini_batch)

        # 5. 只对有效 completion token 求平均 log-ratio
        log_s = masked_mean(
            new_log_probs - old_log_probs,
            completion_mask,
        )
        s = exp(log_s)

        # 6. 一条响应裁剪一次
        objective = min(
            s * advantage,
            clip(s, 1-eps_low, 1+eps_high) * advantage,
        )

        loss = -mean(objective)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    old_policy = snapshot(policy)
```

实际大规模训练还需要处理分布式聚合、熵、可能的 KL 正则、梯度裁剪、过长响应、
动态采样和数据新鲜度等问题；这些不是 GSPO 核心目标本身，不能与序列级 ratio
混为一谈。

---

## 11. 实现检查清单

### 数据与 mask

- 每个 prompt 确实对应 $G$ 条回答；
- group normalize 只在同一 prompt 内进行；
- prompt token 不参与 completion ratio；
- padding token 不参与求和，也不进入长度 $T_i$；
- EOS 是否计入要在 rollout 和训练端保持一致；
- 被截断响应的 reward 规则要明确。

### 新旧策略

- `old_log_probs` 来自真正生成该响应的策略版本；
- `old_log_probs` 必须 `detach`；
- 多个 mini-batch 更新期间不能悄悄覆盖旧 log-prob；
- rollout 引擎与训练引擎必须使用相同 tokenizer 和 token IDs。

### 数值计算

- 在 log 空间先求平均，再 `exp`；
- 不要直接连乘 token probability；
- 不要把 token ratio 的算术平均误写成几何平均；
- 组内标准差为 0 时要加数值稳定项；
- 记录 clip fraction，防止几乎全裁剪或完全不裁剪。

### 超参数

- 不直接复制 GRPO 的 clip epsilon；
- 左右裁剪范围可以不对称；
- clip、学习率、每个 rollout 的更新次数要一起看；
- 论文超参数只是其模型与系统上的已验证起点。

---

## 12. GSPO 的收益与代价

### 主要收益

1. **优化粒度与 reward 对齐**：完整响应奖励对应完整响应 ratio。
2. **降低局部 token 比率噪声的影响**：token 波动先在 log 空间按长度平均。
3. **裁剪语义清楚**：整条回答进入或离开信赖区域。
4. **对 MoE 更稳定**：原论文实验中不再依赖 Routing Replay。
5. **可能简化 RL 基础设施**：论文认为序列统计量对训练/推理引擎精度差异更宽容。

### 需要接受的代价

1. **更粗的 credit assignment**：所有 token 共享同一序列优势和序列缩放系数。
2. **局部关键信息被平均**：一个特别好或特别差的推理步骤可能被整条序列稀释。
3. **长度归一化改变了原始 IS 权重**：它是稳定训练使用的 surrogate 设计。
4. **仍依赖组内奖励差异**：组内 reward 全相同时没有学习信号。
5. **不是完整训练配方**：采样、reward 设计、长度处理和数据质量仍决定最终效果。

---

## 13. 常见问题

### Q1：GSPO 还属于 GRPO 家族吗？

可以把它看成保留 GRPO 组内优势估计、替换策略优化粒度的新方法。它们都不依赖
value model，但 GSPO 不是简单把 GRPO 的 token loss 最后再平均一下。

### Q2：为什么是几何平均，不是算术平均？

序列概率是 token 条件概率的乘积。乘积开 $T$ 次方根自然得到几何平均；在
log 空间中对应 log-ratio 的算术平均，既符合自回归分解，也更稳定。

### Q3：序列级裁剪后，回答里的 token 还有梯度吗？

有。只要序列目标未进入裁剪后的平坦区域，$s_i$ 对每个有效 token 的 log-prob
都有梯度。区别是所有 token 共享同一个序列级权重。

### Q4：GSPO 是否完全解决长序列的 credit assignment？

没有。它解决的是重要性比率和 reward/优化粒度的对齐及稳定性问题，却保留了
“整条响应共享一个 advantage”的粗粒度信用分配。某一步推理到底贡献多少，GSPO
本身并不知道。

### Q5：是否可以加入 KL 正则？

可以把 KL 作为额外正则项加入总 loss，但 GSPO 核心公式为突出序列级目标而省略
了 KL。是否加入、使用哪个 KL 估计器和如何缩放，应视训练配方决定。

### Q6：能否直接使用推理引擎返回的 old log-prob？

GSPO 论文认为序列级聚合更能容忍训练与推理引擎的精度差异，因此有机会避免训练
引擎重算。但“更宽容”不等于“任意不一致都安全”；正式系统仍应先比较两端
sequence log-ratio 的偏差分布。

---

## 14. 最后总结

从 GRPO 到 GSPO 的逻辑链可以压缩为：

$$
\text{完整响应得到 reward}
\Rightarrow
\text{完整响应得到 advantage}
\Rightarrow
\text{完整响应计算 ratio}
\Rightarrow
\text{完整响应共同 clip}.
$$

其中最核心的实现公式是：

$$
\boxed{
s_i
=
\exp\left(
\operatorname{masked\_mean}_t
\left[
\log\pi_\theta(y_{i,t}\mid\cdot)
-
\log\pi_{\mathrm{old}}(y_{i,t}\mid\cdot)
\right]
\right)
}
$$

只要牢牢记住“**先在 log 空间按有效长度平均，再 exp，最后按序列裁剪一次**”，
就抓住了 GSPO 与 GRPO 的本质区别。

---

## 参考资料

1. Zhihong Shao et al.,
   [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300),
   arXiv:2402.03300, 2024。原论文第 4.1.2 节定义了本文采用的
   Outcome Supervision GRPO。
2. Chujie Zheng et al.,
   [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071),
   arXiv:2507.18071, 2025。
3. Qwen Team,
   [GSPO: Towards Scalable Reinforcement Learning for Language Models](https://qwenlm.github.io/blog/gspo/),
   2025。

原论文的实验设置值得特别核对：GSPO 使用序列级 ratio 与裁剪；实验中的 GSPO
左右裁剪范围为 $3\times10^{-4}$ 和 $4\times10^{-4}$，GRPO 基线则为
$0.2$ 和 $0.27$。这些数字用于理解数量级差异，不应未经调试直接当作所有任务
的默认最优值。
