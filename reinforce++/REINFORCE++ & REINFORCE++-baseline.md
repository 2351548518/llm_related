# REINFORCE\+\+ \& REINFORCE\+\+\-baseline

## REINFORCE\+\+

REINFORCE算法形式如下：

$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log \pi_{\theta}(\tau) R(\tau)]$

REINFORCE\+\+相比于REINFORCE算法主要做了如下改进：

1、Token Level KL\-Penalty

使用token粒度的kl散度约束策略模型不要偏离参考模型太远，kl散度会融入到奖励的计算中

2、Mini\-batch Updates

两个好处：

- 加快收敛，减少内存占用

- 引入随机性，有助于避免局部最优，增加泛化性

3、Reward Normalization and Clipping

对奖励进行归一化和clip，平滑奖励信号，提高训练稳定性，并且减少极端奖励的影响

4、Advantage Normalization

提高训练稳定性

5、loss计算

使用ppo loss，引入重要性采样和clip（已经算是标配了）

$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$

## REINFORCE\+\+\-baseline

REINFORCE\+\+\-baseline是REINFORCE\+\+的一个变体，借鉴了GRPO中的思想，使用同一prompt多次采样的奖励平均值作为baseline，和GRPO不同的是，REINFORCE\+\+\-baseline去掉了除以标准差的操作，同时会对优势进行标准化操作（在一个batch内进行标准化）。

$\begin{aligned}
A_{q, o_t} &= R_{q, o_t} - \text{mean}_{\text{group}}(R_{q, o_t}) \\
A_{q, o_t}^{\text{norm}} &= \frac{A_{q, o_t} - \text{mean}_{\text{batch}}(A_{q, o_t})}{\text{std}_{\text{batch}}(A_{q, o_t})}
\end{aligned}$

在kl散度的计算上，REINFORCE\+\+\-baseline使用k2，GRPO中使用的是k3

$\begin{aligned}
\mathcal{L}_{k_2} &= \mathbb{E}_{s \sim D, a \sim \pi_{\theta_{\text{old}}}}(\cdot | s) \left( \frac{1}{2} (-\log x)^2 \right) \\
\mathcal{L}_{k_3} &= \mathbb{E}_{s \sim D, a \sim \pi_{\theta_{\text{old}}}}(\cdot | s) \left( (x - 1) - \log x \right)
\end{aligned}$


## 代码阅读笔记

### 1. 项目定位

`reinforce++` 目录是一个学习性质的单文件 REINFORCE++ 训练原型，基于 TRL 旧版 `RLOOTrainer` 改造，目标是使用 Qwen2.5-7B-Instruct、LoRA 和 GSM8K 规则奖励进行强化学习训练。

目录中的文件分工如下：

- `train_reinforce++.py`：奖励函数、rollout、KL reward、优势计算、PPO 更新、LoRA 和训练配置。
- `data_process.ipynb`：把中文 GSM8K 转换为聊天格式的 `prompt` 和标准答案 `answer`。
- `problem.md`：记录重要性采样以及 k1、k2、k3 KL 估计器问题。
- 本文档：算法概念和代码阅读笔记。

当前默认配置运行的是普通 REINFORCE++，不是 baseline 版本：

```python
token_level_kl = True
normalize_advantage = True
normalize_reward = False
use_baseline = False
rloo_k = 1
```

### 2. 完整训练流程

一次外层训练更新可以概括为：

```text
prompt batch
    ↓
policy 采样回答
    ↓
保存 rollout/old policy 的 token log-prob
    ↓
计算固定 reference policy 的 token log-prob
    ↓
规则奖励：答案正确 +1，错误 -1
    ↓
加入 reference KL penalty
    ↓
可选：同一 prompt 的组均值 baseline
    ↓
batch 内优势标准化
    ↓
多轮 PPO clipped update
```

主要代码位置：

- `extract_answer()` 和 `correctness_reward()`：第 66～74 行。
- rollout 生成：第 154～187 行。
- rollout policy/reference policy log-prob：第 189～248 行。
- KL 和奖励：第 276～310 行。
- baseline 和优势标准化：第 314～331 行。
- PPO 更新：第 335～400 行。
- 模型、LoRA 和训练参数：第 554～631 行。

代码中需要区分三个策略：

- **new/current policy**：当前正在训练的 $\pi_\theta$。
- **old/rollout policy**：生成这一批回答时的 $\pi_{\theta_{\text{old}}}$。
- **reference policy**：固定的初始/SFT 策略 $\pi_{ref}$。

PPO 重要性采样比较的是 current policy 和 old policy；reference KL 比较的是 policy 和 reference policy。这是两个不同的约束。

## 重要性采样

### 1. 公式中的 $s$、$a$ 和 $f(s,a)$

重要性采样的通用目标可以写成：

$$
J(\theta)
=
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[f(s,a)\right]
$$

其中：

- $s$ 是 state，即当前状态；
- $a$ 是 action，即在状态 $s$ 下采取的动作；
- $\pi_\theta(a\mid s)$ 是模型在状态 $s$ 下选择动作 $a$ 的概率；
- $f(s,a)$ 是一个通用函数，表示在状态 $s$ 下采取动作 $a$ 后，我们关心的某个数值，例如即时奖励、未来累计奖励或优势。

这里的 $f(s,a)$ 只是为了推导重要性采样而使用的通用记号，不是某个固定的强化学习变量。

#### LLM 中的状态 $s_t$

在 LLM 强化学习中，状态通常定义为：

$$
s_t=(x,y_{<t})
$$

其中：

- $x$ 是用户输入的 prompt；
- $y_{<t}=y_1,\ldots,y_{t-1}$ 是模型已经生成的 token；
- $s_t$ 是 prompt 加上当前已经生成的文本前缀。

例如 prompt 是：

```text
计算 2 + 3。
```

模型已经生成：

```text
2 + 3 =
```

那么当前状态 $s_t$ 就是模型此刻能看到的完整上下文：

```text
用户：计算 2 + 3。
助手：2 + 3 =
```

#### LLM 中的动作 $a_t$

动作是在当前状态下生成的下一个 token：

$$
a_t=y_t
$$

例如在上面的状态中，模型可以选择 token `5`，也可以选择 token `6`：

$$
\pi_\theta(\text{“5”}\mid s_t)=0.8
$$

$$
\pi_\theta(\text{“6”}\mid s_t)=0.1
$$

$\pi_\theta(\cdot\mid s)$ 中的 $\cdot$ 表示状态 $s$ 下所有可能动作的概率分布。

#### $f(s,a)$ 的三种常见含义

第一种是即时奖励：

$$
f(s,a)=r(s,a)
$$

例如对于一道选择题，可以定义：

$$
r(s,\text{“5”})=1,\qquad
r(s,\text{“6”})=-1
$$

如果模型的概率为：

$$
\pi_\theta(\text{“5”}\mid s)=0.8,\qquad
\pi_\theta(\text{“6”}\mid s)=0.2
$$

那么期望奖励为：

$$
J(\theta)
=0.8\times1+0.2\times(-1)
=0.6
$$

第二种是未来累计奖励，即 action value：

$$
f(s_t,a_t)=Q(s_t,a_t)
$$

LLM 每次只生成一个 token，但通常要等完整回答生成完以后才能判断答案是否正确。因此 $Q(s_t,a_t)$ 表示：在当前文本前缀下生成 token $a_t$，然后继续生成，最终预计能够获得多少奖励。

第三种是优势：

$$
f(s_t,a_t)=A(s_t,a_t)
$$

优势定义为：

$$
A(s,a)=Q(s,a)-V(s)
$$

其中：

- $Q(s,a)$ 表示选择动作 $a$ 后预计得到的未来奖励；
- $V(s)$ 表示状态 $s$ 下按照当前策略行动时的平均预期奖励；
- $A(s,a)$ 表示动作 $a$ 比当前状态下的平均选择好多少。

因此：

- $A(s,a)>0$：动作比平均选择好，应当提高其概率；
- $A(s,a)<0$：动作比平均选择差，应当降低其概率；
- $A(s,a)=0$：动作与平均水平接近。

PPO 的目标通常写成：

$$
L(\theta)
=
\mathbb{E}_{s_t,a_t\sim\pi_{\text{old}}}
\left[r_t(\theta)A_t\right]
$$

所以在 PPO 语境下，通用公式中的 $f(s,a)$ 具体对应优势 $A_t$。

#### 与当前代码的对应关系

| 数学符号 | 当前代码中的含义 |
| --- | --- |
| $s_t$ | prompt 加已经生成的 response 前缀 |
| $a_t$ | `mb_responses` 中第 $t$ 个 token |
| $\pi_{\text{old}}(a_t\mid s_t)$ | `mb_logprobs.exp()` |
| $\pi_\theta(a_t\mid s_t)$ | `new_logprobs.exp()` |
| $r_t(\theta)$ | `ratio` |
| $A_t$ | `mb_advantage` |
| 最终任务奖励 | `scores` |
| KL 修正后的奖励 | `rlhf_reward` |

对应的 PPO 代码为：

```python
mb_advantage = advantages[micro_batch_inds]
mb_responses = responses[micro_batch_inds]
mb_logprobs = logprobs[micro_batch_inds]

new_logprobs = selective_log_softmax(logits, mb_responses)

logprobs_diff = new_logprobs - mb_logprobs
ratio = torch.exp(logprobs_diff)

pg_losses = -mb_advantage * ratio
```

当前代码先得到完整回答的正确性奖励，再加入 reference KL penalty：

$$
R_{\text{RLHF}}
=
R_{\text{task}}
-
\beta\,\mathrm{KL}
$$

然后将 `rlhf_reward` 作为优势的基础。启用 baseline 时，还会先减去同一 prompt 多个回答的平均奖励；最后再对 batch 中的优势进行标准化。

因此，当前代码里的 `mb_advantage` 大致表示：这个完整回答的奖励相对于同组或同 batch 其他回答有多好。它是一个回答级标量优势，然后被用于回答中的 token，而不是为每个 token 单独估计不同的 $Q(s_t,a_t)$。

#### 完整回答与轨迹

LLM 的完整回答可以表示为一条轨迹：

$$
\tau=(s_1,a_1,s_2,a_2,\ldots,s_T,a_T)
$$

例如：

```text
prompt: 2 + 3 等于多少？
a_1: <think>
a_2: 2 + 3 = 5
...
a_T: </answer>
最终回答：<answer>5</answer>
最终奖励：R = 1
```

更完整的 LLM 强化学习目标是：

$$
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
[R(\tau)]
$$

也可以把 prompt 和完整回答分开写成：

$$
J(\theta)
=
\mathbb{E}_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
[R(x,y)]
$$

其中：

- $x$ 是 prompt；
- $y=(y_1,\ldots,y_T)$ 是完整回答；
- $s_t=(x,y_{<t})$ 是生成第 $t$ 个 token 时的状态；
- $a_t=y_t$ 是第 $t$ 个 token；
- $R(x,y)$ 是完整回答获得的最终奖励。

可以直观地记为：$s$ 是模型已经看到了什么，$a$ 是模型接下来生成什么，$f(s,a)$ 是这个选择带来的价值；在当前 PPO 代码中，$f(s,a)$ 具体对应优势 $A_t$。

### 2. 为什么需要重要性采样

当前 batch 的回答由更新前的策略 $\pi_{\theta_{\text{old}}}$ 生成，但在 PPO 内层进行一次参数更新后，正在优化的模型已经变成 $\pi_\theta$。

如果继续使用旧策略产生的样本估计当前策略的目标，需要乘以重要性权重：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
     {\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\theta_{\text{old}}}(a_t\mid s_t)
\right)
$$

代码实现为：

```python
logprobs_diff = new_logprobs - mb_logprobs
ratio = torch.exp(logprobs_diff)
```

这里的：

- `mb_logprobs` 是生成 rollout 时保存的 old policy log-prob；
- `new_logprobs` 是当前模型重新前向计算出的 log-prob；
- `ratio` 用于修正 old policy 和 current policy 之间的分布差异。

### 3. 为什么还需要 PPO clipping

如果某个 token 的概率比率过大或过小，少量旧样本就可能产生非常大的梯度。因此 PPO 将比率限制在 $[1-\epsilon,1+\epsilon]$：

$$
L^{CLIP}(\theta)
=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

重要性采样使同一批 rollout 可以被多轮更新复用；clipping 则限制每轮更新不要离 rollout policy 太远。

## k1、k2、k3 KL 估计器

### 1. 统一定义

假设样本来自分布 $q$，目标分布是 $p$，定义概率比：

$$
x=\frac{p(a)}{q(a)}
$$

三个常见逐样本 KL 估计量为：

| 估计器 | 公式 | 主要性质 |
| --- | --- | --- |
| k1 | $-\log x$ | KL 值的无偏估计；单个样本可能为负，方差较大 |
| k2 | $\frac12(\log x)^2$ | 非负、通常较稳定；作为 KL 数值估计一般是二阶近似 |
| k3 | $(x-1)-\log x$ | 非负；满足支持集等条件时，其期望等于 KL，但概率比可能爆炸 |

对于 reverse KL：

$$
D_{KL}(\pi_\theta\Vert\pi_{ref})
$$

样本来自当前策略，所以：

$$
q=\pi_\theta,\qquad
p=\pi_{ref},\qquad
x=\frac{\pi_{ref}}{\pi_\theta}
$$

因此 k1 可以写成：

$$
k_1
=-log\frac{\pi_{ref}}{\pi_\theta}
=\log\pi_\theta-\log\pi_{ref}
$$

### 2. 当前代码实际使用了哪个估计器

reference KL 的计算是：

```python
kl = logprobs - ref_logprobs
```

它对应 k1，并在 `torch.no_grad()` 中作为 reward shaping：

```python
kl_reward = -args.kl_coef * kl
```

代码还有一个：

```python
approxkl = 0.5 * (logprobs_diff**2).mean()
```

它的形式是 k2，但要注意：

- 它比较的是 current policy 与 rollout old policy；
- 它只用于日志；
- 它没有加入训练 loss；
- 它不是 reference policy 的 KL regularization。

当前代码没有实现 k3。

### 3. REINFORCE++-baseline 为什么使用 k2

论文当前版本中：

- 普通 REINFORCE++ 使用 k1 reference KL 做 reward shaping；
- REINFORCE++-baseline 使用组均值 baseline、全局优势标准化和单独的 k2 reference KL loss；
- GRPO 的常见实现使用 k3 形式的 KL loss。

k2 作为 KL 数值的 Monte Carlo 估计通常是二阶近似，但将平方 log-ratio 直接作为可微 loss 时，其样本梯度可以匹配论文采用的 reverse-KL 实用梯度。

k3 作为 KL 值估计具有非负等优点，但其中包含：

$$
\frac{\pi_{ref}}{\pi_\theta}
$$

当当前策略给某个 token 的概率很小时，这个比率可能非常大，带来较高方差和数值不稳定。

论文：<https://arxiv.org/html/2501.03262>

## 文档描述与当前代码的差异

### 1. Baseline 版本没有真正实现独立 k2 KL loss

当前代码先把 k1 KL 放入 `rlhf_reward`，然后对包含 KL 的总奖励执行组均值相减：

```python
rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
rlhf_reward = rlhf_reward - rlhf_reward.mean(dim=0, keepdim=True)
```

但论文当前版本的 baseline 方案应当是：

```text
group-centered task reward
+ global advantage normalization
+ 独立的 reference k2 KL loss
```

当前 Python 代码没有将 reference k2 加入 loss，因此不能把它视为论文当前版本的完整 REINFORCE++-baseline 实现。

### 2. 全局优势标准化在多 GPU 下不是真正的全局统计

代码直接计算：

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

在单 GPU 上，这可以视为整个 rollout batch 的统计量；在多 GPU 上，每张卡只使用本地 tensor 的均值和标准差，没有先执行跨进程 gather/reduce，因此不符合论文强调的 global batch normalization。

### 3. Token-level PPO loss 的形状和 padding

`mb_advantage` 通常是 `[B]`，token ratio 是 `[B,T]`：

```python
pg_losses = -mb_advantage * ratio
```

当前 `per_device_train_batch_size=1` 时能够广播，但当 $B>1$ 时可能广播失败或错误对齐。通常应当显式扩展：

```python
mb_advantage = mb_advantage.unsqueeze(-1)
```

此外，padding 位置虽然被填成相同的 log-prob，使其 ratio 等于 1，但 `pg_loss_max.mean()` 没有使用有效 token mask，padding token 仍会参与 loss。正确实现应只对有效 response token 做 masked mean。

### 4. `use_baseline` 与 `rloo_k` 必须保持一致

父类根据 `local_batch_size / rloo_k` 读取 prompt，而当前代码只有在 `use_baseline=True` 时才将 prompt 重复 `rloo_k` 次。因此：

- `use_baseline=True, rloo_k=1`：每组只有一个样本，减去自身均值后优势全部为 0；
- `use_baseline=False, rloo_k>1`：rollout 数可能小于后续代码假定的 `local_batch_size`，可能索引越界；
- 当前默认的 `use_baseline=False, rloo_k=1` 是自洽的普通 REINFORCE++ 配置。

## 数据、模型和依赖问题

### 1. 数据路径没有接通

Notebook 保存到：

```text
./data_gsm8k/gsm8k_train.parquet
```

训练脚本读取的却是：

```python
datasets.load_dataset("data")["train"]
```

当前目录中没有 `data`、`data_gsm8k` 或 Notebook 输入所需的 `gsm8k_chinese`。训练前需要统一数据路径，例如显式加载 parquet 文件。

### 2. 模型路径

脚本写的是：

```python
"Qwen2.5-7B-Instruct"
```

当前目录没有同名本地模型。如果从 Hugging Face 下载，通常应使用完整 ID：

```python
"Qwen/Qwen2.5-7B-Instruct"
```

### 3. TRL 版本

代码使用的是旧接口：

```python
RLOOTrainer(
    config=...,
    policy=...,
    ref_policy=...,
    reward_model=...,
)
```

并使用 `rloo_k`、`cliprange`、`kl_coef`、`response_length` 等旧字段，整体与 TRL 0.19.x～0.21.x 接口接近。

TRL 0.22 以后，相关名称已经改为：

| 旧接口 | 新接口 |
| --- | --- |
| `config` | `args` |
| `policy` | `model` |
| `reward_model` | `reward_funcs` |
| `rloo_k` | `num_generations` |
| `cliprange` | `epsilon` |
| `kl_coef` | `beta` |
| `response_length` | `max_completion_length` |

同时，`ref_policy`、`data_collator` 和 `token_level_kl` 等接口也发生了变化。保持当前代码结构时应固定旧版 TRL；若使用新版 TRL，则需要迁移 trainer 和配置接口。

TRL 官方迁移说明：<https://huggingface.co/docs/trl/rloo_trainer>

### 4. 当前环境验证结果

- Python AST 语法解析通过。
- 当前环境没有安装 `torch`、`transformers`、`trl`、`peft`、`accelerate`、`datasets`、`pandas`。
- 目录中没有依赖清单、训练数据和模型文件，因此尚不能进行真实训练验证。

## 其他实现注意事项

- `correctness_reward()` 使用字符串精确比较，`72` 和 `72.0`、带逗号或单位的答案会被判为不同。
- `copy.deepcopy(model)` 会额外复制一份 7B reference model，显存和内存成本较高。
- 父类会关闭 policy/reference model 的 dropout，因此 `lora_dropout=0.1` 很可能也会被关闭。
- 默认 `normalize_reward=False`，所以 reward normalization/clipping 实际没有启用。
- 没有传入 `eval_dataset`，评估生成代码默认不会执行。
- 存在重复/未使用 import，以及频繁调用 `torch.cuda.empty_cache()` 的情况，可进一步清理。

