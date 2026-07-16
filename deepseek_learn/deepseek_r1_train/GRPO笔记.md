# GRPO 学习笔记：从公式到代码与奖励曲线

> 配套代码：[deepseek_r1_train.py](./deepseek_r1_train.py)
>
> 配套图片：[deepseek_reward.png](./deepseek_reward.png)
>
> 示例任务：Qwen2.5-0.5B-Instruct + 中文 GSM8K + TRL `GRPOTrainer`

这份代码并不是完整复现 DeepSeek-R1，而是一个小型教学实验：让模型针对同一道数学题生成多条回答，用规则奖励判断回答质量，再通过 GRPO 增大“相对较好回答”的生成概率。

---

## 1. 一句话理解 GRPO

GRPO（Group Relative Policy Optimization，组相对策略优化）的核心是：

> 对同一道题生成一组回答，在组内比较奖励；高于组内平均水平的回答得到正优势，低于平均水平的回答得到负优势，然后用类似 PPO 的裁剪目标更新模型。

它与 PPO 的关键区别是：GRPO 不需要单独训练一个 Critic/Value Model 来估计优势，而是直接使用同一问题下多条回答的相对奖励作为基线。

```text
问题 q
  │
  ├─生成回答 o₁ ──奖励 r₁
  ├─生成回答 o₂ ──奖励 r₂
  ├─生成回答 o₃ ──奖励 r₃
  └─生成回答 o₄ ──奖励 r₄
              │
              ▼
     组内均值与标准差
              │
              ▼
       相对优势 A₁...A₄
              │
              ▼
       裁剪策略目标更新模型
```

代码中的 `num_generations=16` 表示实际对每个问题生成 16 条回答。后文为了方便手算，会把例子缩小成 4 条回答；原理完全相同。

---

## 2. 为什么不用 Critic

传统 PPO 式 RLHF 通常涉及以下组件：

| 组件 | 作用 | 是否更新 |
|---|---|---|
| Policy/Actor，$\pi_\theta$ | 生成回答 | 更新 |
| Value/Critic，$V_\phi$ | 估计状态价值，辅助计算优势 | 更新 |
| Reference Policy，$\pi_{\mathrm{ref}}$ | 约束策略不要偏离初始模型太远 | 冻结或周期更新 |
| Reward Model/Reward Function | 给回答打分 | 通常冻结；规则函数无需训练 |

Critic 会带来额外的模型参数、优化器状态和训练不稳定性。GRPO 省掉 Critic，使用组内奖励均值作为 baseline：

$$
\text{baseline}(q)=\frac{1}{G}\sum_{j=1}^{G}r_j
$$

但这不是“免费午餐”：每个问题必须采样 $G$ 条回答，生成阶段仍然需要较多算力和显存。

---

## 3. 符号表

| 符号 | 含义 | 代码对应 |
|---|---|---|
| $q$ | 输入问题/prompt | 数据集中的 `prompt` |
| $o_i$ | 同一问题的第 $i$ 条回答 | `completions[i]` |
| $G$ | 每个问题的采样数 | `num_generations=16` |
| $r_i$ | 第 $i$ 条回答的总奖励 | 5 个奖励函数之和 |
| $\hat A_i$ | 第 $i$ 条回答的组相对优势 | `GRPOTrainer` 内部计算 |
| $\pi_\theta$ | 当前正在训练的策略模型 | `model` |
| $\pi_{\theta_{\mathrm{old}}}$ | 生成当前训练样本时的旧策略 | Trainer 内部维护 |
| $\pi_{\mathrm{ref}}$ | KL 约束使用的参考策略 | 是否启用取决于 `beta` 和 TRL 版本 |
| $\epsilon$ | PPO/GRPO 裁剪范围 | `GRPOConfig` 的 `epsilon` |
| $\beta$ | KL 惩罚系数 | `GRPOConfig` 的 `beta` |

---

## 4. 组相对优势：GRPO 最关键的公式

对同一个问题生成 $G$ 条回答，并得到奖励 $r_1,r_2,\ldots,r_G$。第 $i$ 条回答的优势为：

$$
\hat A_i=
\frac{r_i-\operatorname{mean}(r_1,r_2,\ldots,r_G)}
{\operatorname{std}(r_1,r_2,\ldots,r_G)}
$$

可以把它理解成奖励的组内标准分数：

- $\hat A_i>0$：这条回答优于同题的平均回答，应提高它的生成概率；
- $\hat A_i<0$：这条回答低于平均水平，应降低它的生成概率；
- $\hat A_i\approx0$：这条回答接近平均水平，更新信号较弱；
- 如果组内奖励完全相同，标准差接近 0，这个问题几乎不能提供相对学习信号。

在当前代码中，奖励是对整条回答给出的，因此同一回答中的各个 completion token 通常共享同一个 $\hat A_i$。

### 为什么是“相对”奖励

假设一条回答奖励是 1.5：

- 如果同组其他回答都是 0，它是好回答，会得到正优势；
- 如果同组其他回答都是 3.0，它反而是差回答，会得到负优势。

所以 GRPO 关心的不是“1.5 分绝对高不高”，而是“它在同一道题的这组回答中好不好”。

---

## 5. GRPO 策略目标

DeepSeekMath 给出的 token 级 GRPO 目标可以写成：

$$
\begin{aligned}
J_{\mathrm{GRPO}}(\theta)
=\mathbb E\Bigg[
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\Big(&\min\big(
\rho_{i,t}(\theta)\hat A_i,
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat A_i
\big)\\
&-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
\Big)
\Bigg].
\end{aligned}
$$

其中重要性采样比率为：

$$
\rho_{i,t}(\theta)=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
}.
$$

它表示“当前策略生成这个 token 的概率”相对于“旧策略生成这个 token 的概率”改变了多少：

- $\rho=1$：概率没有变化；
- $\rho>1$：当前模型更倾向生成该 token；
- $\rho<1$：当前模型更不倾向生成该 token。

### 裁剪为什么重要

假设 $\epsilon=0.2$，裁剪区间就是 $[0.8,1.2]$。

对于正优势回答，目标希望增大它的概率；但即便 $\rho$ 从 1.0 一下升到 1.8，裁剪项也只按 1.2 计算，避免一次更新走得过远。负优势时则反过来约束概率下降。

### KL 项

原始 GRPO 使用以下非负 KL 估计：

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
=
\frac{\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})}
{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
-\log
\frac{\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})}
{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
-1.
$$

它用于阻止训练后的模型偏离参考模型太远。

需要特别注意：本项目代码没有显式设置 `beta`。不同 TRL 版本的默认值可能不同；当前 TRL 文档中的默认值是 `beta=0.0`，此时不会加载参考模型，也不会计算 KL。因此，不能仅根据代码里的 `GRPOTrainer` 就断定该实验一定启用了 KL。为了实验可复现，应该显式填写 `beta`、`epsilon` 和 TRL 版本。

---

## 6. 代码中的数据如何进入 GRPO

原始数据大致包含：

```python
{
    "question_zh-cn": "小明有 3 个苹果，又买了 2 个，一共有几个？",
    "answer_only": "5",
}
```

`process_data` 将其转换为：

```python
{
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "小明有 3 个苹果，又买了 2 个，一共有几个？"},
    ],
    "answer": "5",
}
```

其中：

- `prompt` 会送入模型；
- `answer` 不会作为提示告诉模型，而是传给 `correctness_reward` 进行评分；
- `SYSTEM_PROMPT` 要求模型使用 `<think>...</think>` 和 `<answer>...</answer>`。

聊天格式下，一条生成结果大致是：

```python
completion = [
    {
        "role": "assistant",
        "content": "<think>\n3+2=5\n</think>\n<answer>\n5\n</answer>\n",
    }
]
```

因此奖励函数使用 `completion[0]["content"]` 取得生成文本。

---

## 7. 五个奖励函数如何计算

代码将五项奖励直接相加：

$$
r_i=r_{\mathrm{mark}}+r_{\mathrm{soft}}+r_{\mathrm{hard}}
+r_{\mathrm{digit}}+r_{\mathrm{correct}}.
$$

| 奖励函数 | 分数 | 实际条件 |
|---|---:|---|
| `mark_reward` | 0～0.5 | 四个指定标签/换行片段各值 0.125 |
| `soft_format_reward` | 0 或 0.5 | 正则匹配单行形式的 think/answer 结构 |
| `hard_format_reward` | 0 或 0.5 | 严格匹配换行、标签顺序和结尾换行 |
| `digit_reward` | 0 或 0.5 | 提取答案后，`str.isdigit()` 为真 |
| `correctness_reward` | 0 或 2.0 | 提取结果与标准答案字符串完全相同 |

### 7.1 `extract_answer`

```python
text = "<think>3+2=5</think><answer>5</answer>"
```

经过两次 `split` 后提取出字符串 `"5"`。如果没有 `<answer>` 标签，函数会把整段输出当成候选答案；它是宽容的提取器，不负责验证格式。

### 7.2 `digit_reward` 的边界

| 提取结果 | `isdigit()` | 奖励 |
|---|---:|---:|
| `"5"` | True | 0.5 |
| `"-5"` | False | 0 |
| `"3.14"` | False | 0 |
| `"1,000"` | False | 0 |
| `"答案是5"` | False | 0 |

所以它只适合当前答案为非负整数的简化数据。

### 7.3 格式正则存在冲突

`hard_format_reward` 的正则要求 `<think>` 后面立刻换行：

```text
<think>\n...\n</think>\n<answer>\n...\n</answer>\n
```

但 `soft_format_reward` 使用的 `.` 没有启用 `re.DOTALL`，不能跨过换行。因此一条回答通过 hard 格式时，通常不能同时通过 soft 格式。

所以五项标称上限相加虽然是 4.0，但按当前代码真正可达到的上限是 3.5，而不是 4.0。

---

## 8. 完整数值例子：奖励如何变成优势

为了手算，假设同一道题只生成 $G=4$ 条回答，标准答案为 `5`。

### 回答与奖励

| 回答 | 情况 | mark | soft | hard | digit | correct | 总奖励 |
|---|---|---:|---:|---:|---:|---:|---:|
| $o_1$ | 严格格式，答案 5 | 0.5 | 0 | 0.5 | 0.5 | 2.0 | **3.5** |
| $o_2$ | 严格格式，答案 4 | 0.5 | 0 | 0.5 | 0.5 | 0 | **1.5** |
| $o_3$ | 单行软格式，答案 5 | 0 | 0.5 | 0 | 0.5 | 2.0 | **3.0** |
| $o_4$ | 输出“不会” | 0 | 0 | 0 | 0 | 0 | **0** |

组内平均奖励：

$$
\bar r=\frac{3.5+1.5+3.0+0}{4}=2.0.
$$

为了便于演示，使用总体标准差：

$$
\sigma=
\sqrt{\frac{(3.5-2)^2+(1.5-2)^2+(3-2)^2+(0-2)^2}{4}}
\approx1.369.
$$

于是：

$$
\hat A_1=\frac{3.5-2}{1.369}\approx1.095,
\qquad
\hat A_2=\frac{1.5-2}{1.369}\approx-0.365,
$$

$$
\hat A_3=\frac{3.0-2}{1.369}\approx0.730,
\qquad
\hat A_4=\frac{0-2}{1.369}\approx-1.461.
$$

含义：

- $o_1$ 最好，得到最大正优势；
- $o_3$ 虽然格式不严格，但答案正确，仍得到正优势；
- $o_2$ 看起来格式完美且输出了数字，但答案错误，低于组内平均，因此得到负优势；
- $o_4$ 最差，得到最大的负优势。

实际 TRL 对标准差的具体约定、数值稳定项和多设备聚合方式与版本有关，因此手算值可能与日志略有不同，但“减均值、除标准差、组内比较”的含义不变。

---

## 9. 从代码配置映射到公式

```python
training_args = GRPOConfig(
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    temperature=1.0,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    max_grad_norm=0.1,
    use_vllm=False,
)
```

| 配置 | 含义 |
|---|---|
| `num_generations=16` | 每个问题生成 $G=16$ 条回答，是组相对比较的基础 |
| `temperature=1.0` | rollout 采样温度，控制同组回答的随机性和多样性 |
| `per_device_train_batch_size=1` | 每张卡每个微批次处理的 prompt 数 |
| `gradient_accumulation_steps=4` | 累积多个微批次后再更新参数 |
| `max_prompt_length=256` | 问题和系统提示的最大 token 数 |
| `max_completion_length=200` | 推理和答案合计最多生成 200 token |
| `learning_rate=5e-6` | 策略更新步长 |
| `max_grad_norm=0.1` | 梯度裁剪，防止梯度爆炸 |
| `use_vllm=False` | 使用 Transformers 生成，而不是 vLLM |

当前代码没有固定 `trl` 版本。不同版本的 `num_generations`、有效 batch size、KL 默认值和 loss 聚合方式可能不同。例如当前 TRL 文档要求有效训练 batch size 能被 `num_generations` 整除；单卡下 `1×4=4` 不能被 16 整除，可能需要更多 GPU、增大 batch/累积步数，或者减小 `num_generations`。

为了可复现，建议配置至少显式加入：

```python
GRPOConfig(
    beta=0.001,       # 是否使用 KL 必须明确；0 表示关闭
    epsilon=0.2,      # 裁剪范围必须明确
    num_generations=4,# 小显存教学实验可先减小
    temperature=1.0  # 显式固定 rollout 的探索强度
)
```

这里的数值只是教学配置示例，不表示它们一定是该任务的最优值。

### 9.1 温度对 GRPO 的作用

温度作用于模型生成下一个 token 时的概率分布。设模型输出的 logit 为 $z_j$，温度为 $T$，采样概率为：

$$
p_T(j)=\frac{\exp(z_j/T)}{\sum_k\exp(z_k/T)}.
$$

| 温度范围 | 概率分布 | 对同题多次采样的影响 | 对 GRPO 的影响 |
|---|---|---|---|
| $T<1$ | 分布更尖锐 | 回答更相似、更确定 | 组内奖励方差可能过小，相对优势信号变弱 |
| $T=1$ | 保持模型原始分布 | 探索与稳定性较均衡 | 适合作为 R1-Zero 风格训练的起点 |
| $T>1$ | 分布更平坦 | 回答更随机、更多样 | 探索更强，但错误、乱码和奖励噪声也可能增加 |

GRPO 必须依靠同一问题下多条回答的差异进行组内比较。如果温度过低，16 条回答可能几乎相同：

$$
r_1\approx r_2\approx\cdots\approx r_{16}
\quad\Longrightarrow\quad
\operatorname{std}(r_1,\ldots,r_{16})\approx0.
$$

此时即使生成了 16 条回答，也很难得到有区分度的相对优势。反过来，温度过高虽然提高了组内多样性，却可能使大部分回答无意义，导致奖励噪声变大、训练不稳定。

因此，温度不是 GRPO 目标函数中的优化系数，而是 **rollout 数据分布的控制参数**：它决定模型拿什么样的回答进行组内比较，进而间接影响奖励均值、奖励标准差和梯度质量。

DeepSeek-R1-Zero 和 DeepSeek-R1 第一阶段推理 RL 的 rollout 温度均为 1.0；第二阶段通用对齐 RL 将温度降到 0.7，以减少高温采样导致的不连贯输出。当前代码属于第一阶段推理 RL 风格，因此显式设置：

```python
temperature=1.0
```

是合理的，也避免了不同 TRL 版本默认值变化带来的复现差异。调参时应同时观察：

- `reward_std`：过低可能说明组内回答缺乏差异；
- 正确性奖励：高温不能只带来随机错误；
- completion 样本：检查推理是否连贯、是否出现乱码或格式崩坏；
- 输出熵和重复率：判断探索不足还是探索过强。

---

## 10. 奖励曲线解读

![GRPO 训练奖励曲线](./deepseek_reward.png)

图片包含总奖励、奖励标准差以及多个子奖励曲线。大致可以观察到：

1. `train/reward` 从接近 0 上升，最后稳定在约 **1.5**；
2. `digit_reward` 最后接近 **0.5**；
3. `hard_format_reward` 最后接近 **0.5**；
4. `mark_reward` 最后接近 **0.5**；
5. `correctness_reward` 早期上升，中期达到峰值，但后期下降到接近 **0**；
6. `reward_std` 后期明显下降，说明同组回答的奖励差异变小，相对优势信号也在变弱。

前三个辅助奖励之和恰好约为：

$$
0.5+0.5+0.5=1.5.
$$

这与最终总奖励约 1.5 高度吻合。最合理的解释不是“模型数学能力稳定提升”，而是：

> 模型主要学会了严格输出标签和一个数字，但这个数字经常不正确。

这是一种典型的奖励投机（reward hacking）或奖励目标错配：代理指标在变好，真正关心的正确性却在恶化。

如果图中的 `correctness_reward` 是每批次的平均原始奖励，因为每个正确答案得 2.0，那么：

$$
\text{近似正确率}
\approx\frac{\operatorname{mean}(r_{\mathrm{correct}})}{2}.
$$

例如曲线值为 0.5 时，近似表示 25% 的回答正确；曲线值为 0.04 时，近似只有 2%。这是根据代码奖励定义进行的推算，实际还应结合日志聚合方式核对。

---

## 11. 为什么会出现奖励投机

### 原因一：辅助奖励太容易

错误答案只要具有严格格式并且是数字，就能得到：

$$
r_{\mathrm{wrong}}=0.5_{\mathrm{mark}}+0.5_{\mathrm{hard}}+0.5_{\mathrm{digit}}=1.5.
$$

模型找到这个捷径后，不需要真正解决数学题也能稳定获得奖励。

### 原因二：正确性奖励过于稀疏

`correctness_reward` 只做字符串完全相等比较：

- `5` 正确；
- `5.0` 错误；
- `答案是 5` 错误；
- 等价分数、带单位答案也可能被判错。

奖励噪声和误判会让真正正确的推理得不到稳定反馈。

### 原因三：格式规则与提示词不完全一致

提示词鼓励多行输出，但 soft 正则不能跨行；hard 正则则要求固定换行和末尾换行。模型优化的是这些字符串细节，而不是推理过程本身。

### 原因四：组内方差越来越小

当 16 条回答都学会输出相似格式和数字时，它们可能都得到约 1.5。此时：

$$
\operatorname{std}(r_1,\ldots,r_G)\rightarrow0,
$$

组内难以区分好坏回答，GRPO 的有效学习信号变弱。

---

## 12. 改进方向

1. **先保证 correctness 主导目标**：降低格式/数字奖励，或对辅助奖励设置随训练衰减的权重。
2. **改进答案归一化**：统一空白、逗号、小数、分数、负号和单位，再判断数学等价性。
3. **修复格式正则**：使用 `re.DOTALL` 或 `re.fullmatch`，使规则与 SYSTEM_PROMPT 的多行格式一致。
4. **记录真实准确率**：除了总 reward，还要单独评测 held-out GSM8K accuracy，不能用总 reward 代替能力指标。
5. **检查零方差组**：记录每个 prompt 的 reward std 和全相同奖励比例。
6. **显式固定版本和超参数**：记录 `trl`、`transformers`、`torch` 版本，并显式设置 `beta`、`epsilon`、采样温度和随机种子。
7. **增加验证集与训练前后对照**：比较 base checkpoint 与 RL checkpoint，而不是只看训练集奖励。
8. **控制推理长度**：`max_completion_length=200` 对长 CoT 偏短，应监控截断比例并根据显存调整。

---

## 13. GRPO 与这份代码的关系

| DeepSeek-R1/GRPO 概念 | 本代码实现 | 差异或简化 |
|---|---|---|
| 每题组采样 | `num_generations=16` | 与 R1 每题 16 个输出一致 |
| 组相对优势 | `GRPOTrainer` 内部完成 | 代码没有手写公式 |
| 准确性奖励 | `correctness_reward` | 仅字符串精确匹配 |
| 格式奖励 | mark/soft/hard 三项 | 比论文的格式奖励更细碎，也更容易被利用 |
| 语言一致性奖励 | 未实现 | 文档提到，但代码缺失 |
| 参考模型与 KL | 未显式配置 `beta` | 实际行为依赖 TRL 版本 |
| 起始模型 | Qwen2.5-0.5B-Instruct | R1-Zero 从 Base 模型开始，不是 Instruct |
| 完整 R1 流程 | 未实现 | 没有冷启动 SFT、拒绝采样、二次 SFT 和通用对齐 RL |

因此更准确的说法是：

> 这份代码演示了 R1-Zero 风格的“规则奖励 + GRPO”训练范式，但不是 DeepSeek-R1 或 R1-Zero 的完整复现。

---

## 14. 最后速记

1. GRPO 的“Group”表示同一个问题生成一组回答。
2. GRPO 的“Relative”表示奖励要减去组内均值，再除以组内标准差。
3. 不需要 Critic，是 GRPO 相比 PPO 的主要资源优势。
4. $\rho$ 衡量当前策略相对旧策略的概率变化，clip 限制单次更新幅度。
5. KL 是否存在取决于 $\beta$；本代码没有显式设置，不能假定一定启用。
6. 奖励函数决定模型真正学到什么，而不仅仅是训练是否收敛。
7. 当前曲线显示“格式奖励收敛、正确性退化”，是奖励投机的强烈信号。
8. 评估 GRPO 不能只看总 reward，必须同时看正确率、截断率、组内方差和独立验证集。

---

## 参考资料

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)——GRPO 的原始论文与 token 级目标。
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)——R1/R1-Zero 的训练流程、规则奖励与 GRPO 应用。
- [Hugging Face TRL: GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)——当前 `GRPOTrainer` 的公式、配置和日志指标说明。
