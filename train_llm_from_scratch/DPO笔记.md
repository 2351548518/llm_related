# DPO（Direct Preference Optimization）笔记

> 对应代码：`dpo_train.py`、`dataset.py` 中的 `DPODataset` / `DPODataCollator`
> 基座：SFT 后的自研小模型（`saves/sft`），策略 π 与参考 π_ref 用同一份权重

## 一、DPO 要解决什么问题

RLHF 的经典流程是：先训 reward model，再用 PPO 等算法对齐策略。这条链路长、不稳定、显存开销大（要同时放 policy / ref / reward / critic）。

DPO 的核心洞察：**可以把 reward 直接用策略本身表达出来，从而跳过 reward model，直接用偏好数据优化策略**，把它变成一个简单的（二）分类损失。

## 二、核心推导

**1）RLHF 目标**（带 KL 约束的 reward 最大化）：

$$\max_\pi\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot|x)}\big[r(x,y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)\big)$$

在 $\int\pi=1$ 约束下求最优解，得到闭式：

$$\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y|x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big),\quad Z(x)=\sum_y \pi_{\mathrm{ref}}(y|x)\exp\!\big(\tfrac{r(x,y)}{\beta}\big)$$

**2）反解 reward**（把 reward 用策略写出来）：

$$r(x,y) = \beta\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta\log Z(x)$$

**3）代入 Bradley-Terry 偏好模型** $P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$。注意 $\beta\log Z(x)$ 在相减时抵消：

$$P(y_w\succ y_l\mid x) = \sigma\!\Big(\beta\Big[\log\frac{\pi(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \log\frac{\pi(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Big]\Big)$$

**4）DPO 损失**（对偏好数据取负对数似然）：

$$\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\!\Big(\beta\big[\log\tfrac{\pi(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \log\tfrac{\pi(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\big]\Big)$$

其中 $\log\pi(y|x)$ 就是序列的对数似然：对回答每个 token 的 log-prob 求和。

## 三、代码怎么落地

数据流：`DPODataset` → `DPODataCollator` → `logits_to_probs` → `mask_logits` → `dpo_loss`。

### 1）数据（`DPODataset` / `DPODataCollator`）

- 每条样本是 `{prompt, chosen, rejected}`。`DPODataset.__getitem__` 返回 `[prompt_ids, chosen_ids+eos, rejected_ids+eos]`。
- `DPODataCollator` 把一个 batch 拼成 **前一半 chosen、后一半 rejected**（共 2N 条序列）。对每条：
  - `input_ids = prompt + answer`
  - `labels = [0]*len(prompt) + answer`（**prompt 用 0 屏蔽**，padding 也补 0）
  - 再做 next-token 错位：`input_ids[:-1]`、`labels[1:]`，使 `labels[i]` 恰是 `input_ids[i]` 要预测的下一个 token。

> ⚠️ 注意：这套数据集统一用 **0**（而不是 -100）作为"忽略"标记。`mask_logits` 里 `label != 0` 就靠这个约定来挑出回答 token。代价是：如果回答里恰好出现 token id=0（通常是特殊符），会被误当成忽略位——这里因为 0 一般是 `<unk>`/pad 类特殊符、不出现在正常回答里，所以实际可用，但比 -100 更脆弱。

### 2）`logits_to_probs(logits, labels)`

逐位置算 log-prob：`log_softmax(logits)` 后 `gather` 出 label 对应的那个 log 概率。返回 `(B, L)`，每格是"模型给真实下一个 token 的对数概率"。（命名叫 probs，语义其实是 log-prob。）

### 3）`mask_logits(logits, labels)`

对每条序列，把 `label != 0`（即回答部分）的 log-prob 求和 → **序列级 $\log\pi(y|x)$**。返回 list（每条一个标量），给 `dpo_loss` 用切片 + `torch.cat` 处理。

### 4）`dpo_loss(ref_probs, probs, beta)`

- `split_probs` 按 `len//2` 切：前半 chosen、后半 rejected。
- `pi_logratios = chosen_probs - reject_probs` $= \log\pi(y_w) - \log\pi(y_l)$
- `ref_logratios` $= \log\pi_{\mathrm{ref}}(y_w) - \log\pi_{\mathrm{ref}}(y_l)$
- `logits = pi_logratios - ref_logratios` $= \log\frac{\pi(y_w)}{\pi_{\mathrm{ref}}(y_w)} - \log\frac{\pi(y_l)}{\pi_{\mathrm{ref}}(y_l)}$
- `loss = -logsigmoid(beta * logits)`，再 `.mean()`。

与上面推导的 $\mathcal{L}_{\mathrm{DPO}}$ 完全一致。

### 5）`compute_loss`

- 参考模型 `ref_model` 在 `torch.no_grad()` 下只前向（冻结、不反传）。
- 策略模型 `model` 前向得到 `.logits`（不用模型自带的 loss，损失由 `dpo_loss` 手算）。
- `beta=0.1`。

## 四、β 的作用

β 是相对参考模型的 KL 约束强度：

- β 越大 → 越保守，策略贴近 π_ref；
- β 越小 → 越激进，更愿意偏离 π_ref 去追偏好数据。

本实现 β=0.1（偏温和）。配合很小的学习率 1e-5、cosine 调度、bf16、1 个 epoch。

## 五、参考模型为什么是"同一份 SFT 权重"

DPO 要求 π_ref 是一个**固定的**参考点（通常是 SFT 后、对齐前的模型）。本实现里 `model` 和 `ref_model` 都 `from_pretrained('.../saves/sft')`：策略会随训练变化，参考保持冻结。这样 $\log\frac{\pi}{\pi_{\mathrm{ref}}}$ 才有"相对变化量"的意义。

## 六、训练配置要点

| 项 | 值 | 说明 |
|---|---|---|
| 基座 | SFT 后的小模型（vocab 6400，8 层） | 接在 SFT 之后 |
| π / π_ref | 同一份 SFT 权重 | ref 冻结 |
| per_device_batch | 16 | collator 翻倍 → 实际 8 个偏好对/step |
| grad_accum | 4 | |
| β | 0.1 | |
| lr | 1e-5 | 太大会"训飞" |
| epochs | 1 | 训多了易输出重复内容 |

## 七、易错点 / 注意

1. **label 用 0 作忽略**：见上文，比 -100 脆弱，但本数据集可接受。
2. **shift 在 collator 里做**（`input_ids[:-1]`、`labels[1:]`），所以 `logits_to_probs` 里 `labels[i]` 与 `logits[i]` 已对齐——模型 forward 内部若再对 loss 做 shift 也不影响 `.logits`，因为我们只取 logits、不取模型自带的 loss。
3. **ref 必须冻结**：`ref_model.eval()` 且 `no_grad`，否则梯度会污染参考点。
4. **chosen/rejected 顺序**：依赖 collator"前 chosen 后 rejected"的约定，`dpo_loss` 的 `len//2` 切分才成立。改数据顺序要同步改切分。
5. **被注释的 `training_step`**：是一次"省算力"尝试——想只算一次 ref_probs、对策略多更新几步。因与 Trainer 默认的梯度累积/优化流程耦合不顺而弃用，改回每个 step 重算 ref。
6. **重复输出**：epoch 过多时模型易重复，故只训 1 轮、学习率极小。

## 八、一个完整例子的维度流转

设：词表 `V=6400`；一个 step 取 **2 个偏好对**（pair）；prompt 都长 3；回答（含 eos）长度 chosen1=3 / chosen2=4 / rejected1=2 / rejected2=3。
（用小数字只为看清维度，真实训练时 L=512、B=16）

### 0）原始样本（`DPODataset.__getitem__`，每条返回一个 list）

```
pair1: prompt1=[p,p,p]   chosen1=[a,a,eos](3)     rejected1=[a,eos](2)
pair2: prompt2=[p,p,p]   chosen2=[a,a,a,eos](4)   rejected2=[a,a,eos](3)
```

### 1）`DPODataCollator`：拼成 chosen+rejected、padding、错位

先按"前 chosen、后 rejected"排成 4 条序列（prompt+answer）：

```
seq0 (chosen1)   = [p,p,p, a,a,eos]        len=6
seq1 (chosen2)   = [p,p,p, a,a,a,eos]      len=7
seq2 (rejected1) = [p,p,p, a,eos]          len=5
seq3 (rejected2) = [p,p,p, a,a,eos]        len=6
```

padding 到 batch 内最长 = 7（补 0），labels 把 prompt 位也填 0：

```
input_ids (shift 前): (4, 7)        labels (shift 前): (4, 7)    prompt/pad 位=0
```

再错位 `input_ids[:-1]`、`labels[1:]` → 长度变 6（把最后一个 input 砍掉、第一个 label 砍掉，使 `labels[i]` 对齐 `logits[i]` 要预测的下一 token）：

```
input_ids: (4, 6)        # 形如 [p,p,p, a,a,eos]      （砍掉末尾）
labels:    (4, 6)        # 形如 [0,0, a,a,eos,0]     （prompt 位=0、末尾 padding=0）
```

每条序列里 `label != 0`（即回答 token）的个数 k：

```
seq0 chosen1    k=3     (a,a,eos)
seq1 chosen2    k=4     (a,a,a,eos)
seq2 rejected1  k=2     (a,eos)
seq3 rejected2  k=3     (a,a,eos)
```

### 2）模型 forward → logits

```
model(input_ids)      -> logits      (4, 6, 6400)     # 每位置对词表预测下一 token 的分布
ref_model(input_ids)  -> ref_logits  (4, 6, 6400)     # no_grad，只前向不反传
```

### 3）`logits_to_probs`：取出"真实下一 token"的 log 概率

```
logits               (4, 6, 6400)
log_softmax(dim=2)   (4, 6, 6400)
labels.unsqueeze(2)  (4, 6, 1)
gather(dim=2)         (4, 6, 1)
squeeze(-1)           (4, 6)                         # 每位置 log p(真实下一 token)
```

策略与参考各得到一份：`probs (4,6)`、`ref_probs (4,6)`。

### 4）`mask_logits`：每条序列求和 → 序列级 log p(answer|prompt)

```
zip 到 batch 维：4 次，每次  logit (6,)、label (6,)
  logit[label != 0]   (k,)            # 只取回答 token，k 因序列而异(3/4/2/3)
  .sum()              ()  (0 维标量)
  .unsqueeze(0)       (1,)
new_logits = list[4]，每元素 (1,)                    # 注意：返回的是 list 不是 tensor
```

策略 `probs` → `list[4]×(1,)`；参考 `ref_probs` → `list[4]×(1,)`。

### 5）`dpo_loss`：切分 + 相减 + logsigmoid

```
split_probs：len//2 = 2
  torch.cat(chosen[:2])  -> (2,)      # [log π(y_w) of pair1, pair2]
  torch.cat(reject[2:])  -> (2,)      # [log π(y_l) of pair1, pair2]
chosen_probs (2,)    reject_probs (2,)
ref_chosen   (2,)    ref_reject   (2,)

pi_logratios  = chosen_probs - reject_probs      (2,)   # log[π(y_w)/π(y_l)]
ref_logratios = ref_chosen - ref_reject           (2,)   # log[π_ref(y_w)/π_ref(y_l)]
logits        = pi_logratios - ref_logratios      (2,)   # 相对 ref 的偏好对数似然比
loss          = -logsigmoid(0.1 * logits)          (2,)
loss.mean()                                          ()  标量
```

最终一个 step 的 loss 是一个**标量**，反传到策略模型。

### 维度变化总览

| 阶段 | shape |
|---|---|
| collator 输出 input_ids / labels | `(4, 6)` |
| forward logits | `(4, 6, 6400)` |
| `logits_to_probs` | `(4, 6)` |
| `mask_logits`（返回 list） | `list[4]` × `(1,)` |
| `split_probs` 内 `torch.cat` 后 | `(2,)` |
| `dpo_loss` 的 logits / loss | `(2,)` |
| `loss.mean()` | `()` 标量 |

一句话：`(4,6,6400)` → gather 成 `(4,6)` → 每行求和成 `(1,)`×4 → cat 切成 chosen/rejected 各 `(2,)` → 相减得 `(2,)` → logsigmoid+mean 成标量。

## 九、为什么 `mask_logits` 的"求和"有意义（对数域核心）

最容易卡住的一点：`mask_logits` 把每个回答 token 的概率"求和"，有意义吗？

**关键：求和的是 log-prob（对数概率），不是概率本身。** 函数 `logits_to_probs` 名字有误导——它先 `log_softmax` 再 gather，返回的其实是 $\log p$。

### 1）对数域里，和 = 积的对数

$$\log(a\cdot b)=\log a+\log b\;\Rightarrow\;\sum_t \log \pi(y_t\mid y_{<t},x)=\log\prod_t \pi(y_t\mid y_{<t},x)=\log\pi(y\mid x)$$

把每个 token 的 log-prob 加起来，得到的就是**序列的对数似然** $\log\pi(y\mid x)$——正是 DPO 损失里要的那一项。

### 2）为什么连乘 = 联合概率

自回归分解 + teacher forcing：把**真实回答**整段喂进去，每个位置读"预测真实下一 token"的概率，且每个 $\pi(y_t\mid y_{<t},x)$ 都以**真实**前文为条件，故

$$\pi(y_1\mid x)\,\pi(y_2\mid y_1,x)\,\pi(y_3\mid y_{1:2},x)\cdots=\pi(y\mid x)$$

连乘 = 生成这条回答的联合概率；取对数 → 求和。这就是 `mask_logits` 做的事。

### 3）小数字验证

设回答 2 个 token，给定上下文 $p_1=0.5$、$p_2=0.4$：

| | 概率 | log 概率 |
|---|---|---|
| token1 | 0.5 | -0.693 |
| token2 | 0.4 | -0.916 |
| **联合** | 0.5×0.4 = **0.2** | log(0.2) = **-1.609** |

`-0.693 + (-0.916) = -1.609` ✓ —— log-prob 求和 = log(联合)。

### 4）如果真的是"概率求和"就没意义了

把概率（非 log）相加 $0.5+0.4=0.9$，没有概率含义，甚至可能 >1。所以求和只在 log 域才成立——这也是整条链路从头到尾都在 log 域（`log_softmax` 出来就没离开过对数）的原因，`dpo_loss` 里的相减 `chosen_probs - reject_probs` 才能化成 $\log\frac{\pi(y_w)}{\pi(y_l)}$ 的对数似然比。

> 一句话：`mask_logits` 的"求和"是在对数域做的，**和 = log(积) = 序列联合对数似然**，即 DPO 需要的 $\log\pi(y\mid x)$。
