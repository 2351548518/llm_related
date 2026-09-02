# Engram

> 本文整理自 Engram 架构说明截图，并结合 [`engram.ipynb`](./engram.ipynb) 中的实现与后续讨论，补充原理、代码、张量形状及推理流程。

## 背景

语言模型需要完成两类任务：

- **动态推理**：根据当前上下文进行组合、推理和生成。
- **静态检索**：查找命名实体、固定表达、公式化模式等相对稳定的局部知识。

Transformer 缺少原生的知识查找（静态检索）机制，通常只能通过多层计算来模拟检索过程，因此效率较低。

文本中的大量内容（例如命名实体和公式化模式）具有局部、静态、高度固定的特点，基本不会发生变化。这类内容没有必要反复计算，可以通过低成本的查表操作快速获取，查找复杂度接近 $O(1)$。

Engram 使用查表方式处理需要重复计算的局部依赖，释放注意力容量，使注意力机制能够更加专注于全局上下文，从而改善长上下文场景中的表现。

## “静态检索”究竟查找什么？

这里的“检索”不是 RAG 式的文档搜索，也不是直接查询事实答案或下一个 token。Engram 查找的是：

> 当前已经出现的局部 N-gram 在训练参数表中对应的连续特征向量。

形式上可以写成：

$$
\text{已观测到的局部 N-gram}
\longrightarrow
\text{可训练的 Engram Embedding}
$$

例如模型处理到：

```text
Only Alexander the Great
```

在 `Great` 位置，可以构造：

```text
2-gram: the Great
3-gram: Alexander the Great
```

经过多头哈希后，从参数表中取出多个向量并拼接：

$$
e_t=
E_2[\operatorname{hash}(\text{the Great})]
\mathbin\Vert
E_3[\operatorname{hash}(\text{Alexander the Great})]
$$

真实实现中，每种 N-gram 使用多个哈希头，因此上式的每一项还会展开为多个向量片段。概念上的“一条 N-gram 对应一个向量”只是简化说法。

这些向量不是可读文本，也没有人工指定的字段。训练后，它们可能隐式编码对语言模型有用的局部特征：

| 局部模式 | Engram 可能学习到的特征 |
|---|---|
| `Alexander the Great` | 完整人名、历史人物、`Great` 是称号 |
| `as a result` | 固定连接表达、后续常出现 `of` |
| `for i in` | Python 循环结构、后续可能出现可迭代对象 |
| `New York City` | 完整地名，而不是三个彼此独立的 token |

这些只是帮助理解的语义描述。实际信息分布在高维浮点向量中，通常无法把某一个维度解释成某一条明确事实。

### 为什么称为“静态”？

推理期间，以下部分是静态的：

- N-gram 到哈希索引的映射是确定的；
- Engram Embedding 表已经训练完成并被冻结；
- 同一个 N-gram 总是访问相同的一组基础向量；
- 查表只依赖 token ID，不依赖运行时隐藏状态。

但查出的向量是否适合当前语境是动态的。Engram 会利用当前隐藏状态计算门控：

```text
静态 N-gram 查表
        +
动态上下文门控
        ↓
条件记忆（Conditional Memory）
```

### 为什么可以减少重复计算？

没有 Engram 时，Transformer 需要通过若干层 Attention 和 MLP 逐步组合：

```text
Alexander + the + Great
             ↓
      多层动态计算
             ↓
识别为一个完整命名实体
```

有 Engram 时，可以直接利用局部 token 计算哈希并取出训练好的短语级特征：

```text
Alexander the Great
          ↓
        哈希
          ↓
    短语级特征向量
```

这并不会让 Attention 停止工作，而是在训练中形成分工：Engram 更适合处理固定、局部、重复出现的模式，Attention 则可以把更多能力用于长距离和动态关系。

这里的 $O(1)$ 指单次哈希表访问的复杂度不随整张 Engram 表的大小增加。对于长度为 $L$、N-gram 种类数为 $N$、每种 N-gram 有 $K$ 个哈希头的序列，总查询量仍约为 $O(LNK)$，实际延迟还受内存带宽、缓存命中率和设备通信影响。

## 整体结构

Engram 被插入指定的 Transformer Block 中，其位置在 Attention 和 MoE 之前：

```text
Input IDs
   │
Vocab Embedding
   │
Engram ──→ Residual
   │
Attention ──→ Residual
   │
MoE ──→ Residual
   │
 Output
```

Engram 内部的数据流为：

```text
Input IDs
   │
压缩 Token ID
   │
构造 2-gram / 3-gram
   │
多头哈希
   │
查询 N-gram Embedding
   │
拼接 Embedding
   ├──────────────→ Value 投影 ───────────────┐
   │                                          │
   └→ Key 投影 ← Input Hidden（Query）→ 门控 ─┤
                                              │
                                  门控 Value + 短卷积
                                              │
                                           Output
```

## 实现步骤

### 1. 分词器压缩

原始词表通常很大，例如：

- Qwen：151665
- DeepSeek：129280

直接组合 N-gram 会导致组合数量爆炸，因此首先压缩原始词表，尽量减小词表规模以及可能产生的 N-gram 数量。

#### 为什么原始词表可以压缩？

大模型分词器通常使用无损重建机制。对于一些语义等价、但形式不完全相同的 token，分词器可能分配不同的 ID，例如：

```text
hello: 14990
Hello: 9707
```

因此可以对原始词表进行标准化，包括：

- Unicode 规范化；
- 转换为小写；
- 去除重音符号；
- 归一化或去除多余空白。

在 `engram.ipynb` 的 Qwen2.5-0.5B 示例中，压缩结果为：

```python
print("压缩后的词表大小:", len(compressed_tokenizer))
print("原始词表大小:", compressed_tokenizer.tokenizer.vocab_size)
print(
    "压缩率:",
    1 - len(compressed_tokenizer) / compressed_tokenizer.tokenizer.vocab_size,
)
```

```text
压缩后的词表大小: 107453
原始词表大小: 151643
压缩率: 0.29140810983691956
```

标准化可以把形式不同但归一化结果相同的 token 映射到同一个压缩 ID：

```python
input_ids = compressed_tokenizer.tokenizer.encode("hello world, Hello world")
print("原始 input_ids:", input_ids)

compressed_input_ids = compressed_tokenizer(input_ids)
print("压缩后的 input_ids:", compressed_input_ids)
```

```text
原始 input_ids: [14990, 1879, 11, 21927, 1879]
压缩后的 input_ids: [6378, 1346, 11, 6378, 1346]
```

这里的 `hello` 和 `Hello` 最终都映射为压缩 ID `6378`。

### 2. 多头哈希

为所有可能的 N-gram 组合分别建立 Embedding 查找表是不现实的。Engram 使用哈希将 N-gram 映射到固定大小的虚拟词表中，并通过多头哈希尽量减小冲突影响。

具体过程如下：

1. 为每个指定的模型层初始化一组奇数乘子，不同层使用不同的随机种子。
2. 获取以当前 token 结尾的 N-gram，例如 2-gram 和 3-gram。
3. 将 N-gram 中各 token 的压缩 ID 与对应乘子相乘，再对结果执行按位异或，得到混合值 `mix`：

   $$
   \operatorname{mix}_t^{(n)}=
   \bigoplus_{k=0}^{n-1}\left(x_{t-k}\cdot m_k\right)
   $$

4. 每种 N-gram 配置多个哈希头。每个头使用不同的质数词表大小，对 `mix` 取模：

   $$
   \operatorname{hash}_{t,j}^{(n)}=
   \operatorname{mix}_t^{(n)}\bmod N_{n,j}
   $$

5. 使用各个哈希值从对应的 Embedding 表中检索向量。
6. 拼接当前 token 的所有 N-gram、所有哈希头对应的向量，得到该 token 的 N-gram Embedding。

Notebook 默认配置包含 2-gram 和 3-gram，每种 N-gram 使用 8 个哈希头，因此每个 token 会得到：

$$
(3-1)\times 8=16
$$

个哈希索引。每个头检索一个 64 维向量，拼接后得到：

$$
16\times 64=1024
$$

维的 Engram 表征。

对应的张量变化为：

```text
哈希索引:       [B, L, 16]
多头 Embedding: [B, L, 16, 64]
拼接后:         [B, L, 1024]
```

### 3. Engram 表征的用途

Transformer 会为每个 token 位置维护一个隐藏状态：

```text
Only      → h₁
Alexander → h₂
the       → h₃
Great     → h₄
```

`Great` 位置的 $h_4$ 不只表示单词 `Great`，还包含模型当前对前缀 `Only Alexander the Great` 的动态理解。拼接得到的 Engram 表征 $e_4$ 则是根据局部 N-gram 查出的静态先验。

Engram 表征不会直接作为答案输出，它的用途是：

> 经过 Key/Value 投影、上下文门控和短卷积后，作为额外特征加到当前位置的隐藏状态中。

```text
N-gram Embedding eₜ
      ├── Key 投影 ──→ 与 hidden state 计算相关性
      └── Value 投影 ───────────────┐
                                     ↓
hidden state hₜ ──→ gate αₜ ──→ αₜ·Value(eₜ)
                                     ↓
                             短卷积与残差融合
                                     ↓
                         更新后的 hidden state h'ₜ
```

基本计算为：

$$
k_t=W_Ke_t,\qquad v_t=W_Ve_t
$$

$$
\alpha_t=
\sigma\left(
\frac{
\operatorname{RMSNorm}(h_t)^\top
\operatorname{RMSNorm}(k_t)
}{\sqrt d}
\right)
$$

$$
\widetilde v_t=\alpha_tv_t
$$

$$
h'_t=h_t+\operatorname{EngramOutput}(\widetilde v_t)
$$

更新后的隐藏状态会继续经过 Attention、MoE 和后续 Transformer 层，最终由 LM Head 转换成下一个 token 的概率：

$$
P(x_{t+1}\mid x_{\leq t})=
\operatorname{Softmax}\left(W_{\text{LM}}h_t^{\text{final}}\right)
$$

因此，Engram 不直接返回 `could`、`was` 等答案，而是改变隐藏状态，间接调整这些候选 token 的概率。

此外，更新后的隐藏状态会参与生成 Attention 的 Key/Value。它们进入 KV Cache 后，后续 token 仍然可以通过 Attention 访问已经注入的局部模式信息。

### 4. 上下文门控

N-gram 本身只包含局部 token 信息，不包含完整上下文。为了判断检索到的 N-gram 信息对当前上下文是否有用，Engram 使用门控机制进行过滤：

- 与当前上下文相关的信息被保留；
- 与当前上下文无关的信息被抑制；
- 相关程度由一个 $0$ 到 $1$ 之间的门控分数表示。

该过程与注意力机制类似：

- 当前隐藏状态 $h_t$ 作为 Query；
- N-gram Embedding $e_t$ 经线性投影得到 Key 和 Value；
- Query 与 Key 计算点积相关性；
- 相关性经过 Sigmoid 得到门控值；
- 门控值与 Value 逐元素相乘，过滤无关信息。

## 多分支架构集成

对于第 $m$ 个 hyper-connection 分支，隐藏状态和 Engram Embedding 的相关性为：

$$
\alpha_t^{(m)}=
\sigma\left(
\frac{
\operatorname{RMSNorm}\!\left(h_t^{(m)}\right)^\top
\operatorname{RMSNorm}\!\left(W_K^{(m)}e_t\right)
}{\sqrt{d}}
\right)
$$

其中：

- $h_t^{(m)}$ 是第 $m$ 个分支在位置 $t$ 的隐藏状态；
- $e_t$ 是位置 $t$ 检索到的 Engram Embedding；
- $W_K^{(m)}$ 是该分支独立使用的 Key 投影矩阵；
- $d$ 是隐藏维度；
- $\alpha_t^{(m)}$ 是该分支的门控值。

每个分支拥有独立的 Key 投影，用于实现特定分支的门控；所有分支共享 Engram Embedding 和 Value 投影。

```python
gates = []

for hc_idx in range(backbone_config.hc_mult):
    key = self.key_projs[hc_idx](embeddings)
    normed_key = self.norm1[hc_idx](key)

    query = hidden_states[:, :, hc_idx, :]
    normed_query = self.norm2[hc_idx](query)

    gate = (normed_key * normed_query).sum(dim=-1)
    gate = gate / math.sqrt(backbone_config.hidden_size)
    gate = gate.sigmoid().unsqueeze(-1)
    gates.append(gate)
```

将所有分支的门控值堆叠，然后过滤共享的 Value：

```python
# gates: [B, L, hc_mult, 1]
gates = torch.stack(gates, dim=2)
value = gates * self.value_proj(embeddings).unsqueeze(2)
```

Notebook 的实际实现还在 Sigmoid 之前加入了保留符号的平方根变换：

```python
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
```

## 短卷积

门控后的 Value 经过短卷积，以扩大局部感受野并增强模型的非线性。最终输出为：

$$
\mathbf{Y}=
\operatorname{SiLU}\!\left(
\operatorname{Conv1D}\!\left(
\operatorname{RMSNorm}\!\left(\widetilde{\mathbf{V}}\right)
\right)
\right)
+\widetilde{\mathbf{V}}
$$

对应代码为：

```python
output = value + self.short_conv(value)
```

默认配置下，Engram 模块的输入和输出形状如下：

```text
hidden_states: [B, L, 4, 1024]
input_ids:     [B, L]
output:        [B, L, 4, 1024]
```

输出可以直接通过残差连接加回 Transformer Block 的隐藏状态。

## 自回归位置偏移：Engram 如何帮助预测下一个 token

Engram 使用已经观测到的 token 构造 N-gram，再利用查出的特征帮助预测下一个 token。它不能使用尚未生成的 token 作为查询条件，否则会泄漏未来信息。

假设初始 Prompt 是：

```text
Only Alexander the
```

生成过程如下：

| 阶段 | 当前已经观测到的前缀 | 当前查询的 N-gram | 正在预测 |
|---|---|---|---|
| Prefill 最后位置 | `Only Alexander the` | `Alexander the`、`Only Alexander the` | `Great` |
| Decode 第 1 步 | `... Alexander the Great` | `the Great`、`Alexander the Great` | `could` |
| Decode 第 2 步 | `... the Great could` | `Great could`、`the Great could` | `tame` |
| Decode 第 3 步 | `... Great could tame` | `could tame`、`Great could tame` | `the` |

上一步生成的 token 会在下一步变成已知输入：

```text
上一步输出 token
       ↓
加入当前前缀
       ↓
作为下一次 Decode 的输入
       ↓
构造新的后缀 N-gram
       ↓
预测再下一个 token
```

因此，生成 `could` 后查询 `Great could` 并不是为了再次处理已经完成的输出，而是为了给预测 `tame` 提供局部模式特征。如果某个局部组合本身没有用，或者发生哈希冲突，上下文门控可以将其抑制。

### 与经典 N-gram 语言模型的区别

经典 N-gram 模型直接统计局部前缀后的 token 分布：

```text
("Alexander", "the")
    → {"Great": 0.93, "king": 0.02, ...}
```

Engram 不直接保存概率，而是保存可训练的潜在特征：

```text
("Alexander", "the")
    → [0.18, -0.73, 1.21, ...]
```

随后把这份特征与完整上下文融合，再由后续 Transformer 和 LM Head 计算 token 分布：

```text
局部 N-gram 记忆
        +
完整上下文 hidden state
        ↓
上下文门控与 Transformer
        ↓
下一个 token 的概率
```

同一个短语可以有多种合理后续：

```text
Alexander the Great was ...
Alexander the Great conquered ...
Alexander the Great became ...
Alexander the Great could ...
```

因此不能简单保存 `Alexander the Great → could`。Engram 向量需要学习对多种上下文普遍有帮助的实体、句法和续写特征，具体输出仍由完整上下文决定。

## 训练与推理

### 训练阶段

Engram Embedding 初始化时只是随机参数，没有预先写入文本、事实或答案。在教师强制训练中，每个位置使用截至当前位置的 N-gram 查表，并预测下一个 token。

例如训练样本包括：

```text
Only Alexander the Great → could
Alexander the Great      → conquered
Alexander the Great      → was
```

这些样本会访问部分相同的 Engram 行。预测损失通过下面的路径更新这些参数：

```text
Next-token Loss
       ↓
LM Head
       ↓
后续 Transformer 层
       ↓
Engram 残差输出
       ├── Value 投影
       ├── 上下文门控
       └── 本次访问的 Embedding 行
```

对应的参数更新可写成：

$$
e\leftarrow e-\eta\frac{\partial\mathcal{L}}{\partial e}
$$

由于同一个 N-gram 会出现在许多不同上下文中，它对应的向量不能只表示某一个固定后续，而会逐渐学习一组对多种预测都有帮助的局部潜在特征。

Engram 可以插入多个指定的 Transformer Block。大规模分布式训练时，Embedding 表会切分到多张 GPU；前向阶段通过 All-to-All 收集本批次激活的行，反向阶段把梯度发送回持有相应参数分片的设备。

### 推理阶段

推理期间，Engram 表参数被冻结。查表地址完全由输入 token ID 决定，不需要等待隐藏状态或 MoE 路由结果，因此可以提前计算地址和预取向量。

Engram 不必全部驻留在加速器显存中，可以把大表卸载到主机内存或分层内存系统：

- Transformer、Attention、MoE 和门控计算保留在设备侧；
- Input IDs 同时用于设备侧计算和主机侧 Engram 查表；
- 主机完成 N-gram 哈希与 Embedding 检索后，将结果发送回设备；
- 高频 Embedding 行可以缓存在 GPU HBM 或主机 DRAM，低频长尾可以放在更大、更慢的存储层；
- 传输过程可以与 Engram 层之前的 Transformer 计算重叠。

这种设计将计算密集的动态推理与内存密集的静态知识检索分离，使模型可以在不持续占用大量加速器显存的情况下扩展参数化记忆。

### Prefill

Prefill 一次处理完整 Prompt。由于所有 Prompt token 都已知，可以提前为所有位置并行构造后缀 N-gram：

```text
完整 Prompt IDs
       ↓
并行压缩所有 token ID
       ↓
并行计算所有位置的 2-gram / 3-gram 哈希
       ↓
批量查询或预取所有 Engram Embedding
       ↓
执行 Transformer Forward
       ↓
在指定 Engram 层利用当前 hidden state 做门控融合
       ↓
Attention 生成 Prompt 的 KV Cache
```

对于序列长度 $L$，默认配置每个位置有 16 个哈希索引，因此每个 Engram 层大约查询 $16L$ 个 Embedding 行。实现时还可以合并重复索引，减少重复的内存读取。

查表可以提前，但门控不能完全提前：Embedding $e_t$ 只依赖 token，可以预取；门控还需要指定层产生的隐藏状态 $h_t$，必须等模型计算到该层时执行。

Engram 没有单独的 Attention KV Cache，但 Engram 更新后的隐藏状态会进入后续 Attention，因此 Prefill 生成的 K/V 已经间接包含 Engram 注入的信息。

### Decode

Decode 每次只处理最新的一个输入 token。最大 N-gram 阶数为 $N$ 时，只需额外保存最近 $N-1$ 个压缩 token ID。

例如刚生成 `could`，下一轮 Decode 会把它作为输入，并构造：

```text
2-gram: Great could
3-gram: the Great could
```

这些查询结果用于增强 `could` 位置的隐藏状态，从而帮助预测再下一个 token，例如 `tame`。

单步 Decode 流程为：

```text
上一步生成的新 token xₜ
       ↓
与最后 N-1 个 token 组成后缀 N-gram
       ↓
计算固定数量的哈希索引并查询 Embedding
       ↓
在指定层与当前 token 的 hidden state 做门控融合
       ↓
结合已有 Attention KV Cache 执行增量前向
       ↓
预测 xₜ₊₁，并把 xₜ₊₁ 作为下一轮输入
```

先前位置不需要重新查表，因为位置 $t$ 的 Engram 只依赖 $x_{\leq t}$，未来 token 不会改变已经计算过的因果后缀 N-gram。过去位置注入的 Engram 信息已经进入其隐藏表示和 Attention KV Cache，后续 token 可以通过 Attention 继续访问。

短卷积需要单独维护增量状态。若卷积核大小为 $w$、膨胀率为 $\delta$，实现通常需要保留覆盖 $(w-1)\delta$ 历史距离的门控 Value，以避免每一步重新卷积整段历史。

### Prefill 与 Decode 对比

| 项目 | Prefill | Decode |
|---|---|---|
| 一次处理的 token | 完整 Prompt | 最新的一个 token |
| N-gram 计算 | 所有位置并行 | 只计算最新位置 |
| 默认单层查询量 | 约 $16L$ | 每序列每步约 16 个索引 |
| Attention KV Cache | 创建 Prompt 的全部 K/V | 每步追加一个 K/V |
| Engram 历史处理 | 所有位置批量查表 | 不重新查询历史位置 |
| 额外状态 | 批量 Engram 结果 | 最近 $N-1$ 个 ID、短卷积状态 |
| 可预取范围 | 整个 Prompt 的地址 | 当前步地址；未来 token 地址未知 |

### Decode 时的异步预取

未来 token 尚未生成，因此不能提前查询任意多个未来步骤。但一个 token 一旦采样完成，下一轮所需的 N-gram 地址就立即确定，可以把主机查表和设备上的前置层计算重叠：

```text
CPU / Host                         GPU
生成 token 后计算 Engram 地址      开始下一轮早期层计算
            ↓                            ↓
从主机内存读取 Embedding  ───────→ 到达 Engram 层时使用
```

Engram 放置得更深，会获得更长的通信隐藏窗口；但从建模角度看，较早注入更有利于避免前几层重复重建局部模式。因此插入层位置需要同时考虑模型效果和系统延迟。

### 当前 notebook 的范围

`engram.ipynb` 对整段输入一次性执行前向，更接近一个 Prefill/训练结构演示。它目前没有实现：

- 自回归生成循环；
- Attention KV Cache；
- 增量 N-gram 状态；
- 短卷积状态缓存；
- 主机内存到 GPU 的异步预取；
- 面向 GPU 的哈希与设备放置处理。

因此 notebook 证明的是 Engram 的核心查表、门控、卷积与张量形状能够走通，而不是一套完整的高性能推理服务实现。

## 核心理解

Engram 的完整功能链条可以概括为：

```text
已经观测到的局部 N-gram
        ↓
确定性哈希查询训练参数表
        ↓
取得与该局部模式关联的静态潜在特征
        ↓
根据当前 hidden state 动态门控
        ↓
把有用特征加入当前位置的 hidden state
        ↓
参与后续 Attention / MoE / LM Head
        ↓
改变下一个 token 的概率，并通过 KV Cache 影响后续生成
```

一句话总结：

> Engram 用已经出现的局部 token 模式查询可训练的短语级特征，把这些特征作为额外信息注入 Transformer；它检索的是帮助预测的潜在表示，而不是文档、明确事实、最终答案或固定的下一个 token。

## 参考资料

- [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/html/2601.07372v2)
- [DeepSeek-AI/Engram 官方仓库](https://github.com/deepseek-ai/Engram)
- [`engram.ipynb`](./engram.ipynb)
