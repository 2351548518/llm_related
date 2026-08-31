
这段代码的作用是：让第 `head_index` 个 MTP 模块在上一层预测深度的 hidden state 基础上，再看到一个更靠后的真实 token，从而预测更远的下一个 token。

假设：

```text
input_ids = [t1, t2, t3, t4, t5, t6]
hidden_size = H
batch_size = 1
```

主模型产生：

```text
previous_hidden_output = [h1⁰, h2⁰, h3⁰, h4⁰, h5⁰, h6⁰]
```

其中 `hi⁰` 是主模型处理到 `ti` 后的隐藏状态。

### 第一个 MTP 头：`head_index = 0`

```python
mtp_input_ids = input_ids[:, 1:-1]
```

得到：

```text
[t2, t3, t4, t5]
```

之所以不取 `t6`，是因为 `t6` 后面已经没有 token 可以作为预测目标。

经过 embedding：

```text
[e(t2), e(t3), e(t4), e(t5)]
```

然后截取相同长度的上一层 hidden state：

```python
current_hidden_output = previous_hidden_output[:, :4, :]
```

得到：

```text
[h1⁰, h2⁰, h3⁰, h4⁰]
```

二者拼接后的对应关系是：

| 上一层状态 | 新加入的真实 token | 第一个 MTP 头的目标 |
|---|---|---|
| `h1⁰` | `e(t2)` | `t3` |
| `h2⁰` | `e(t3)` | `t4` |
| `h3⁰` | `e(t4)` | `t5` |
| `h4⁰` | `e(t5)` | `t6` |

也就是：

```python
mtp_input = torch.cat(
    [current_hidden_output, input_embed],
    dim=-1
)
```

如果两个输入的形状都是：

```text
[1, 4, H]
```

拼接后的形状就是：

```text
[1, 4, 2H]
```

接着：

```python
mtp_hidden_output = self.mtp_modules[0](mtp_input)
```

把 `[1, 4, 2H]` 映射回：

```text
[1, 4, H]
```

最后共享的词表输出头：

```python
mtp_head_output = self.output_head(mtp_hidden_output)
```

将其映射为词表 logits：

```text
[1, 4, vocab_size]
```

分别用来预测：

```text
[t3, t4, t5, t6]
```

因此，第一个 MTP 头相对于主模型位置预测的是“下下个 token”。

---

### 第二个 MTP 头：`head_index = 1`

此时传入的 `previous_hidden_output` 是第一个 MTP 头的输出：

```text
[h1¹, h2¹, h3¹, h4¹]
```

其中：

```text
h1¹ = MTP₀(h1⁰, e(t2))
h2¹ = MTP₀(h2⁰, e(t3))
...
```

执行：

```python
mtp_input_ids = input_ids[:, 2:-1]
```

得到：

```text
[t3, t4, t5]
```

上一深度的 hidden state 也截成三个：

```text
[h1¹, h2¹, h3¹]
```

对应关系变成：

| 上一深度状态 | 新加入的真实 token | 第二个 MTP 头的目标 |
|---|---|---|
| `h1¹`，已经包含 `t1、t2` 信息 | `e(t3)` | `t4` |
| `h2¹`，已经包含到 `t3` 的信息 | `e(t4)` | `t5` |
| `h3¹`，已经包含到 `t4` 的信息 | `e(t5)` | `t6` |

第二个 MTP 头相对于原始位置预测的是“下下下个 token”。

### 四个 MTP 头的整体关系

对于长度为 6 的序列：

```text
主模型：   t1 → t2，t2 → t3，t3 → t4，t4 → t5，t5 → t6
MTP头0： (t1,t2) → t3，(t2,t3) → t4，…… 
MTP头1： (t1,t2,t3) → t4，……
MTP头2： (t1,t2,t3,t4) → t5，……
MTP头3： (t1,t2,t3,t4,t5) → t6
```

对应的张量长度：

| 模块 | `head_index` | 输入 embedding | 输出目标 | 输出长度 |
|---|---:|---|---|---:|
| MTP 头 0 | 0 | `t2…t5` | `t3…t6` | 4 |
| MTP 头 1 | 1 | `t3…t5` | `t4…t6` | 3 |
| MTP 头 2 | 2 | `t4…t5` | `t5…t6` | 2 |
| MTP 头 3 | 3 | `t5` | `t6` | 1 |

核心就是这个递推关系：

```text
本层状态 = MTP模块(
    上一预测深度的状态,
    当前新增的真实token embedding
)

本层状态 → 预测再后面的一个token
```

训练时这里使用的 `input_ids` 是真实 token，属于 teacher forcing；生成时没有未来真实 token，所以代码里的 `forward_mtp_step()` 会改用上一个头刚刚预测出来的 token。

---

关键区别是：**训练时整条答案已知，可以并行计算所有位置；推理时未来 token 未知，主模型一次通常只能可靠地产生一个新 token。**

所以，“主模型已经预测到 `t6`”只是在训练数据 `[t1,t2,t3,t4,t5,t6]` 全部已知的前提下成立。

### 从同一个位置看，才体现“多 token 预测”

以 `t1` 所在的位置为观察点：

```text
主模型： h1⁰                       → t2
MTP头0：(h1⁰, t2)                 → t3
MTP头1：(h1¹, t3)，h1¹包含t1,t2   → t4
MTP头2：(h1², t4)，h1²包含t1~t3   → t5
MTP头3：(h1³, t5)，h1³包含t1~t4   → t6
```

也就是说，从以 `t1` 为起点的一条预测链上，一次构造出了：

```text
t2、t3、t4、t5、t6
```

“多个预测”指的是：**对于同一个原始位置，训练多个预测深度，让模型预测多个未来 token。**

### 为什么主模型看起来也预测了所有 token？

训练时使用 teacher forcing，完整序列已经提供：

```text
输入：[t1, t2, t3, t4, t5]
标签：[t2, t3, t4, t5, t6]
```

利用 causal mask，主模型可以在一次矩阵计算中并行得到：

```text
看到 t1       → 预测 t2
看到 t1,t2    → 预测 t3
看到 t1,t2,t3 → 预测 t4
...
看到 t1~t5    → 预测 t6
```

但这里主模型预测 `t6` 时，**真实的 `t2、t3、t4、t5` 已经作为输入提供给它了**。

这不代表推理时只输入 `t1`，主模型就能一次产生到 `t6`。

### 推理时的区别

假设当前 prompt 的最后一个 token 是 `t1`。

普通自回归生成需要多次调用主模型：

```text
主模型调用1：t1             → t2
主模型调用2：t1,t2          → t3
主模型调用3：t1,t2,t3       → t4
主模型调用4：t1,t2,t3,t4    → t5
主模型调用5：t1,t2,t3,t4,t5 → t6
```

MTP 推理则是：

```text
主模型：t1 → t2

MTP头0：h(t1) + 预测的t2 → t3
MTP头1：上一层状态 + 预测的t3 → t4
MTP头2：上一层状态 + 预测的t4 → t5
MTP头3：上一层状态 + 预测的t5 → t6
```

于是一次主模型调用后，可以得到一组候选 token：

```text
[t2, t3, t4, t5, t6]
```

然后再让主模型验证这些候选 token，接受正确的连续前缀。

因此，MTP 的重点不是“主模型永远预测不到 `t6`”，而是：

```text
普通生成：需要主模型顺序生成 t2 → t3 → t4 → t5 → t6

MTP生成：主模型生成 t2，轻量MTP模块继续草拟 t3 → t4 → t5 → t6
```

另外，在 DeepSeek-V3 论文中，MTP 的首要目标其实是增加训练信号、促使主模型提前规划未来表示；推理加速只是它可以附带支持的一种用途。