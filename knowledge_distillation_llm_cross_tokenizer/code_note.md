# 1

```python
    def merge_prob_with_alignment_groups(self, probs, alignment_groups):
        """
        把多个 token 位置的概率合并成一个对齐位置。

        输入 ``probs`` 的形状为 ``[answer_len, vocab_size]``。例如 group 为
        ``[[0, 1], [2]]``，输出长度会从 3 变为 2：位置 0、1 合并，位置 2
        原样保留。

        多位置组使用的当前规则是：对同一 vocab id 的概率取乘积后重新归一化::

            merged[v] = normalize(probs[0, v] * probs[1, v] * ...)

        代码在 log 空间执行乘法以提高数值稳定性。这是一种实验性启发式，
        并不等同于两个连续 token 组成文本片段的严格联合概率。
        """

        if not alignment_groups:
            # 无需对齐时保持原始序列长度和概率不变。
            return probs

        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device)

    
        for group_idx, group in enumerate(alignment_groups):
            if len(group) > 1:
                eps = 1e-8
                # log(a*b) = log(a) + log(b)；clamp_min 防止 log(0)。
                logp = torch.log(probs[group[0]].clamp_min(eps))
                for idx in group[1:]:
                    if idx < probs.size(0):
                        logp = logp + torch.log(probs[idx].clamp_min(eps))
                aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                # 某一 tokenizer 没有与另一边剩余文本对应的 token 时，用全零
                # 概率向量占位，使学生、教师拥有相同的对齐组数量。
                aligned_probs[group_idx] = torch.zeros_like(probs[0])

        return aligned_probs
```

这段函数的作用是：

> 把学生模型或教师模型中“多个 token 对应同一个文本片段”的概率分布合并，使两边的序列长度一致。

例如，学生 tokenizer 把“九月”切成两个 token：

```text
学生：["偷", "星", "九", "月", "333"]
教师：["偷", "星", "九月", "333"]
```

对应的分组可能是：

```python
student_groups = [[0], [1], [2, 3], [4]]
teacher_groups = [[0], [1], [2], [3]]
```

其中：

```text
学生的 [2, 3] 表示“九月”
教师的 [2] 表示“九月”
```

因此，学生侧需要把第 2、3 个 token 的概率合并成一个位置。

## 输入形状

假设：

```python
probs.shape == [3, 4]
```

表示：

- 序列长度为 3；
- 词表大小为 4；
- 每一行是一个 token 位置上的概率分布。

例如：

```python
probs = [
    [0.70, 0.20, 0.10, 0.00],  # 第 0 个 token
    [0.40, 0.10, 0.40, 0.10],  # 第 1 个 token
    [0.10, 0.20, 0.30, 0.40],  # 第 2 个 token
]
```

如果：

```python
alignment_groups = [[0, 1], [2]]
```

目标就是：

- 把第 0、1 个位置合并；
- 保留第 2 个位置；
- 最终输出形状从 `[3, 4]` 变成 `[2, 4]`。

## 1. 没有分组时

```python
if not alignment_groups:
    return probs
```

如果没有对齐分组，函数直接返回原始概率：

```text
输入形状：[3, 4]
输出形状：[3, 4]
```

## 2. 创建输出张量

```python
vocab_size = probs.size(-1)
target_len = len(alignment_groups)
aligned_probs = torch.zeros(
    target_len,
    vocab_size,
    device=probs.device,
)
```

在这个例子中：

```python
target_len = 2
vocab_size = 4
```

因此创建：

```text
aligned_probs.shape = [2, 4]
```

## 3. 多 token 分组：概率相乘

对于：

```python
group = [0, 1]
```

代码先取出第 0 行概率：

```python
[0.70, 0.20, 0.10, 0.00]
```

然后取出第 1 行概率：

```python
[0.40, 0.10, 0.40, 0.10]
```

对相同的词表维度逐元素相乘：

```text
[0.70 × 0.40,
 0.20 × 0.10,
 0.10 × 0.40,
 0.00 × 0.10]
```

得到：

```text
[0.28, 0.02, 0.04, 0.00]
```

代码使用 log 空间实现这个乘法：

```python
logp = torch.log(probs[group[0]].clamp_min(eps))

for idx in group[1:]:
    logp = logp + torch.log(probs[idx].clamp_min(eps))
```

因为：

```text
log(a × b) = log(a) + log(b)
```

这样做比直接连续相乘更稳定，尤其是概率很小时。

## 4. 使用 softmax 重新归一化

概率相乘之后：

```text
[0.28, 0.02, 0.04, 0.00]
```

总和不再是 1，所以需要重新归一化：

```python
aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
```

近似得到：

```text
[0.8235, 0.0588, 0.1176, 0.0000]
```

这个结果表示：

> 只有在原来多个 token 位置上都具有较高概率的词表项，合并后才会保持较高概率。

这是一种“共同支持”的效果。

## 5. 单 token 分组：直接复制

对于：

```python
group = [2]
```

代码执行：

```python
aligned_probs[group_idx] = probs[group[0]]
```

也就是直接复制：

```text
输入第 2 行：[0.10, 0.20, 0.30, 0.40]
输出第 1 行：[0.10, 0.20, 0.30, 0.40]
```

所以最终结果大致是：

```text
[
    [0.8235, 0.0588, 0.1176, 0.0000],
    [0.1000, 0.2000, 0.3000, 0.4000],
]
```

形状从：

```text
[3, 4]
```

变成：

```text
[2, 4]
```

## 6. 空分组：用全零占位

```python
else:
    aligned_probs[group_idx] = torch.zeros_like(probs[0])
```

如果某个分组是：

```python
group = []
```

则输出一行全零：

```text
[0.0, 0.0, 0.0, 0.0]
```

这样做的目的是保证学生和教师拥有相同数量的对齐位置，但这行并不是一个合法的概率分布，后续计算 KL 时需要特别注意。

## 总结

这个函数可以概括为：

```text
多个 token 对应一个文本片段
        ↓
取这些位置的概率分布
        ↓
逐词表维度相乘
        ↓
softmax 重新归一化
        ↓
得到一个新的对齐位置
```

不过要注意，这里的概率相乘是一种实验性启发式，并不是严格的连续 token 联合概率。它的作用主要是让不同 tokenizer 的序列长度对齐，从而后续可以比较学生和教师的概率分布。

# 2

```python
    def compute_hybrid_uld_loss(self, student_aligned, teacher_aligned):
        """
        在每个已对齐文本位置比较两个不同大小的词表分布。

        输入形状为::

            student_aligned: [aligned_len, student_vocab_size]
            teacher_aligned: [aligned_len, teacher_vocab_size]

        假设教师词表有 5 个 token、学生词表有 4 个 token，其中 3 个 token
        字符串可以一一匹配，则：

        * matched 分支比较 3 对具有相同字符串语义的概率；
        * unmatched 分支分别取教师剩余 2 维、学生剩余 1 维，降序排列后把
          较短的一边补零，再计算 L1；
        * 权重为 ``matched_weight=3/5``、``unmatched_weight=2/5``。

        排序后的 unmatched 分支只比较概率分布的“形状”，不再假设两边第 k
        个 token 具有相同语义。
        """

        device = student_aligned.device
        student_vocab_size = student_aligned.size(-1)
        teacher_vocab_size = teacher_aligned.size(-1)

        if self.teacher_matched_ids:
            # 两个 id tensor 的相同下标代表同一个 token 字符串：
            # teacher_matched_token_ids[k] <-> student_matched_token_ids[k]。
            teacher_matched_token_ids = torch.tensor(sorted(self.teacher_matched_ids), dtype=torch.long, device=device)
            student_matched_token_ids = torch.tensor(
                [self.vocab_mapping[token_id.item()] for token_id in teacher_matched_token_ids], dtype=torch.long, device=device
            )
        else:
            teacher_matched_token_ids = torch.tensor([], dtype=torch.long, device=device)
            student_matched_token_ids = torch.tensor([], dtype=torch.long, device=device)

        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)

        if len(teacher_matched_token_ids) > 0:
            # True 表示该词表位置已经有跨 tokenizer 的明确映射。
            teacher_matched_mask[teacher_matched_token_ids] = True
            student_matched_mask[student_matched_token_ids] = True

        matched_loss = torch.tensor(0.0, device=device)
        matched_token_count = 0
        if len(teacher_matched_token_ids) > 0:
            # 按映射后的相同顺序抽取概率，得到 [aligned_len, num_matched]。
            teacher_matched_probs = teacher_aligned[:, teacher_matched_token_ids]  # [seq_len, num_matched]
            student_matched_probs = student_aligned[:, student_matched_token_ids]  # [seq_len, num_matched]
            matched_token_count = teacher_matched_probs.size(-1)
            matched_loss = self.compute_kl_loss(student_matched_probs, teacher_matched_probs)

        # 取反后，mask 选中的都是无法按 token 字符串建立映射的词表项。
        teacher_unmatched_mask = ~teacher_matched_mask
        student_unmatched_mask = ~student_matched_mask

        teacher_unmatched_probs = teacher_aligned[:, teacher_unmatched_mask]  # [seq_len, num_teacher_unmatched]
        student_unmatched_probs = student_aligned[:, student_unmatched_mask]  # [seq_len, num_student_unmatched]

        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_probs.size(-1) > 0 and student_unmatched_probs.size(-1) > 0:
            # 例：[0.1, 0.7, 0.2] 排序后为 [0.7, 0.2, 0.1]。排序会丢弃
            # 原 token id，只保留高、中、低概率的相对形状。
            teacher_unmatched_sorted = teacher_unmatched_probs.sort(dim=-1, descending=True).values
            student_unmatched_sorted = student_unmatched_probs.sort(dim=-1, descending=True).values

            teacher_unmatched_size = teacher_unmatched_sorted.size(-1)
            student_unmatched_size = student_unmatched_sorted.size(-1)
            max_unmatched_size = max(teacher_unmatched_size, student_unmatched_size)

            # 两边 unmatched 维数不同，较短的一侧在 vocab 维右侧补 0。
            if teacher_unmatched_size < max_unmatched_size:
                teacher_unmatched_sorted = F.pad(
                    teacher_unmatched_sorted, (0, max_unmatched_size - teacher_unmatched_size)
                )
            if student_unmatched_size < max_unmatched_size:
                student_unmatched_sorted = F.pad(
                    student_unmatched_sorted, (0, max_unmatched_size - student_unmatched_size)
                )

            unmatched_loss = F.l1_loss(student_unmatched_sorted, teacher_unmatched_sorted, reduction="sum")
            # 先对所有位置和词表维求和，再除以对齐后的序列长度。
            unmatched_loss /= student_aligned.size(0)  

        # 当前实现以“教师词表中有多少比例可匹配”作为两类损失的权重。
        matched_weight = matched_token_count / max(1, teacher_vocab_size)
        unmatched_weight = 1.0 - matched_weight

        total_loss = matched_weight * matched_loss + unmatched_weight * unmatched_loss

        return total_loss
```


这段函数负责处理“词表大小不同”的情况。

它把两个模型的概率分布拆成两部分：

1. **matched**：两个 tokenizer 中能找到相同文本的 token，使用 KL 散度；
2. **unmatched**：找不到对应关系的 token，排序后使用 L1 距离。

最后把两部分损失加权求和。

---

## 1. 输入示例

假设已经完成序列长度对齐：

```python
student_aligned.shape = [2, 4]
teacher_aligned.shape = [2, 5]
```

其中：

- `2`：有两个已经对齐的文本位置；
- 学生词表大小为 `4`；
- 教师词表大小为 `5`。

学生词表：

| student id | token |
|---:|---|
| 0 | 猫 |
| 1 | 狗 |
| 2 | 跑 |
| 3 | 红 |

教师词表：

| teacher id | token |
|---:|---|
| 0 | 鸟 |
| 1 | 猫 |
| 2 | 蓝 |
| 3 | 狗 |
| 4 | 跑 |

虽然 token id 不一样，但通过 token 文本可以建立映射：

```text
教师 id 1（猫） -> 学生 id 0（猫）
教师 id 3（狗） -> 学生 id 1（狗）
教师 id 4（跑） -> 学生 id 2（跑）
```

因此：

```python
self.vocab_mapping = {
    1: 0,
    3: 1,
    4: 2,
}
```

匹配的 token 是：

```text
猫、狗、跑
```

不匹配的 token 是：

```text
教师侧：鸟、蓝
学生侧：红
```

---

## 2. 构造 matched token id

```python
teacher_matched_token_ids = torch.tensor(
    [1, 3, 4],
    dtype=torch.long,
)
```

这些是教师侧匹配 token 的 id。

然后通过映射得到学生侧对应的 id：

```python
student_matched_token_ids = torch.tensor(
    [0, 1, 2],
    dtype=torch.long,
)
```

这两个 tensor 的相同位置表示相同文本：

```text
teacher_matched_token_ids: [1, 3, 4]
student_matched_token_ids: [0, 1, 2]

对应关系：
教师 1 ↔ 学生 0，即“猫”
教师 3 ↔ 学生 1，即“狗”
教师 4 ↔ 学生 2，即“跑”
```

`sorted(self.teacher_matched_ids)` 的作用是保证每次取出的顺序一致。

---

## 3. 构造 matched mask

教师词表大小为 5：

```python
teacher_matched_mask = [False, True, False, True, True]
```

含义是：

```text
teacher id 0：鸟，不匹配
teacher id 1：猫，匹配
teacher id 2：蓝，不匹配
teacher id 3：狗，匹配
teacher id 4：跑，匹配
```

学生词表大小为 4：

```python
student_matched_mask = [True, True, True, False]
```

含义是：

```text
student id 0：猫，匹配
student id 1：狗，匹配
student id 2：跑，匹配
student id 3：红，不匹配
```

---

## 4. 取出 matched 概率

假设某一个对齐位置上的概率是：

```python
student_aligned = [0.50, 0.30, 0.10, 0.10]
```

对应：

```text
猫 0.50
狗 0.30
跑 0.10
红 0.10
```

教师概率是：

```python
teacher_aligned = [0.10, 0.45, 0.10, 0.25, 0.10]
```

对应：

```text
鸟 0.10
猫 0.45
蓝 0.10
狗 0.25
跑 0.10
```

执行：

```python
teacher_matched_probs = teacher_aligned[:, [1, 3, 4]]
student_matched_probs = student_aligned[:, [0, 1, 2]]
```

得到：

```text
教师 matched：[0.45, 0.25, 0.10]
学生 matched：[0.50, 0.30, 0.10]
```

因为这些 token 具有明确的文本对应关系，所以可以计算 KL 散度：

```python
matched_loss = self.compute_kl_loss(
    student_matched_probs,
    teacher_matched_probs,
)
```

它比较的是：

```text
学生“猫”的概率 vs 教师“猫”的概率
学生“狗”的概率 vs 教师“狗”的概率
学生“跑”的概率 vs 教师“跑”的概率
```

而不是比较词表中相同位置的概率。

---

## 5. 取出 unmatched 概率

代码使用取反后的 mask：

```python
teacher_unmatched_mask = ~teacher_matched_mask
student_unmatched_mask = ~student_matched_mask
```

得到：

```text
教师 unmatched：[鸟, 蓝]
学生 unmatched：[红]
```

对应概率为：

```text
teacher_unmatched_probs = [0.10, 0.10]
student_unmatched_probs = [0.10]
```

如果有多个序列位置，形状大致是：

```text
teacher_unmatched_probs: [aligned_len, 2]
student_unmatched_probs: [aligned_len, 1]
```

---

## 6. unmatched 部分排序

```python
teacher_unmatched_sorted = teacher_unmatched_probs.sort(
    dim=-1,
    descending=True,
).values

student_unmatched_sorted = student_unmatched_probs.sort(
    dim=-1,
    descending=True,
).values
```

排序的目的，是忽略具体 token 身份，只比较概率分布的形状。

例如：

```text
学生 unmatched：[红 0.10]
教师 unmatched：[鸟 0.10, 蓝 0.10]
```

这里不能认为：

```text
红 ↔ 鸟
```

所以只比较：

```text
学生的第 1 大 unmatched 概率
教师的第 1 大 unmatched 概率
教师的第 2 大 unmatched 概率
```

---

## 7. 对较短的 unmatched 分布补零

学生 unmatched 维度为 1，教师 unmatched 维度为 2：

```python
student_unmatched_size = 1
teacher_unmatched_size = 2
max_unmatched_size = 2
```

于是学生分布补零：

```text
学生：[0.10] -> [0.10, 0.00]
教师：[0.10, 0.10]
```

代码：

```python
student_unmatched_sorted = F.pad(
    student_unmatched_sorted,
    (0, 2 - 1),
)
```

这里 `(0, 1)` 表示在最后一个维度的右侧补 1 个零。

---

## 8. 计算 unmatched L1 损失

```python
unmatched_loss = F.l1_loss(
    student_unmatched_sorted,
    teacher_unmatched_sorted,
    reduction="sum",
)
```

对于一个位置：

```text
|0.10 - 0.10| + |0.00 - 0.10|
= 0.00 + 0.10
= 0.10
```

如果有两个对齐位置，假设：

```text
第 1 个位置的 L1 = 0.10
第 2 个位置的 L1 = 0.05
```

那么：

```python
unmatched_loss = 0.10 + 0.05
```

接着除以对齐序列长度：

```python
unmatched_loss /= student_aligned.size(0)
```

如果序列长度为 2：

```text
unmatched_loss = 0.15 / 2 = 0.075
```

---

## 9. 计算两部分的权重

教师词表大小为 5，匹配 token 数量为 3：

```python
matched_token_count = 3
teacher_vocab_size = 5
```

因此：

```python
matched_weight = 3 / 5 = 0.6
unmatched_weight = 1 - 0.6 = 0.4
```

代码：

```python
matched_weight = matched_token_count / max(1, teacher_vocab_size)
unmatched_weight = 1.0 - matched_weight
```

匹配 token 越多，KL 部分的权重越大；匹配 token 越少，unmatched 的 L1 部分权重越大。

---

## 10. 得到最终损失

假设：

```text
matched_loss = 0.02
unmatched_loss = 0.075
```

则：

```text
total_loss
= 0.6 × 0.02 + 0.4 × 0.075
= 0.012 + 0.030
= 0.042
```

对应代码：

```python
total_loss = (
    matched_weight * matched_loss
    + unmatched_weight * unmatched_loss
)
```

整个流程可以概括为：

```text
比较两个词表
    |
    +-- token 文本相同
    |       |
    |       +-- 按对应 token 提取概率
    |               |
    |               +-- KL 散度
    |
    +-- token 文本不同
            |
            +-- 提取 unmatched 概率
                    |
                    +-- 排序
                    +-- padding
                    +-- L1 距离

KL 损失 + L1 损失
        |
        +-- 按匹配比例加权
                |
                +-- 最终蒸馏损失
```

需要注意，当前代码中 `compute_kl_loss` 接收到的是已经 softmax 后的概率，但函数内部又执行了一次 `log_softmax`。因此上面的解释是它的设计意图；实际运行时的 KL 数值还会受到这一实现细节影响。