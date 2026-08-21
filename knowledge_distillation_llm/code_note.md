
# 1

```python
    def selective_log_softmax(self, logits, index):
        """
        logits.shape = [batch, seq_len, vocab_size]
        index.shape  = [batch, seq_len]
        
        只取实际生成 token 的 log probability。

        例：某位置三个词的 log-softmax 为 ``[-2.0, -0.3, -1.8]``，实际生成
        token id=1，则返回 ``-0.3``。输入形状分别为
        ``[batch, seq_len, vocab]`` 和 ``[batch, seq_len]``，输出为
        ``[batch, seq_len]``。
        """

        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values 
        else:
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index): 
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps
```

这段函数的作用是：从完整词表 logits 中，只取模型“实际生成的 token”的对数概率，用于后面的 PPO 重要性采样。

假设：

```python
logits.shape = [batch, seq_len, vocab_size]
index.shape  = [batch, seq_len]
```

`index[b, t]` 表示第 `b` 条序列在第 `t` 个位置实际生成的 token ID。最终返回：

```python
per_token_logps.shape = [batch, seq_len]
```

核心公式是：

```text
log P(token_id)
= token_id 对应的 logit
  - logsumexp(该位置所有词的 logits)
```

例如：

```python
logits = [2.0, 1.0, 0.0]
index = 1
```

计算：

```text
logsumexp = log(e² + e¹ + e⁰) ≈ 2.4076
log P(token=1) = 1.0 - 2.4076 = -1.4076
P(token=1) = exp(-1.4076) ≈ 0.2447
```

各行代码的含义如下。

```python
selected_logits = torch.gather(
    logits,
    dim=-1,
    index=index.unsqueeze(-1)
).squeeze(-1)
```

假设：

```python
logits.shape            # [B, S, V]
index.shape             # [B, S]
index.unsqueeze(-1)     # [B, S, 1]
```

`gather` 根据 token ID 从词表维中取出对应的 logit：

```python
logits = [
    [[2.0, 1.0, 0.0],
     [0.5, 1.5, 0.2]]
]

index = [[1, 0]]
```

得到：

```python
selected_logits = [[1.0, 0.5]]
```

接着：

```python
logsumexp_values = torch.stack([
    torch.logsumexp(lg, dim=-1)
    for lg in logits
])
```

计算每个位置整个词表的归一化项：

```text
log(e^logit_0 + e^logit_1 + ... + e^logit_V)
```

最后：

```python
per_token_logps = selected_logits - logsumexp_values
```

就得到了实际生成 token 的 log probability。

两个分支计算的是同一个东西：

- `float32/float64`：直接使用 `selected_logit - logsumexp`。
- `float16/bfloat16`：逐个 batch 调用 `log_softmax`，再用 `gather` 取出目标 token。

它没有计算 KL，也没有返回普通概率。虽然变量叫 `mini_student_probs`，实际保存的是 `log_probs`。后面用它计算 PPO ratio：

```python
logprobs_diff = new_log_probs - old_log_probs
ratio = torch.exp(logprobs_diff)
```

对应：

```text
ratio = exp(log P_new - log P_old)
      = P_new / P_old
```

其中 float32 分支可以简化为：

```python
selected_logits = logits.gather(
    dim=-1,
    index=index.unsqueeze(-1),
).squeeze(-1)

logsumexp_values = torch.logsumexp(logits, dim=-1)

return selected_logits - logsumexp_values
```

另外要注意：`index` 中的生成 token 必须和 logits 的预测位置严格错位对齐——CausalLM 的 `logits[:, t]` 预测的是 `token[:, t+1]`。当前调用位置存在一位对齐问题。

# 2

```python
mini_completion_mask = mini_attention_mask[:, prompt_ids.shape[-1]:]
```

这行代码从完整序列的 attention mask 中，只截取“学生生成回答”部分的有效 token mask。

```python
mini_completion_mask = mini_attention_mask[
    :, prompt_ids.shape[-1]:
]
```

假设完整序列是：

```text
[prompt | completion | padding]
```

对应：

```python
mini_attention_mask = [
    [1, 1, 1, 1,  1, 1, 1, 0, 0]
     └─prompt──┘   └completion──┘
]
```

如果 prompt 长度为 4：

```python
prompt_ids.shape[-1] == 4
```

切片得到：

```python
mini_completion_mask = [
    [1, 1, 1, 0, 0]
]
```

含义是：

- `1`：有效的 completion token，需要计算奖励和 loss。
- `0`：生成结束后的 padding，不参与奖励和 loss。

它主要用在三个地方。

计算有效 token 的平均奖励：

```python
reward_mean = (
    reward * mini_completion_mask
).sum(dim=1, keepdim=True) / mini_completion_mask.sum(
    dim=1, keepdim=True
)
```

例如：

```python
reward = [-0.2, -0.4, -0.6, -0.9, -0.9]
mask   = [   1,    1,    1,    0,    0]
```

实际平均值只计算前三个：

```text
reward_mean = (-0.2 - 0.4 - 0.6) / 3 = -0.4
```

屏蔽 padding 位置的 advantage：

```python
adv = adv * mini_completion_mask.float()
```

以及计算最终 loss：

```python
token_loss_per_seq = (
    pg_loss_max * micro_completion_mask
).sum(dim=1) / micro_completion_mask.sum(dim=1)
```

因此，这行代码可以理解为：

> 标记学生回答中哪些 token 是真实生成内容，哪些只是为了凑齐 batch 长度而添加的 padding。

这里使用 `prompt_ids.shape[-1]` 只是为了取得固定 prompt 长度。例如 prompt 被 padding 到 512，则相当于：

```python
mini_completion_mask = mini_attention_mask[:, 512:]
```

需要注意，如果 tokenizer 的 `pad_token_id` 和 `eos_token_id` 相同，那么前面通过：

```python
attention_mask = (sequences != pad_token_id).long()
```

构造 mask 时可能连 EOS 也屏蔽掉，需要单独处理。