# 代码 在哪里 体现出来了 On Policy 同策略 ？

核心判断标准是：

> 训练用的 completion 是否由“当前学生模型”采样产生。

这两份代码的 On-Policy 都体现在：

```text
数据集只提供 prompt
        ↓
当前学生模型生成 completion
        ↓
教师在学生生成的 completion 上提供 KL 信号
        ↓
更新这个学生模型
        ↓
更新后的学生重新生成下一批 completion
```

On-Policy 不是因为使用了教师模型，而是因为训练状态和回答来自当前学生策略。

## 1. 数据集只提供 prompt

[dataset.py:126](/data2/home/jiapeng2/code/LLM/llm_related/knowledge_distillation_llm/dataset.py:126) 中的 `OnPolicyDataset` 只读取：

```python
instruction_text = line['instruction']
input_text = line['input']
```

没有读取：

```python
line["output"]
```

最终也只返回：

```python
return {
    'input_ids': ...,
    'attention_mask': ...
}
```

因此训练代码没有标准答案 completion，必须由学生自己生成。

## 2. 直接优化 KL 的版本

关键位置是 [on_policy_distillation_train.py:65](/data2/home/jiapeng2/code/LLM/llm_related/knowledge_distillation_llm/on_policy_distillation_train.py:65)：

```python
sequences = self.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    do_sample=True,
)
```

这里的：

```python
self.model
```

就是当前正在训练的学生模型。

在每次 `compute_loss()` 中，都会重新调用它：

```python
sequences = self.generate_sequences(prompt_ids, prompt_mask)
```

然后把学生生成的完整序列同时交给学生和教师：

```python
logits = model(sequences, ...).logits
teacher_outputs = self.teacher_model(sequences, ...)
```

也就是说，教师并没有生成训练回答，只是在学生自己生成的回答上提供分布监督。

最后：

```python
kl = compute_rkl(
    logits,
    teacher_logits,
    completion_ids,
    ...
)

loss = kl.mean()
```

更新学生。

举例：

```text
prompt：1+1 等于多少？
当前学生生成：3
教师在“学生生成了 3”的状态上提供分布
学生通过 KL 更新

下一轮更新后的学生可能生成：2
再在新的学生分布上训练
```

这就是 On-Policy：学生在自己真正会访问到的状态上学习。

## 3. RL/PPO 版本

RL 版本同样先使用当前学生 rollout：

[on_policy_distillation_train_rl.py:176](/data2/home/jiapeng2/code/LLM/llm_related/knowledge_distillation_llm/on_policy_distillation_train_rl.py:176)

```python
prompt_ids = inputs["input_ids"].to(self.model.device)
prompt_mask = inputs["attention_mask"].to(self.model.device)

sequences = self.generate_sequences(
    prompt_ids,
    prompt_mask,
)
```

而 `generate_sequences()` 内部仍然是：

```python
sequences = self.model.generate(...)
```

随后，从学生生成的 `sequences` 中提取 completion：

```python
mini_input_ids = sequences[mini_idx:...]
mini_completion_ids = mini_input_ids[:, prompt_ids.shape[-1]:]
```

教师在这些学生轨迹上计算 logits：

```python
mini_teacher_outputs = self.teacher_model(
    mini_input_ids,
    attention_mask=mini_attention_mask,
)
```

然后用教师与学生的反向 KL 构造奖励：

```python
kl = compute_rkl(...)

reward = -kl
adv = reward - reward_mean
```

最后根据学生实际生成 token 的 log probability 做 PPO 更新：

```python
ratio = torch.exp(
    new_log_probs - old_log_probs
)

pg_losses = -adv * ratio
loss.backward()
self.optimizer.step()
```

所以 RL 版本的 On-Policy 链路是：

```text
当前学生 rollout
→ 教师计算 KL
→ KL 转成 reward/advantage
→ PPO 更新当前学生
```

## 与 Off-Policy 的区别

[train.py](/data2/home/jiapeng2/code/LLM/llm_related/knowledge_distillation_llm/train.py:57) 没有调用：

```python
student.generate()
```

它直接使用数据集中的固定 `output`：

```text
prompt + 人工标准答案
```

学生和教师都在固定答案上计算 logits，这属于 Off-Policy 蒸馏。

需要注意，RL 版本先生成整个 batch，然后针对该 batch 做若干次 mini-batch 更新。第一次更新后，剩余 rollout 相对于新模型已经稍微“过时”，所以代码通过：

```python
ratio = new_prob / old_prob
```

做重要性采样修正。这是 PPO 常见的近似 On-Policy 实现。

# completion是啥？

`completion` 就是模型根据 prompt 生成的“回答部分”。

例如：

```text
prompt：
用户：1+1 等于多少？
助手：

completion：
2
```

拼在一起就是完整序列：

```text
sequence = prompt + completion
```

在代码中：

```python
sequences = self.model.generate(
    input_ids=prompt_ids,
    ...
)
```

`sequences` 包含 prompt 和学生生成的回答：

```text
[prompt token | completion token]
```

然后通过下面的切片只取回答：

```python
completion_ids = sequences[:, prompt_ids.shape[-1]:]
```

假设：

```python
prompt_ids = [10, 11, 12, 13]
sequences  = [10, 11, 12, 13, 20, 21, 22]
```

那么：

```python
completion_ids = [20, 21, 22]
```

其中：

```text
[10, 11, 12, 13] = prompt
[20, 21, 22]     = completion
```

在中文里可以理解为：

- `prompt`：输入、问题、上文。
- `completion`：模型续写、生成结果、回答。
- `sequence`：prompt 与 completion 拼接后的完整序列。

这个项目里的 On-Policy，指的就是 completion 不是来自数据集标准答案，而是当前学生模型自己生成的。