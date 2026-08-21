# On-Policy Distillation

# On Policy Distillation

## 🎯 On Policy vs Off Policy

### Off Policy（最传统、经典的知识蒸馏）

学生模型的训练数据来自于真实数据或者由教师模型生成，即：

> 教师模型生成数据，学生模型学习教师模型的分布。

**优势：** 数据可复用，训练资源占用较低（教师模型数据可提前生成，无需在训练过程中生成）。

**劣势：** 只是一味的学习教师模型的分布，当教师模型产生的数据多样性或者质量较低，会导致学生模型泛化性能很差（推理与训练不一致，训练在教师模型的分布上学习，推理时在自己分布上生成）。

### On Policy

学生模型的训练数据由学生模型自己生成，即：

> 学生模型自己与环境交互 → 生成数据 → 教师模型纠正学生模型分布。

**优势：** 归根结底学生模型是在自己的分布上学习，不再是一味地模仿，模型不仅能接受到正确的反馈，也能接收负反馈，这也决定了其泛化性能会更好（训练和推理都在自己的分布上）。

**劣势：** 如果学生模型自身产生了一些质量比较低的样本，会导致难以优化。

---

## 🎯 On Policy Distillation

我们很容易发现，on policy distillation 和 RL 类似，都是由当前需要优化的模型进行 rollout 生成数据，然后根据外界的反馈信号进行优化。

on policy distillation 的反馈信号来自教师模型（KL 散度），RL 的反馈信号来自奖励模型，那么很自然的就可以想到，可以将教师模型与学生模型的 KL 散度应用到训练优化中（KL），并且 KL 散度可以提供一种更细粒度的奖励信号（token 级）。

```python
reward = -kl
```

Token 的 KL 散度越小，说明教师模型与学生模型在当前 token 的分布越相似，就认为学生模型生成的 token 是好的，其奖励越高；反之，则奖励越小。

因为 KL 散度非负，所以 reward 都是负的，这不是很符合直觉，于是我们减去一个 baseline：

```python
adv = reward - reward_mean
```

`reward_mean` 是样本中所有 token 奖励的均值。

到这里，就可以使用策略梯度算法进行优化了：

```python
logprobs_diff = student_probs - old_student_probs
ratio = torch.exp(logprobs_diff)
pg_losses = -adv * ratio
pg_losses2 = -adv * torch.clamp(
    ratio,
    1.0 - self.args.cliprange,
    1.0 + self.args.cliprange,
)
pg_loss_max = torch.max(pg_losses, pg_losses2)
```

> **PS：** 这里也不一定使用 token 粒度的奖励，将 token 粒度的 KL 散度聚合成句子粒度的奖励也可以，并且重要性权重也可以采用句子粒度。

所以对于 on policy distillation，可以有两种做法进行优化：

- 将 KL 散度作为直接目标进行优化（KL 散度直接作为损失）；
- 将 KL 散度作为奖励信号，使用策略梯度算法进行优化。
