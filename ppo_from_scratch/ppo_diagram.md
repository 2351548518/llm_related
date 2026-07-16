# PPO-RLHF 数据流架构图

> 配套代码：`ppo_train.py`。下图按代码实际执行顺序绘制，标注了每个函数、关键张量形状与四个模型的角色。

## 总体流程图（主循环）

```mermaid
flowchart TD
    subgraph 数据["① 数据准备"]
        P["prompt_list<br/>8 条问题"] --> PD["PromptDataset<br/>套 chat 模板"]
        PD --> PDL["prompts_dataloader<br/>batch=rollout_batch_size"]
    end

    subgraph 采样["② generate_samples（rollout）"]
        PDL -->|"每个 prompt ×n_samples_per_prompt"| GEN["Actor.generate<br/>(max_new_tokens=50)"]
        GEN --> SMP["Samples<br/>seqs, attention_mask,<br/>action_mask, num_actions"]
    end

    subgraph 经验["③ generate_experiences（no_grad / detach）"]
        SMP --> ACT["Actor 前向<br/>logits→log_softmax→gather<br/>→ action_log_probs (OLD)"]
        SMP --> REF["Reference 前向<br/>→ ref_action_log_probs"]
        SMP --> CRT["Critic 前向<br/>→ value V(s_t)"]
        SMP --> DEC["batch_decode 成文本"]
        DEC --> RM["Reward Model<br/>→ 标量 r"]

        ACT --> KL["compute_approx_kl<br/>logπ_θ − logπ_ref"]
        REF --> KL
        KL --> RW["compute_rewards<br/>rewards = -kl_ctl·kl<br/>＋ clip(r) 加到最后 token"]
        RM --> RW

        CRT --> GAE["get_advantages_and_returns<br/>GAE 倒序递推<br/>A_t, returns_t"]
        RW --> GAE

        ACT --> EXP["Experience<br/>(全部 detach)"]
        REF --> EXP
        CRT --> EXP
        KL --> EXP
        GAE --> EXP
        RM --> EXP
        RW --> EXP
    end

    subgraph 缓冲["④ 经验池"]
        EXP --> BUF["ExperienceBuffer<br/>limit=100 滑窗"]
        BUF --> DL["DataLoader<br/>+ collate_fn<br/>(batch=micro_train_batch_size)"]
        DL --> BI["BufferItem<br/>(cat 成训练大 batch)"]
    end

    subgraph 训练["⑤ train_step（max_epochs=5 轮重用）"]
        BI --> TACT["Actor 前向(重新)<br/>→ new action_log_probs"]
        TACT --> PL{"compute_policy_loss<br/>ratio=exp(new-old)<br/>clip[0.8,1.2]·A<br/>min(surr1,surr2)"}
        BI -->|"old_log_probs, advantages"| PL
        PL --> PB["policy_loss.backward<br/>optimizer_actor.step"]

        BI --> TCRT["Critic 前向(重新)<br/>→ new values"]
        TCRT --> VL{"compute_value_loss<br/>(values − returns)²<br/>MSE"}
        BI -->|"old_values, returns"| VL
        VL --> VB["value_loss.backward<br/>optimizer_critic.step"]

        PB --> TB[("TensorBoard<br/>policy_loss 曲线")]
        VB --> TB
    end

    BUF -.->|"buffer.clear()<br/>下一批新经验"| PDL
    PB -.->|"参数更新后"| GEN
```

## 张量对齐细节（最易错点）

```mermaid
flowchart LR
    subgraph 对齐["序列对齐：prompt=3, response=2, 共 5 个 token"]
        S["seqs<br/>[p0, p1, p2, a0, a1]"]
        L["logits[:, :-1]<br/>预测 [p1, p2, a0, a1]<br/>长度 4"]
        G["gather(seqs[:,1:])<br/>取出实际下一个 token 的 logp"]
        A["action_log_probs<br/>[:, -num_actions=2]<br/>= [logp(a0), logp(a1)]"]

        S --> L --> G --> A
    end
```

## GAE 递推（倒序）

```mermaid
flowchart RL
    t1["t=1 (最后)<br/>δ₁ = r₁ + γ·0 − V₁<br/>A₁ = δ₁"]
    t0["t=0<br/>δ₀ = r₀ + γ·V₁ − V₀<br/>A₀ = δ₀ + γλ·A₁"]
    R["returns = A + V"]

    t1 --> t0 --> R
```

## 四个模型的角色对照

| 模型 | 输入 | 输出 | 是否更新 | 在流程中的位置 |
|---|---|---|---|---|
| **Actor** | seqs | token 概率分布 | ✅ PPO clip 更新 | ②生成 ③算old_logp ⑤算new_logp |
| **Reference** | seqs | token 概率分布 | ❌ 冻结 | ③算 ref_logp → KL 约束 |
| **Reward Model** | 解码文本 | 标量 r | ❌ 冻结 | ③打结果奖励分 |
| **Critic** | seqs | V(s_t) per token | ✅ MSE 更新 | ③算value ⑤算new_value |

## 关键设计点

1. **on-policy 校正**：`old_*`（采样时 detach 的快照）做分母，`new_*`（`train_step` 里重新前向）做分子 → PPO 的 `ratio`。
2. **经验重用**：一批 rollout 训 `max_epochs=5` 轮，靠 clip 防止策略离旧策略过远。
3. **KL 罚 vs KL 损失**：本代码把 KL 当作**逐 token 奖励惩罚**（`compute_rewards`），而非额外的损失项 —— 这是 InstructGPT 式方案。
4. **左填充**：`padding_side='left'` 保证 prompt 末尾紧贴、pad 在开头，生成位置对齐。
5. **结果奖励只给末 token**：`reward_clip` 只加到 response 最后一个有效 token，符合"回复结束才结算"的语义。
```
