## 知识蒸馏方法

1. 大模型生成一批数据 ， 对小模型 微调
2. 对其 大参数模型 和 小参数 模型 的 概率 分布，要求 大模型 和 小模型 的 词表 是一样的(tokenizer)

## 知识蒸馏 和 微调

标签不一样：
    知识蒸馏 标签 来自于 大参数 模型
    而 微调 标签 来自于 手动标注 或者 大模型标注的 标签
对齐的目标 不一样：
    知识蒸馏 是 对齐 分布
    微调 是 对齐 标签 概率

损失函数：
    知识蒸馏 一般是 KLDivLoss
    微调 是 CE

## 知识蒸馏

```
        """
        计算学生模型的 student_log_probs
        """
        # 通过广播计算 query 与全部候选的相似度，得到 [B, 1+N]。
        student_scores = similarity(query_embeddings, pos_neg_embeddings, dim=2)
        # 温度缩放后转换为 log probability，满足 KLDivLoss 对 input 的要求。
        student_scores = student_scores / args.temperature
        student_log_probs = torch.log_softmax(student_scores, dim=1)

        """
        教师模型的 teacher_probs
        """
        # 教师分数使用相同温度转换为普通 probability，作为 KLDivLoss target。
        teacher_scores = labels / args.temperature
        teacher_probs = torch.softmax(teacher_scores, dim=1)
        loss = loss_fct(student_log_probs, teacher_probs)

        # 标准知识蒸馏通常乘 T^2，以抵消温度缩放造成的梯度量级变化。
        loss = loss * (args.temperature**2)
```

## 微调

比对的 是 laebl

