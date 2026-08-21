"""知识蒸馏损失函数。

本文件统一使用下面的记号：

* ``p``：教师模型在词表上的概率分布；
* ``q``：学生模型在词表上的概率分布；
* 输入 logits 的形状均为 ``[batch_size, seq_len, vocab_size]``；
* 函数先对词表维求和，得到每个位置的 KL，再屏蔽 padding。

形状示例：batch=2、序列长度=3、词表大小=4 时，
``logits.shape == teacher_logits.shape == [2, 3, 4]``，
``target.shape == [2, 3]``。若 ``target[0] == [10, 11, -100]``，
并传入 ``padding_id=-100``，则第一个样本最后一个位置不计入损失。
"""

import torch

# 计算前向kl散度  KL(p‖q) = Σ_x p(x)·log(p(x)/q(x))，p=教师，q=学生
def compute_fkl(
        logits,              # 学生 logits: [batch, seq_len, vocab]
        teacher_logits,      # 教师 logits: [batch, seq_len, vocab]
        target,              # 真实 token id: [batch, seq_len]，用于定位 padding
        padding_id,          # padding token 的 id
        reduction="sum",     # 聚合方式: "sum"=按样本求和, "mean"=按真实 token 平均
        temp = 1.0,          # 温度 T，T>1 让分布变平滑，暴露"暗知识"
    ):
        """计算前向 KL：``KL(p || q)``。

        例：某个位置上教师分布为 ``p=[0.8, 0.2]``，学生分布为
        ``q=[0.6, 0.4]``，该位置的损失为
        ``0.8*log(0.8/0.6) + 0.2*log(0.2/0.4) ≈ 0.0915``。

        返回值：``sum``/``mean`` 返回 ``[batch]``；其他 reduction 值保留
        ``[batch, seq_len]`` 的逐 token KL。RL 脚本目前利用后一行为每个 token
        构造奖励。
        """
        # —— 温度缩放：除以 T 让 softmax 平滑。严格蒸馏损失还应乘 T² 补偿梯度，
        #    这里省略，相当于整体缩放 loss，不影响优化方向 ——
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        # 学生 q 的对数概率: log_softmax 数值稳定，避免 log(0)
        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        # 教师 p 的概率
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        # 教师 p 的对数概率
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        # 逐元素: p*(log p - log q) = p*log(p/q)，对 vocab 维求和后即前向 KL
        kl = (teacher_probs * (teacher_log_probs - log_probs))
        # 在 vocab 维(dim=-1)求和 → 每个 token 一个 KL 值: [batch, seq_len]
        kl = kl.sum(-1)
        # padding 掩码: target 中等于 padding_id 的位置为 True
        pad_mask = target.eq(padding_id)
        # 把 padding 位置(非真实 token)的 KL 清零; masked_fill_ 下划线=原地操作省内存
        kl = kl.masked_fill_(pad_mask, 0.0)
        if reduction == "sum":
            # 在 seq 维(dim=1)求和 → 每个样本一个标量: [batch]
            kl = kl.sum(dim=1)
        elif reduction == "mean":
            # 除以非 padding 的真实 token 数取平均，避免长样本被过度惩罚
            kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1)

        return kl

        
# 计算反向kl散度
def compute_rkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum", 
        temp = 1.0
    ):
        """计算反向 KL：``KL(q || p)``，其中 q 是学生、p 是教师。

        每一项由学生概率 q 加权：``sum(q * (log(q) - log(p)))``。
        例如学生把 0.4 的概率放在教师只有 0.01 概率的 token 上时，
        该 token 会受到较大的惩罚。
        """
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)

        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        
        kl = (probs * (log_probs - teacher_log_probs))
        kl = kl.sum(-1)
        
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill(pad_mask, 0.0)
        if reduction == "sum":
            kl = kl.sum(dim=1)
        elif reduction == "mean":
            kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1)
        
        return kl

# 计算偏向前kl散度
def compute_skewed_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id, 
        reduction="sum", 
        temp = 1.0,
        skew_lambda = 0.1
    ):
        """计算偏向前向 KL：``KL(p || lambda*p + (1-lambda)*q)``。

        例如 ``skew_lambda=0.1`` 时，混合分布中 10% 来自教师、90% 来自学生。
        """
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * probs
        mixed_log_probs = torch.log(mixed_probs)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - mixed_log_probs))
        kl = kl.sum(-1)
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        if reduction == "sum":
            kl = kl.sum(dim=1)
        elif reduction == "mean":
            kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1)

            
        return kl
# 计算偏向反kl散度    
def compute_skewed_rkl(
    logits, 
    teacher_logits, 
    target,
    padding_id,
    reduction="sum", 
    temp = 1.0,
    skew_lambda = 0.1
):
    """计算偏向反向 KL：``KL(q || (1-lambda)*p + lambda*q)``。

    例如 ``skew_lambda=0.1`` 时，参考分布由 90% 教师和 10% 学生组成。
    """
    logits = logits / temp
    teacher_logits = teacher_logits / temp
    
    probs = torch.softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * probs
    mixed_log_probs = torch.log(mixed_probs)
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    kl = (probs * (log_probs - mixed_log_probs))
    kl = kl.sum(-1)
    
    pad_mask = target.eq(padding_id)
    kl = kl.masked_fill_(pad_mask, 0.0)
    if reduction == "sum":
        
        kl = kl.sum(dim=1)
    elif reduction == "mean":
        kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1)


    return kl
