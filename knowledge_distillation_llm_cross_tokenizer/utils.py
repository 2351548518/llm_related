"""几种 KL 散度的独立实验实现。

张量形状约定：

* ``logits``、``teacher_logits``: ``[batch_size, seq_len, vocab_size]``；
* ``target``: ``[batch_size, seq_len]``，仅用于找出 padding 位置；
* 返回值：当 ``reduction="sum"`` 时为标量，否则为 ``[batch_size, seq_len]``。

例如，batch 中有 2 条、每条 4 个位置、词表大小为 100，则 logits 的形状为
``[2, 4, 100]``，沿最后一维求和后，每个 token 位置得到一个 KL 值。

注意：这些函数当前没有被 ``train.py`` 的训练流程调用；实际训练使用的是
``ULDLoss.compute_kl_loss``。
"""

import torch


# 计算前向 KL 散度 KL(teacher || student)。
def compute_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum",
        temp = 1.0, 
        
    ):
        """让学生分布覆盖教师认为可能的 token。

        单个位置上的公式为::

            sum_v p_teacher(v) * (log p_teacher(v) - log p_student(v))

        例：教师分布为 ``[0.8, 0.2]``，学生分布为 ``[0.5, 0.5]``，该值
        会大于 0；两个分布完全相同时为 0。
        """

        # 温度越大，softmax 后的分布通常越平滑；两个模型必须使用同一温度。
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        # 在 float32 中计算概率和 log 概率，避免 bf16/fp16 下的数值精度损失。
        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - log_probs)) 
        # 对 vocab 维求和：[B, L, V] -> [B, L]。
        kl = kl.sum(-1)
        if reduction == "sum":
            # padding 位置不应参与损失；例如 target=[5, 6, 0, 0] 且
            # padding_id=0 时，后两个位置的 KL 会被清零。
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()

        return kl


# 计算反向 KL 散度 KL(student || teacher)。
def compute_rkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum", 
        temp = 1.0
    ):
        """惩罚学生把概率放到教师认为不可能的 token 上。

        与前向 KL 的主要差别是期望取在学生分布上。直观上，反向 KL 往往更
        偏向教师分布的高概率峰值，而前向 KL 更强调覆盖教师的全部概率质量。
        """

        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (probs * (log_probs - teacher_log_probs))
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()
        return kl


# 计算偏向前向 KL（skewed forward KL）。
def compute_skewed_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id, 
        reduction="sum", 
        temp = 1.0,
        skew_lambda = 0.1
    ):
        """用教师/学生混合分布代替普通前向 KL 中的学生分布。

        当前混合方式为::

            mixed = skew_lambda * teacher + (1 - skew_lambda) * student
            loss  = KL(teacher || mixed)

        例：``skew_lambda=0.1`` 时，mixed 中教师占 10%，学生占 90%。因为
        mixed 主动包含一部分教师概率，极端概率差异对损失的影响会更温和。
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
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()

            
        return kl


# 计算偏向反向 KL（skewed reverse KL）。
def compute_skewed_rkl(
    logits, 
    teacher_logits, 
    target,
    padding_id,
    reduction="sum", 
    temp = 1.0,
    skew_lambda = 0.1
):
    """计算 ``KL(student || mixed)``。

    当前混合方式为::

        mixed = (1 - skew_lambda) * teacher + skew_lambda * student

    例：``skew_lambda=0.1`` 时，mixed 中教师占 90%，学生占 10%。
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
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()


    return kl
