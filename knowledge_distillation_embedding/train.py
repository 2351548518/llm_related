"""用教师模型的候选相似度分布蒸馏一个较小的 Embedding 模型。

这里蒸馏的不是 token 词表概率，而是同一组候选文档上的排序概率：

    教师分数 = [sim(q, positive), sim(q, negative_1), ...]
    学生分数 = [sim(q, positive), sim(q, negative_1), ...]

两组分数经过带温度 T 的 softmax 后，通过 KL 散度约束学生分布接近教师分布：

    teacher_probs    = softmax(teacher_scores / T)
    student_log_probs = log_softmax(student_scores / T)
    loss = T^2 * KL(teacher_probs || student_probs)

例如教师分数 [0.9, 0.2] 比 [0.6, 0.5] 更明确地偏向正样本；学生不仅学习
“哪一个是正样本”，还会学习教师对候选难度的相对判断。
"""

from transformers import AutoModel, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import KGDataset
from torch import Tensor

def similarity(emb1: torch.Tensor, emb2: torch.Tensor, dim =1):
    """在指定维度上计算余弦相似度。

    在训练中：
        emb1.shape = [batch_size, 1, hidden_size]
        emb2.shape = [batch_size, candidate_num, hidden_size]
    广播后返回 [batch_size, candidate_num]。
    """
    return F.cosine_similarity(emb1, emb2, dim=dim)

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """取每段文本最后一个有效 token 的 hidden state 作为句子向量。

    左 padding 示例（1 表示有效 token）：
        attention_mask = [0, 0, 1, 1, 1]
        最后一个位置一定有效，直接取下标 -1。

    右 padding 示例：
        attention_mask = [1, 1, 1, 0, 0]
        有效长度为 3，因此取下标 3 - 1 = 2。

    Args:
        last_hidden_states: [text_num, seq_len, hidden_size]。
        attention_mask: [text_num, seq_len]。

    Returns:
        [text_num, hidden_size] 的句子向量。
    """
    # 如果 batch 中每条文本的最后一个位置都有效，则 tokenizer 使用的是左 padding。
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        # 对右 padding，attention_mask 的和就是有效 token 数量。
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class KGTrainingArguments(TrainingArguments):
    """在 Hugging Face TrainingArguments 上附加蒸馏温度。"""
    """
    https://zhuanlan.zhihu.com/p/504323465
    温度系数的作用

    温度系数T主要用于调整softmax函数的输出平滑度。在知识蒸馏中，教师模型的softmax输出会除以一个温度系数T，得到soft target，学生模型的softmax输出也会除以同样的T，然后计算交叉熵损失。这种做法的目的是放大类别之间的相似信息，从而让学生模型更好地学习教师模型的泛化能力。

    温度系数的选择

    温度系数T的选择对知识蒸馏的效果有重要影响。一般来说，T的取值通常大于1，这样可以使softmax输出的概率分布更加平滑，减少过度自信的问题。例如，当T=1时，softmax输出的概率差距较大，而当T=20时，概率差距变小，各类别的输出被同等考量。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # T 越大，softmax 分布通常越平滑，学生能看到更多候选之间的相对关系。
        # 当前实验入口把温度固定为 1。
        self.temperature = 1



class KGTrainer(Trainer):
    """自定义 Trainer，只替换默认的监督学习损失为 KL 蒸馏损失。"""
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """计算一个 batch 的候选分布蒸馏损失。

        假设 batch_size=2、每条样本有 1 个负样本、seq_len=512：

        - input_ids 初始形状：[2, 3, 512]
          其中 3 表示 [query, positive, negative]。
        - labels 形状：[2, 2]
          其中 2 表示 [positive_score, negative_score]。
        - student_scores 最终形状：[2, 2]，与 labels 一一对应。
        """
        loss_fct = nn.KLDivLoss(reduction="batchmean") # KL 散度 损失

        # labels 是教师相似度，不需要传给学生模型的 forward。
        labels = inputs.pop("labels")
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[-1]
        
        # 原形状：[batch_size, sample_num, seq_len]
        # sample_num = 1(query) + 1(positive) + negative_num。
        # 模型只接受二维 token 序列，所以把前两个维度展平：
        # [B, S, L] -> [B*S, L]。
        # labels 、input_ids 都是 [B*S, L]。
        inputs = {key: inputs[key].reshape(-1, seq_len) for key in inputs}

        # outputs.last_hidden_state.shape = [B*S, L, H]。
        outputs = model(**inputs)
        # 池化后得到 [B*S, H]，再恢复为 [B, S, H]。
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = embeddings.reshape(batch_size, -1, embeddings.shape[-1])

        # 每条样本的第 0 个向量是 query：[B, 1, H]。
        query_embeddings = embeddings[:, :1]

        # 其余向量依次是 positive 和 negatives：[B, 1+N, H]。
        pos_neg_embeddings = embeddings[:, 1:]

        # 通过广播计算 query 与全部候选的相似度，得到 [B, 1+N]。
        student_scores = similarity(query_embeddings, pos_neg_embeddings, dim=2)
        # 温度缩放后转换为 log probability，满足 KLDivLoss 对 input 的要求。
        student_scores = student_scores / args.temperature
        student_log_probs = torch.log_softmax(student_scores, dim=1)

        # 教师分数使用相同温度转换为普通 probability，作为 KLDivLoss target。
        teacher_scores = labels / args.temperature
        teacher_probs = torch.softmax(teacher_scores, dim=1)
        loss = loss_fct(student_log_probs, teacher_probs)

        # 标准知识蒸馏通常乘 T^2，以抵消温度缩放造成的梯度量级变化。
        loss = loss * (args.temperature**2)
        return (loss, outputs) if return_outputs else loss
        

if __name__ == '__main__':
    # 学生模型负责学习教师生成的候选相似度分布。
    model = AutoModel.from_pretrained("Qwen3-Embedding-0.6B")
    
    # LoRA 只为列出的线性投影层增加低秩可训练参数，原始模型参数保持冻结。
    # r=8 是低秩矩阵的秩，lora_alpha=256 控制 LoRA 更新的缩放强度。
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.SEQ_CLS)
    # 把 LoRA adapter 注入学生模型。
    model = get_peft_model(model, lora_config)
    model.cuda()
    # print_trainable_parameters 会显示可训练参数量及其占总参数的比例。
    print(model.print_trainable_parameters())
    
    # 使用左 padding，保证最后一个位置就是每条文本的最后一个有效 token。
    tokenizer = AutoTokenizer.from_pretrained("Qwen3-Embedding-0.6B", padding_side='left')
    
    
    # 有效 batch size = per_device_train_batch_size * gradient_accumulation_steps
    # 当前单卡配置下为 2 * 4 = 8。
    args = KGTrainingArguments(output_dir='./results_lora_8b_negative_1', 
                            num_train_epochs=2, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=4,
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=10,
                            bf16=True,
                            learning_rate=5e-5,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    # 该 JSON 中每条数据已经包含教师模型生成的 label。
    dataset = KGDataset('train_data/train_negative_num_1_8b.json', tokenizer=tokenizer, max_seq_len=512)
    trainer = KGTrainer(model=model,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    # 初次训练设为 False；如果输出目录中已有合法 checkpoint，可设为 True 续训。
    trainer.train(resume_from_checkpoint=False)
    # 这里只保存 LoRA adapter；完整模型需要再由 merge.py 合并。
    trainer.save_model('./saves_lora_8b_negative_1')
    trainer.save_state()
