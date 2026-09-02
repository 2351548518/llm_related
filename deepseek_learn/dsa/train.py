"""阶段二：Sparse Training，联合语言建模 Loss 与 Top-K Indexer KL Loss。

与 Dense Warm-up 的区别是：

1. 主模型不再冻结，交叉熵 Loss 负责保持/提升文本生成能力；
2. KL Loss 只比较 Indexer 选中的 Top-K 位置，不再比较完整序列分布；
3. 总损失为 ``ce_loss + attention_kl_loss``。
"""

from model import Qwen2ForCausalLM
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator
import torch.nn as nn
import torch

from dataset import SFTDataset
import torch.nn.functional as F




class DSATrainer(Trainer):
    """执行 CE + Top-K KL 联合训练的 Trainer。"""
    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算语言模型交叉熵和所有层的平均 Top-K KL Loss。"""

        # labels 让模型产生 CE Loss；output_attentions=True 提供 KL Loss 所需的中间结果。
        outputs = model(**inputs, output_attentions=True)
        all_attentions = outputs.attentions
        ce_loss = outputs.loss

        # KL Loss 累加器与 ce_loss 位于同一设备。
        attention_kl_loss = torch.tensor(0.0, device=outputs.loss.device)
        
        for attention in all_attentions:
        
            topk_indices, raw_attn_weights, indexer_attn_scores = attention

            # raw_attn_weights 有 num_heads 个 Head，而 topk_indices 只有一个 Indexer Head。
            # expand 只扩展视图，使每个注意力 Head 都在相同的 Top-K 位置 gather。
            # 例：topk_indices=[1, 4, 7]，则每个 Head 都只取位置 1、4、7。
            raw_attn_weights_topk = torch.gather(raw_attn_weights, -1, topk_indices.expand(-1, raw_attn_weights.shape[1], -1, -1))

            # 只在 Top-K 维度上做 Softmax，而不是在完整 key_len 上做 Softmax。
            # 形状：[B, num_heads, query_len, K]。
            raw_attn_weights_topk = F.softmax(raw_attn_weights_topk, dim=-1)
  
            # 聚合所有主注意力 Head：[B, num_heads, query_len, K] -> [B, 1, query_len, K]。
            raw_attn_weights_topk = raw_attn_weights_topk.sum(1, keepdim=True)
 
            # 沿 Top-K 维度做 L1 归一化，得到教师目标分布 p_{t,S_t}。
            raw_attn_weights_topk = raw_attn_weights_topk / torch.norm(raw_attn_weights_topk, dim=-1, p=1, keepdim=True)

            # 从 Indexer 的完整分数中取出相同 Top-K 位置。
            # [B, 1, query_len, key_len] -> [B, 1, query_len, K]。
            indexer_attn_scores_topk = torch.gather(indexer_attn_scores, -1, topk_indices)

            # 在 K 个候选位置上得到 Indexer 概率分布。
            indexer_attn_scores_topk = F.softmax(indexer_attn_scores_topk, dim=-1)
            indexer_attn_scores_topk = torch.clamp(indexer_attn_scores_topk, min=1e-8)

            # 主注意力分布 detach 后仅作为目标，避免 KL Loss 反向修改教师目标本身。
            kl_loss = F.kl_div(indexer_attn_scores_topk.log(), raw_attn_weights_topk.detach())
         
            attention_kl_loss = attention_kl_loss + kl_loss
        
        # 对所有 Decoder Layer 的 KL Loss 求平均。
        attention_kl_loss = attention_kl_loss / len(all_attentions)
        
        # 联合目标：CE 训练文本生成，KL 训练 Indexer 的候选检索能力。
        loss = ce_loss + attention_kl_loss
        
        return (loss, outputs) if return_outputs else loss
    
        
        


if __name__ == '__main__':
    import os

    # 从阶段一保存的模型开始；此时 Indexer 已经学会近似完整主注意力分布。
    model = Qwen2ForCausalLM.from_pretrained("step1_model")


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"可训练参数数量: {trainable_params:,}")
    print(f"总参数数量: {total_params:,}")
    
    
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct")
    
    args = TrainingArguments(output_dir='./step2', 
                            max_steps=2000, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            # 等效单卡 batch_size=2*4=8。
                            gradient_accumulation_steps=4,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_strategy='steps',
                            save_steps=250,
                            save_total_limit=3,
                            bf16=True,
                            # 联合训练整个模型时使用比 Indexer 预热阶段更小的学习率。
                            learning_rate=0.000005,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    dataset = SFTDataset('warmup_data.jsonl', tokenizer=tokenizer, max_seq_len=2048)
    trainer = DSATrainer(model=model,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    # True 表示从 ./step2 中最近的 checkpoint 恢复；首次执行且目录中没有 checkpoint 时应改为 False。
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model('./step2_model')
    trainer.save_state()
