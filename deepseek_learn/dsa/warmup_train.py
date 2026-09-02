"""阶段一：Dense Warm-up，只训练 Lightning Indexer。

主模型参数被冻结，完整主注意力分布充当“教师”，Indexer 分布充当“学生”。
训练目标是让 Indexer 学会给主注意力认为重要的 Token 更高分；语言模型 CE Loss
虽然会由模型计算出来，但这个阶段不会把它加入最终 Loss。
"""

from model import Qwen2ForCausalLM
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator
import torch.nn as nn
import torch

from dataset import SFTDataset
import torch.nn.functional as F




class DSATrainer(Trainer):
    """使用完整主注意力分布监督 Indexer 的 Trainer。"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算所有 Decoder Layer 的平均 Indexer KL Loss。

        形状示例：
            raw_attn_weights:    [B, 14, L, L]，主注意力每个 Head 的 logits
            indexer_attn_scores: [B, 1,  L, L]，Indexer 的单头检索 logits

        先把主注意力在 14 个 Head 上聚合成 [B, 1, L, L]，再与 Indexer 对齐。
        """

        # output_attentions=True 让模型返回每层 DSA 的三个中间结果。
        outputs = model(**inputs, output_attentions=True)
        all_attentions = outputs.attentions

        # 使用与模型 Loss 相同的设备创建标量，避免 CPU/GPU 混用。
        attention_kl_loss = torch.tensor(0.0, device=outputs.loss.device)
        
        for attention in all_attentions:
            # topk_indices 在 Dense Warm-up 中不会参与 Loss，但模型仍统一返回它。
            topk_indices, raw_attn_weights, indexer_attn_scores = attention

            # 把主注意力 logits 转成每个 Head 上的概率分布。
            raw_attn_weights = F.softmax(raw_attn_weights, dim=-1)
            
            # Head 维度求和：[B, num_heads, L, L] -> [B, 1, L, L]。
            # 例：某位置在两个 Head 上的概率分别为 [0.8, 0.2] 和 [0.4, 0.6]，
            # 求和后为 [1.2, 0.8]。
            raw_attn_weights = raw_attn_weights.sum(1, keepdim=True)
            
            # 沿 Key 维度做 L1 归一化；上例 [1.2, 0.8] -> [0.6, 0.4]。
            raw_attn_weights = raw_attn_weights / torch.norm(raw_attn_weights, dim=-1, p=1, keepdim=True)
           
            # Indexer logits -> Indexer 概率，形状保持 [B, 1, L, L]。
            indexer_attn_scores = F.softmax(indexer_attn_scores, dim=-1)

            # 防止 log(0) 产生负无穷或 NaN。
            indexer_attn_scores = torch.clamp(indexer_attn_scores, min=1e-8)

            # F.kl_div(log(Q), P) 计算 D_KL(P || Q)：
            # P 是 detach 后的主注意力目标，Q 是需要学习的 Indexer 分布。
            kl_loss = F.kl_div(indexer_attn_scores.log(), raw_attn_weights.detach())
         
            attention_kl_loss += kl_loss
        
        # 对所有 Decoder Layer 的 KL Loss 求平均。
        loss = attention_kl_loss / len(all_attentions)
        return (loss, outputs) if return_outputs else loss
            
        
        


if __name__ == '__main__':
    import os

    model = Qwen2ForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct")
    
    # 阶段一只训练参数名包含 indexer 的 wk 和 weights_proj；Qwen 主模型全部冻结。
    for name, param in model.named_parameters():
     
        if 'indexer' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"可训练参数数量: {trainable_params:,}")
    print(f"总参数数量: {total_params:,}")
    
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct")
    
    args = TrainingArguments(output_dir='./step1', 
                            # 总共执行 500 个优化 Step。
                            max_steps=500, 
                            do_train=True, 
                            per_device_train_batch_size=4,
                            # 累积 4 个小批次后更新一次，等效单卡 batch_size=4*4=16。
                            gradient_accumulation_steps=4,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_strategy='steps',
                            save_steps=250,
                            save_total_limit=3,
                            bf16=True,
                            # 仅训练小型 Indexer，因此预热阶段使用较大的学习率。
                            learning_rate=0.001,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    # 所有样本固定为 2048 Token；Prompt 和 Padding 的 labels 都是 -100。
    dataset = SFTDataset('warmup_data.jsonl', tokenizer=tokenizer, max_seq_len=2048)
    trainer = DSATrainer(model=model,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    trainer.train(resume_from_checkpoint=False)
    # step1_model 包含原始 Qwen 权重和已经预热的 Indexer 权重。
    trainer.save_model('./step1_model')
    trainer.save_state()
