"""离线（off-policy）白盒知识蒸馏训练入口。

每个 batch 同时送入学生和教师模型。教师只提供软分布且不计算梯度；学生通过
前向 KL 学习教师分布，并可选择与标准答案的交叉熵损失混合。

流程：``data.json -> SFTDataset -> 学生/教师 logits -> KL (+ CE) -> 更新学生 LoRA``。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset
from utils import compute_fkl, compute_rkl, compute_skewed_fkl, compute_skewed_rkl


class KGTrainer(Trainer):
    """在 HuggingFace ``Trainer`` 上增加教师模型和蒸馏损失。"""
    
    def __init__(
        self,
        model = None,
        teacher_model = None,
        if_use_entropy = False,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        optimizers = (None, None), 
        preprocess_logits_for_metrics = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算一个 batch 的蒸馏损失。

        形状示例（batch=2、seq_len=512、vocab=151936）：

        * ``input_ids`` / ``labels``：``[2, 512]``；
        * 学生和教师 logits：``[2, 512, 151936]``；
        * ``compute_fkl(..., reduction='sum')``：``[2]``；
        * 最后的 ``.mean()`` 将两个样本合成一个标量。

        注意：CausalLM 的第 t 个 logits 预测第 t+1 个 token。当前实现直接使用
        未移位的 labels 选择回答区间；精确对齐时还需要把 logits、labels 和
        mask 错开一位。
        """

        outputs = model(**inputs)

        # 教师模型 不参与 模型的更新 梯度的计算
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        # HuggingFace CausalLM 会在内部完成 next-token shift，所以 outputs.loss
        # 已经是“当前位置 logits 预测下一个 token”的标准交叉熵。
        loss = outputs.loss # 学生的 模型输出 的交叉熵损失（输出值 和 真实值 的比较）
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
        # 模型输出的 维度 不同，也就是 vocab_size 不同，可能是因为学生模型的词表比教师模型的词表小
        # [batch, seq_len, vocab_size] 
        if logits.shape[-1] != teacher_logits.shape[-1]:
            # gap = teacher_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad_logits], dim=-1)
            
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        labels = inputs['labels']
        # 这里的 -100 不是 tokenizer.pad_token_id，而是 labels 中的 ignore_index：
        # prompt 和 padding 标签都被设为 -100，因此这些位置不参与 KL。
        # 例：labels=[-100, -100, 42, 43, -100] 时，-100 位置的 KL 被屏蔽。
        # compute_fkl 默认沿序列求和，而 CE 按有效 token 平均，所以 0.5/0.5
        # 只是代码系数，不表示两部分的实际梯度贡献一定各占一半。
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0).mean()
        
        if self.if_use_entropy:
            # 同时学习教师软标签（KL）和数据集硬标签（CE）。
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            # 纯蒸馏：只要求学生输出分布接近教师。
            loss_total = kl
        
        return (loss_total, outputs) if return_outputs else loss_total
        

if __name__ == '__main__':
    
    # 学生模型：参数量较小，是实际被优化、最终保存的模型。
    model = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-Instruct")
    
    # LoRA 配置，指定要微调的模块和参数
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=256,  
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, 
        task_type=TaskType.CAUSAL_LM
    )

    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.cuda()
    print(model.print_trainable_parameters())
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct")
    
    # 教师模型：参数量较大，并加载预先训练好的 LoRA 权重作为监督来源。
    teacher_model = AutoModelForCausalLM.from_pretrained("Qwen2.5-7B-Instruct")
    # 是否加载lora模型
    lora_path = 'qwen2.5_7b/lora/sft'
    teacher_model = PeftModel.from_pretrained(teacher_model, lora_path)
    teacher_model.cuda()
    teacher_model.eval() # 教师模型 是 不参与训练的，所以设置为 eval 模式，避免 dropout 等影响
    
    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=10, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=16,
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=10,
                            bf16=True,
                            learning_rate=0.0005,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    dataset = SFTDataset('data.json', tokenizer=tokenizer, max_seq_len=512)
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        if_use_entropy = True,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
