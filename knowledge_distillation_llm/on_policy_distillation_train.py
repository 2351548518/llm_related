"""直接优化反向 KL 的 on-policy 蒸馏。

与 ``train.py`` 使用固定标准答案不同，本脚本先让学生根据 prompt 生成回答，
再让教师和学生对这些“学生实际会访问到的状态”打分，最后最小化反向 KL。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset, OnPolicyDataset
from utils import compute_rkl



class KGTrainer(Trainer):
    """在学生实时生成的 completion 上优化 ``KL(student || teacher)``。"""
    
    def __init__(
        self,
        model = None,
        teacher_model = None,
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

        # 教师模型
        self.teacher_model = teacher_model

    
    # 学生模型 rollout 采样
    @torch.no_grad()
    def generate_sequences(self, input_ids, attention_mask):
        """
        用当前学生策略采样回答，生成过程本身不保留计算图。

        例：输入固定为 512 个位置、模型生成 20 个 token，则返回的
        ``sequences`` 形状约为 ``[batch, 532]``，前 512 位仍是 prompt。
        """

        self.model.eval()
        sequences = self.model.generate(input_ids=input_ids, 
                                      attention_mask=attention_mask,
                                      max_length=1024,
                                      do_sample=True,
                                      temperature=1.0,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id
                                      )
        
        
        
        self.model.train()
        return sequences
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """完成一次“学生生成 -> 教师打分 -> 反向 KL”的训练步骤。"""
        
        prompt_ids = inputs["input_ids"].to(self.model.device)
        prompt_mask = inputs["attention_mask"].to(self.model.device)
        sequences = self.generate_sequences(prompt_ids, prompt_mask) # 学生模型采样

        attention_mask = (sequences != self.tokenizer.pad_token_id).long()
        # 只保留 completion 区间。假设 prompt_len=512、总长度=532，切片后长度为20。
        # 注意：CausalLM 的 logits[t] 预测 token[t+1]。当前切片和 completion_ids
        # 使用相同起点，会错开一个 token；严格实现应使用前一位置的 logits 对齐。
        logits = model(sequences, attention_mask=attention_mask).logits[:, prompt_ids.shape[-1]:] # 只取生成的 部分
        
        loss = None

        with torch.no_grad():
            teacher_outputs = self.teacher_model(sequences, attention_mask=attention_mask)
        teacher_logits = teacher_outputs.logits[:, prompt_ids.shape[-1]:] # 只取生成的 部分
        
        """
        Qwen 4B 模型 和 小模型 词表有一些细微差别
        """
        if logits.shape[-1] != teacher_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]] # 直接进行截断
        
        # completion_ids 是学生自己采样的回答，prompt 部分不参加本次 KL。
        completion_ids = sequences[:, prompt_ids.shape[-1]:]
        # 计算反向kl散度
        kl = compute_rkl(logits, teacher_logits,  completion_ids, padding_id=self.tokenizer.pad_token_id, reduction="mean")

        # 求平均 作为损失
        loss = kl.mean()
        
        return loss
        

if __name__ == '__main__':
    
    model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.cuda()
    print(model.print_trainable_parameters())
    

    teacher_model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    model.cuda()
    teacher_model.cuda()
    teacher_model.eval()
    
    
    train_dataset = OnPolicyDataset('data.json', tokenizer)
    
    
    
    args = TrainingArguments(output_dir='./outputs', 
                            num_train_epochs=1, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=4,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        args=args, 
                        train_dataset=train_dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
