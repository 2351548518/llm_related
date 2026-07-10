from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from dataset import DPODataset, DPODataCollator
from train import LLM, Config


def logits_to_probs(logits, labels):
    # logits: (batch_size, seq_len, vocab_size)，labels: (batch_size, seq_len)
    # 返回每个位置上"真实下一个 token"的 log 概率 (batch_size, seq_len)
    # 命名叫 probs，语义其实是 log-prob（对数概率）
    # 配合 DPODataCollator 已做的错位(input_ids[:-1], labels[1:])，labels[i] 正是 logits[i] 要预测的下一个 token
    log_probs = F.log_softmax(logits, dim=2)
    # gather 出每个位置对应 label 的 log 概率
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs # (batch_size, seq_len) 真实 token 的 log 概率

def mask_logits(logits, labels):
    # 入参 logits 实为 logits_to_probs 的输出 (batch_size, seq_len)：每位置的 log p(label)
    # labels (batch_size, seq_len) 中，0 表示需忽略的位置（prompt 与 padding 都填 0，见 DPODataCollator）
    # 本函数：对每条序列，把回答部分(label != 0)的 log 概率求和 -> 序列级 log p(answer|prompt)
    # 返回 list（每条序列一个标量），供 dpo_loss 用切片 + torch.cat 处理
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))
    
    return new_logits  # （list，B 个 (1,)）


def dpo_loss(ref_probs, probs, beta):
    # DPO 损失：直接用偏好数据优化策略，无需显式训练 reward model
    # 数据布局(DPODataCollator)：一个 batch 前一半是 chosen、后一半是 rejected，故按 len//2 切分
    def split_probs(probs):
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]       # 前半 = chosen 的序列 log-prob
        reject_data = probs[len_chosen:]       # 后半 = rejected 的序列 log-prob
        return torch.cat(chosen_data), torch.cat(reject_data)
    
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)
    # pi_logratios = log π(y_w) - log π(y_l) = log[π(y_w)/π(y_l)]
    pi_logratios = chosen_probs - reject_probs
    # ref_logratios = log[π_ref(y_w)/π_ref(y_l)]
    ref_logratios = ref_chosen_probs - ref_reject_probs
    # logits = log[π(y_w)/π_ref(y_w)] - log[π(y_l)/π_ref(y_l)]：相对参考模型的偏好对数似然比
    logits = pi_logratios - ref_logratios
    # L = -log σ(β · logits)，即 DPO 的 Bradley-Terry 形式
    loss = -F.logsigmoid(beta*logits)
    return loss.mean()
    


class DPOTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        # 参考模型(冻结)只前向不反传：算 chosen/rejected 的序列 log-prob 作为"锚点"
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels = labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)
        ref_probs = mask_logits(ref_probs, labels)
        # 待优化的策略模型：同样算 chosen/rejected 的序列 log-prob（这一路才反传梯度）
        logits = model(input_ids=input_ids, labels = labels).logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)
        # beta=0.1 控制 KL 正则强度：越大越靠近 ref，越小越激进
        loss = dpo_loss(ref_probs, probs, 0.1)
        return loss

    # 以下 training_step 是一次"被弃用的优化尝试"：参考模型累计概率不变，本想只算一次 ref_probs、
    # 对策略模型多做几次更新以省算力；但与 Trainer 默认的梯度累积/优化流程耦合不顺，最终改回用上面的
    # compute_loss（每个 step 重算一次 ref）。保留作参考。
    # def training_step(
    #     self, model, inputs, num_items_in_batch=None
    # ) -> torch.Tensor:
    #     input_ids = inputs['input_ids']
    #     labels = inputs['labels']
    #     with torch.no_grad():
    #         ref_logits = ref_model(input_ids=input_ids, labels = labels).logits
    #     ref_probs = logits_to_probs(ref_logits, labels)
    #     ref_probs = mask_logits(ref_probs, labels)
    #     # 因为参考模型的累计概率不发生变化，为了尽量减少多次计算，计算一次参考模型的累积概率，多训练几次需要优化的模型
    #     for _ in range(1):
            
    #         model.train()
    #         logits = model(input_ids=input_ids, labels = labels).logits
    #         probs = logits_to_probs(logits, labels)
    #         probs = mask_logits(probs, labels)
        
    #         if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #             self.optimizer.train()

    #         with self.compute_loss_context_manager():
    #             # loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
    #             loss = dpo_loss(ref_probs, probs, 0.2)

    #         # del inputs
    #         if (
    #             self.args.torch_empty_cache_steps is not None
    #             and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #         ):
                
    #             torch.cuda.empty_cache()

    #         kwargs = {}

    #         if self.args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #         self.accelerator.backward(loss, retain_graph=True, **kwargs)
    #     # Finally we need to normalize the loss for reporting
    #     if num_items_in_batch is None:
    #         return loss.detach() / self.args.gradient_accumulation_steps
    #     return loss.detach()
    
        
if __name__ == "__main__":
    AutoConfig.register("small_model", Config)
    AutoModelForCausalLM.register(Config, LLM)
    # 策略模型 π：从 SFT 阶段保存的权重继续训（DPO 通常接在 SFT 之后）
    model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_model_from_scratch/saves/sft')

    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # 参考模型 π_ref：用同一份 SFT 权重，.eval() 关掉 dropout 且全程不训练（梯度不流入）
    ref_model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_model_from_scratch/saves/sft').eval().to('cuda')
    
    tokenizer = AutoTokenizer.from_pretrained("/home/user/wyf/train_model_from_scratch/tokenizer", use_fast=True)
    data_collator = DPODataCollator(tokenizer, max_seq_len=512) # 加载的大模型旋转位置编码最大长度为1024，这里不能超过这个值
    args = TrainingArguments(output_dir='./dpo-1-epoch', 
                            num_train_epochs=1,  # 训练太多轮，模型似乎会输出很多重复内容
                            do_train=True, 
                            per_device_train_batch_size=16,
                            gradient_accumulation_steps=4,
                            # max_steps=15000,
                            logging_steps=50,
                            report_to='tensorboard',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001,  # 学习率很重要，太大会把模型训飞
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            save_steps=100)          
    dataset = DPODataset('/home/user/wyf/train_model_from_scratch/dataset/dpo_data_512.json', tokenizer=tokenizer)
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/dpo-1-epoch')
    trainer.save_state()