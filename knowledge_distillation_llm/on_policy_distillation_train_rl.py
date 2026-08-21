"""把教师/学生 KL 当作奖励的 on-policy 蒸馏实验。

一次更新分三层批次：

* ``batch_size``：一次生成多少条完整回答；
* ``mini_batch_size``：一次构造多少条 KL 奖励并执行一次 optimizer.step；
* ``micro_batch_size``：一次前向/反向的样本数，梯度累积到 mini-batch。

默认值 16/8/2 表示：先生成 16 条回答，拆成两个 8 条的 mini-batch；每个
mini-batch 再拆成四个 2 条的 micro-batch，以降低峰值显存。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, DefaultDataCollator
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from tqdm import tqdm
import torch

from utils import *
from dataset import *


@dataclass
class TrainingArguments:
    """本实验使用的最小训练配置。

    这些值目前是类属性，例如可在训练前用 ``args.learning_rate = 5e-7`` 覆盖。
    """
    num_train_epochs = 1
    batch_size = 16 # 一次生成多少条完整回答
    mini_batch_size = 8 # 一次生成多少条 经验
    micro_batch_size = 2 # 拿出来 多少 进行 反向传播
    learning_rate = 1e-7
    weight_decay = 0.01
    logger_steps = 1
    save_steps = 500
    output_dir = "./outputs_rl"
    max_grad_norm = 1.0
    warmup_steps = 0
    max_length = 1024
    max_prompt_length = 512
    temperature = 1.0
    max_steps = None
    cliprange = 0.2
    

class OnPolicyDistillationTrainer:
    """
    执行 rollout、KL 奖励计算和 PPO 式 clipped policy update。
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        teacher_model: PreTrainedModel,

        args: TrainingArguments,
        tokenizer: PreTrainedTokenizer = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        optimizers = (None, None)
    ):
        
        self.model = model
        self.teacher_model = teacher_model
        self.args = args
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        
        
        if self.data_collator is None:
            self.data_collator = DefaultDataCollator()
            
            
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                collate_fn=self.data_collator,
                shuffle=True,
                drop_last=True,
                num_workers=8
            )

        # 优化器
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        
        self.max_steps = args.max_steps
        
        if self.max_steps is None:
            self.max_steps = len(self.train_dataloader) * args.num_train_epochs * args.batch_size // args.mini_batch_size

        # 学习率调度器
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=self.max_steps
            )
            
        self.writer = SummaryWriter(log_dir=args.output_dir)
        
        
            

    @torch.no_grad()
    def generate_sequences(self, input_ids, attention_mask):
        """
        使用更新前的学生策略(模型)采样一批轨迹（回答序列）。
        """

        self.model.eval()
        sequences = self.model.generate(input_ids=input_ids, 
                                      attention_mask=attention_mask,
                                      max_length=self.args.max_length,
                                      do_sample=True,
                                      temperature=self.args.temperature,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id
                                      )
        
     
        
        self.model.train()
        return sequences

    
    def selective_log_softmax(self, logits, index):
        """
        logits.shape = [batch, seq_len, vocab_size]
        index.shape  = [batch, seq_len]

        只取实际生成 token 的 log probability。

        例：某位置三个词的 log-softmax 为 ``[-2.0, -0.3, -1.8]``，实际生成
        token id=1，则返回 ``-0.3``。输入形状分别为
        ``[batch, seq_len, vocab]`` 和 ``[batch, seq_len]``，输出为
        ``[batch, seq_len]``。
        """

        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values 
        else:
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index): 
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def train(self):
        """
        运行完整的 on-policy 训练循环。
        """

        global_steps = 0
        pbar = tqdm(total=self.max_steps, desc="Training")
        
            
        for epoch in range(self.args.num_train_epochs):
            # 取一个batch_size生成数据
            for _, inputs in enumerate(self.train_dataloader):
                # 学生模型 采样生成
                prompt_ids = inputs["input_ids"].to(self.model.device)
                prompt_mask = inputs["attention_mask"].to(self.model.device)
                sequences = self.generate_sequences(prompt_ids, prompt_mask)
                
                
                with torch.no_grad():
                    attention_mask = (sequences != self.tokenizer.pad_token_id).long()
                    logits = self.model(sequences, attention_mask=attention_mask).logits
               
                
                # 取一个 mini_batch_size 构造“经验”：旧策略 logprob、KL 奖励和优势。
                for mini_idx in range(0, self.args.batch_size, self.args.mini_batch_size):
                    mini_input_ids = sequences[mini_idx:mini_idx+self.args.mini_batch_size]
                    mini_attention_mask = attention_mask[mini_idx:mini_idx+self.args.mini_batch_size]

                    with torch.no_grad():
                        mini_teacher_outputs = self.teacher_model(mini_input_ids, attention_mask=mini_attention_mask)

                        # 获取学生模型和教师模型输出的 logits
                        mini_student_logits = logits[mini_idx:mini_idx+self.args.mini_batch_size, prompt_ids.shape[-1]:]
                        mini_teacher_logits = mini_teacher_outputs.logits[:, prompt_ids.shape[-1]:]

                        # 提取学生模型生成的回答 token
                        mini_completion_ids = mini_input_ids[:, prompt_ids.shape[-1]:]

                        # 让教师和学生的词表维度相同
                        mini_teacher_logits_clamp = mini_teacher_logits[:, :, :mini_student_logits.shape[-1]]
                    
                        # 变量名虽然叫 probs，实际保存的是 log probabilities。
                        # 它作为 old policy logprob，稍后用于 importance ratio。
                        # 获取输出部分 token 的概率, 用于 重要性采样
                        mini_student_probs = self.selective_log_softmax(mini_student_logits, mini_completion_ids)

                        # 计算 KL 散度
                        kl = compute_rkl(mini_student_logits, mini_teacher_logits_clamp, mini_completion_ids, self.tokenizer.pad_token_id, reduction="")

                        """
                        从完整序列的 attention mask 中，只截取“学生生成回答”部分的有效 token mask
                        见笔记
                        """
                        mini_completion_mask = mini_attention_mask[:, prompt_ids.shape[-1]:]
                     
                        # KL 越小代表学生越接近教师，所以负 KL 越大越好：
                        # KL=0.1 -> reward=-0.1，优于 KL=1.2 -> reward=-1.2。
                        reward = -kl
                        reward_mean = (reward * mini_completion_mask).sum(dim=1, keepdim=True) / mini_completion_mask.sum(dim=1, keepdim=True)

                        """
                        token level 的 advantage
                        """
                        # 例：有效 token reward=[-0.2,-0.8]，均值=-0.5，
                        # 则 advantage=[+0.3,-0.3]，第一个 token 相对更接近教师。
                        # 这是“同一序列内按时间求均值”的简化 baseline，并非价值模型。
                        adv = reward - reward_mean
                        adv = adv * mini_completion_mask.float()
                        
                   
                    del mini_student_logits,mini_teacher_logits, mini_teacher_logits_clamp, mini_teacher_outputs
                    torch.cuda.empty_cache()
                  
                    self.optimizer.zero_grad()
                    
                    # 拆成 micro-batch，只做 backward；循环结束后才 optimizer.step，
                    # 从而用梯度累积降低峰值显存。
                    for micro_idx in range(0, self.args.mini_batch_size, self.args.micro_batch_size):
                        micro_input_ids = mini_input_ids[micro_idx:micro_idx+self.args.micro_batch_size]
                        micro_attention_mask = mini_attention_mask[micro_idx:micro_idx+self.args.micro_batch_size]
                        old_micro_student_probs = mini_student_probs[micro_idx:micro_idx+self.args.micro_batch_size]

                        
                        micro_adv = adv[micro_idx:micro_idx+self.args.micro_batch_size]
                        
                        micro_completion_ids = mini_completion_ids[micro_idx:micro_idx+self.args.micro_batch_size]

                        """
                        新策略 模型 的 micro_student_probs
                        """
                        micro_outputs = self.model(micro_input_ids, attention_mask=micro_attention_mask)
                        micro_student_logits = micro_outputs.logits[:, prompt_ids.shape[-1]:, :]
                        
                        micro_student_probs = self.selective_log_softmax(micro_student_logits, micro_completion_ids)
                    
                        del micro_outputs, micro_student_logits
                        torch.cuda.empty_cache()

                        micro_student_probs = micro_student_probs.masked_fill(micro_completion_ids == self.tokenizer.pad_token_id, 0.0)

                        """
                        旧策略模型 对 每个 token 输出 的 概率 old_micro_student_probs
                        """
                        old_micro_student_probs = old_micro_student_probs.masked_fill(micro_completion_ids == self.tokenizer.pad_token_id, 0.0)
                        
                        # PPO 重要性比率：ratio = new_prob / old_prob
                        # = exp(new_logprob - old_logprob)。策略未变化时 ratio=1。
                        logprobs_diff = micro_student_probs - old_micro_student_probs
              
                        ratio = torch.exp(logprobs_diff)
                        micro_completion_mask = micro_attention_mask[:, prompt_ids.shape[-1]:]
                        
                     
                        # unclipped 与 clipped 两项取较大的 loss，限制策略更新幅度。
                        # cliprange=0.2 时，ratio 在第二项中被限制到 [0.8, 1.2]。
                        pg_losses = -micro_adv * ratio
                        pg_losses2 = -micro_adv * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        
                    
                        # sequence loss：先对每条序列的有效 token 平均，再对 batch 平均。
                        token_loss_per_seq = (pg_loss_max * micro_completion_mask).sum(dim=1) / (micro_completion_mask.sum(dim=1) + 1e-8)
                        loss = token_loss_per_seq.mean()
                        
                        # token loss
                        # loss = (pg_loss_max * micro_completion_mask).sum() / micro_completion_mask.sum()
                        
                    
                        # 当前代码直接累加各 micro-batch 的平均 loss 梯度；如果希望梯度
                        # 尺度等价于整个 mini-batch 的平均值，还应除以 micro-batch 数量。
                        loss.backward()
                        
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    self.optimizer.step()
                    self.lr_scheduler.step() 
                    
                    
                    global_steps += 1
                    
                        
                    if global_steps % self.args.logger_steps == 0:
                       
                        pbar.set_postfix({
                        'epoch': f"{epoch+1}/{self.args.num_train_epochs}",
                        'global_step': global_steps,
                        'loss': f"{loss.item():.6f}",
                        'lr': f"{self.lr_scheduler.get_last_lr()[0]:.6f}",
                        'adv': f"{adv.mean().item():.6f}",
                        'kl': f"{kl.mean().item():.6f}"
                        })
                        pbar.update(1)
                        
                    if global_steps % self.args.save_steps == 0:
                        
                        self.model.save_pretrained(f"{self.args.output_dir}/model_{global_steps}")
                    
                    self.writer.add_scalar("loss", loss.item(), global_steps)
                    self.writer.add_scalar("lr", self.lr_scheduler.get_last_lr()[0], global_steps)
                    self.writer.add_scalar("adv", adv.mean().item(), global_steps)
                    self.writer.add_scalar("kl", kl.mean().item(), global_steps)
                    
           
          
        self.model.save_pretrained(f"{self.args.output_dir}/model_{global_steps}")
        self.writer.close()
        pbar.close()                    
                            

if __name__ == "__main__":
    
    model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    args = TrainingArguments()
    teacher_model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    model.cuda()
    teacher_model.cuda()
    teacher_model.eval()
    
    
    train_dataset = OnPolicyDataset('data.json', tokenizer, args)
    
    trainer = OnPolicyDistillationTrainer(
        model, 
        teacher_model, 
        args, 
        tokenizer=tokenizer,
        train_dataset=train_dataset)
    
    trainer.train()
