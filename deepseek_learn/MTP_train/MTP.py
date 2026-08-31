from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Dict, Any
import json
import os
from torch.utils.tensorboard import SummaryWriter


class Config():
    def __init__(self,
                llm_model_path = '/data2/home/jiapeng2/code/LLM/llm_related/models/Qwen2.5-0.5B-Instruct',
                predict_tokens_num = 5, # 主模型 预测的 token 加上 MTP 预测的 token ( 1 个 主模型 token +  MTP 预测的 4 个 token)
                **kwargs):
        self.llm_model_path = llm_model_path # 主模型 路径
        self.predict_tokens_num = predict_tokens_num # 预测多少个 token
        super().__init__(**kwargs)

"""
使用 MLP 代替 原始论文的 Transformer block
"""
class MTPModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
        

class MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # base_model 去掉了 预测头 去掉了
        self.main_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path).base_model 

        # self.main_model.eval()
        # mtp模块 一个主模型 + MLP预测头
        self.mtp_modules = nn.ModuleList([MTPModule(self.main_model.config.hidden_size) for _ in range(self.config.predict_tokens_num-1)])
        
        # 每个头共享参数
        self.output_head = nn.Linear(self.main_model.config.hidden_size, self.main_model.config.vocab_size)
        
         
    def forward_main(self, input_ids, attention_mask=None, **kwargs):
        """
        主模型头的 前向传播
        """
        # with torch.no_grad():
        main_hidden_output = self.main_model(input_ids, attention_mask, **kwargs).last_hidden_state
        main_head_output = self.output_head(main_hidden_output)
        
        return main_hidden_output, main_head_output
    
    def forward_mtp(self, input_ids, previous_hidden_output, head_index):
        """
        MTP 头 的 前向传播

        head_index: 第几个 MTP 头

        input_ids = [t1, t2, t3, t4, t5, t6]
        previous_hidden_output = [h1⁰, h2⁰, h3⁰, h4⁰, h5⁰, h6⁰]

        第一个 MTP 头：head_index = 0
        [e(t2), e(t3), e(t4), e(t5)] cat [h1⁰, h2⁰, h3⁰, h4⁰] -> 预测 [t3, t4, t5, t6]

        第二个 MTP 头：head_index = 1
        previous_hidden_output = [h1¹, h2¹, h3¹, h4¹]
        [e(t3), e(t4), e(t5)] cat [h1¹, h2¹, h3¹] -> [t4, t5, t6]

        ---
        训练时：
            主模型：   t1 → t2，t2 → t3，t3 → t4，t4 → t5，t5 → t6
            MTP头0： (t1,t2) → t3，(t2,t3) → t4，…… 
            MTP头1： (t1,t2,t3) → t4，……
            MTP头2： (t1,t2,t3,t4) → t5，……
            MTP头3： (t1,t2,t3,t4,t5) → t6
        推理时：
            主模型： h1⁰                       → t2
            MTP头0：(h1⁰, t2)                 → t3
            MTP头1：(h1¹, t3)，h1¹包含t1,t2   → t4
            MTP头2：(h1², t4)，h1²包含t1~t3   → t5
            MTP头3：(h1³, t5)，h1³包含t1~t4   → t6
        ---
        对比：
            普通生成：需要主模型顺序生成 t2 → t3 → t4 → t5 → t6
            MTP生成：主模型生成 t2，轻量MTP模块继续草拟 t3 → t4 → t5 → t6
        """

        """
        这是训练的 代码
        """
        mtp_input_ids = input_ids[:, head_index + 1:-1]
        input_embed = self.main_model.get_input_embeddings()(mtp_input_ids)

        current_hidden_output = previous_hidden_output[:, :mtp_input_ids.size(1), :]
        

        mtp_input = torch.cat([current_hidden_output, input_embed], dim=-1)
        mtp_hidden_output = self.mtp_modules[head_index](mtp_input)
        mtp_head_output = self.output_head(mtp_hidden_output)
        
        return mtp_hidden_output, mtp_head_output

    def forward_mtp_step(self, input_ids, previous_hidden_output, head_index):
        """
        推理时：
            主模型： h1⁰                      → t2
            MTP头0：(h1⁰, e(t2))                → t3
            MTP头1：(h1¹, e(t3))，h1¹包含t1,t2   → t4
            MTP头2：(h1², e(t4))，h1²包含t1~t3   → t5
            MTP头3：(h1³, e(t5))，h1³包含t1~t4   → t6
        举例：
            (h1, 预测出来的t2) → t3
        """
        input_embed = self.main_model.get_input_embeddings()(input_ids)
        mtp_input = torch.cat([previous_hidden_output, input_embed], dim=-1)
        mtp_hidden_output = self.mtp_modules[head_index](mtp_input)
        mtp_head_output = self.output_head(mtp_hidden_output)

        return mtp_hidden_output, mtp_head_output
    
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        最终的 前向传播, 主模型 和 MTP

        用来 训练的 前向 传播的 代码
        """
        outputs = {}
        main_hidden_output, main_head_output = self.forward_main(input_ids, attention_mask, **kwargs)
        previous_hidden_output = main_hidden_output
        outputs['head_main'] = main_head_output
        for head_index in range(0, self.config.predict_tokens_num-1):
            previous_hidden_output, mtp_head_output = self.forward_mtp(input_ids, previous_hidden_output, head_index)
            outputs[f'mtp_head_{head_index}'] = mtp_head_output
            
        return outputs
    
    def generate(self,input_ids,max_length, **kwargs):
        """
        生成部分 的 代码
        推理时：
            主模型： h1⁰                      → t2
            MTP头0：(h1⁰, e(t2))                → t3
            MTP头1：(h1¹, e(t3))，h1¹包含t1,t2   → t4
            MTP头2：(h1², e(t4))，h1²包含t1~t3   → t5
            MTP头3：(h1³, e(t5))，h1³包含t1~t4   → t6
        
        之后 需要 验证 MTP 生成的 token
        """
        self.eval()
        seq = input_ids.clone()
        b, s = seq.size()
        
        with torch.no_grad():
            
            while seq.size(1) < max_length:
                print(seq.shape)
                speculative_tokens = []
                
                # main模型头生成的token
                main_hidden_output, main_head_output = self.forward_main(seq)
                logits = main_head_output
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                speculative_tokens.append(next_token.unsqueeze(1))

                previous_hidden_output = main_hidden_output[:, -1:, :]
                current_input_ids = next_token.unsqueeze(1)
                
                # 汇总mtp头生成的token
                for i in  range(self.config.predict_tokens_num-1):
                    previous_hidden_output, mtp_head_output = self.forward_mtp_step(
                        current_input_ids,
                        previous_hidden_output,
                        i,
                    )
                    logits = mtp_head_output
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1)
                    
                    speculative_tokens.append(next_token.unsqueeze(1))
                    current_input_ids = next_token.unsqueeze(1)
                
                  
                
                speculative_tokens = torch.cat(speculative_tokens, dim=-1)
                
                # 将新生成的tokens和原始序列拼接
                all_tokens = torch.cat([seq, speculative_tokens], dim=-1)
                
                # 将新序列输入main模型(验证模型)进行验证，保留符合条件的token
                _, all_logits = self.forward_main(all_tokens)
                
                """
                第一个token由main模型直接生成，这里只验证后续的mtp token
                NOTE: all_logits[:, seq.shape[1]:-1] 到 -1 是因为 上一个 的 概率 是用来预测 下一个 token的
                位置 i 的 logits，用来预测位置 i+1 的 token
                """
                validation_logits = all_logits[:, seq.shape[1]:-1]
                validation_tokens = speculative_tokens[:, 1:]
                
                # 获取各个token在main模型的输出概率
                accept_num = 1
                if validation_tokens.shape[1] > 0:
                    accept_probs =  []

                    for i in range(validation_tokens.shape[1]): # validation_tokens.shape[1] = MTP 预测 的 token数量
                        logits = validation_logits[:, i] # (batch_size, vocab_size)
                        probs = torch.softmax(logits, dim=-1) # (batch_size, vocab_size)
                        token = validation_tokens[:, i] # 例如：token = tensor([42]) 说明需要查看主模型分配给词表编号 42 的概率。
                       
                        token_prob = probs.gather(1, token.unsqueeze(1)) # [1] → [1, 1], gather 在词表维度中找到 token 对应的概率：
                        
                        accept_probs.append(token_prob)

                    """
                    [
                        y2的概率，   # shape [1,1]
                        y3的概率，   # shape [1,1]
                        y4的概率，   # shape [1,1]
                        y5的概率     # shape [1,1]
                    ]
                    """
                    # 拼接各个token的生成概率 accept_probs.shape = [1, 4] = [batch_size, token_num]
                    accept_probs = torch.cat(accept_probs, dim=-1) 
                    
                    # 保留概率值大于阈值的token, 接受这部分token,否则舍弃（舍弃某个token时，后面的token都要舍弃）
                    # 接受token的掩码
                    accept_mask = (accept_probs > 1e-6) # accept_mask.shape = [1, 4] = [batch_size, token_num]
                    print(f'接受掩码：{accept_mask}')
                    print(f'拒绝掩码：{~accept_mask}')
                    # 获取被拒绝（舍弃）token对应的索引 找出第一个拒绝位置
                    reject_token_index = (~accept_mask).nonzero(as_tuple=True)[1]
                    print(f'拒绝token的索引：{reject_token_index}')
                    
                    if reject_token_index.shape[0] > 0:
                        # 如果有需要舍弃的token
                        # 第一个token默认接受，后续接受数量由第一个被拒绝的mtp token决定
                        accept_num += reject_token_index[0].item()
                    
                    else:
                        # 如果没有需要舍弃的token，则全部接受
                        accept_num = speculative_tokens.shape[1]
                
                # 接受生成的 token
                if accept_num > 0:
                    
                   # 取出通过验证的token
                    accept_tokens = speculative_tokens[:, :accept_num]
                   
                    seq = torch.cat([seq, accept_tokens], dim=1)
                
                else:
                    logits = main_head_output
                    
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1)
                    next_token = next_token.unsqueeze(1)
                
                    
                    seq = torch.cat([seq, next_token], dim=-1)
                    # print(seq)
                    
                
            return seq
            
            
        
        
                    
        
def train(config, model, dataloader, optimizer, writer, device, epochs, print_step, save_step, save_path):
    steps = 0
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            
            main_hidden_output, main_head_output = model.forward_main(input_ids)
            previous_hidden_output = main_hidden_output
            for index in range(0, config.predict_tokens_num-1):
                previous_hidden_output, mtp_head_output = model.forward_mtp(input_ids, previous_hidden_output, index)
                
                mtp_head_output = mtp_head_output.reshape(-1, model.main_model.config.vocab_size) # [batch_size * seq_len, vocab_size]
                
                target = labels[:, 1+index+1:] # [batch_size, seq_len]
                target = target.contiguous().view(-1) # [batch_size * seq_len]
                # 用来训练 MTP 模块
                mtp_loss = F.cross_entropy(mtp_head_output, target, ignore_index=-100)
                mtp_loss.backward(retain_graph=True)
                
            # 对 主模型 进行 反向传播
            main_loss = F.cross_entropy(main_head_output[:, :-1].reshape(-1, model.main_model.config.vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100)
            main_loss.backward()
            
            optimizer.step() # 优化器使用累加后的总梯度更新参数
            
            if (steps+1) % print_step==0:
                writer.add_scalar('main_loss', main_loss.item(), steps)
                writer.add_scalar('mtp_loss', mtp_loss.item(), steps)
                print(f"Epoch {epoch+1}], Step {steps+1}, main_loss: {main_loss.item():.4f}, mtp_loss: {mtp_loss.item():.4f}")
                
            if (steps+1) % save_step==0:
                torch.save(model.state_dict(), f"{save_path}/model_{steps}.pth")
            
            steps += 1  
        
    
class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        
        self.tokenizer = tokenizer
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = f.readlines()

            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index].strip()
        sample = json.loads(sample)
        conversations = sample['conversations']
        user = conversations[0]['content']
        assistant = conversations[1]['content']
        
        q = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        
        a = assistant + self.tokenizer.eos_token
        q_input_ids = self.tokenizer(q)['input_ids']
        a_input_ids = self.tokenizer(a)['input_ids']
        
        input_ids = q_input_ids + a_input_ids
        
        labels = [-100] * len(q_input_ids) + a_input_ids
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
        
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        """
        保证 一个 batch 的 所有 sample 的长度相同
        这里是 右 padding
        """
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [-100] * (max_len - len(feature['labels'])))
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)}
        

            
        
        
if __name__ == '__main__':
    # 日志记录
    writer = SummaryWriter('./runs')
    config = Config()
    model = MTP(config)
    model.cuda()
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    dataset = MyDataset('/home/user/wyf/deepseek_learn/MTP_train/lora_medical.jsonl', tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=MyDataCollator(tokenizer))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_path = './mtp'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train(config, model, dataloader, optimizer, writer, device='cuda', epochs=10, print_step=10, save_step=500, save_path='mtp')