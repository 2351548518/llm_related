"""一个便于学习的 GRPO（Group Relative Policy Optimization）最小实现。

训练主线：
1. 对同一个 prompt 生成 ``num_generations`` 个回答，组成一个 group。
2. 用规则函数或奖励模型分别给每个回答打分。
3. 只在组内将奖励标准化为优势 ``A_i = (r_i - mean(r)) / (std(r) + eps)``，
   因此不需要像 PPO 那样额外训练一个 value/critic 模型。
4. 把同一回答的句子级优势广播给它的每个输出 token，优化 PPO 风格的裁剪目标，
   并可选择加入相对于参考模型的 KL 惩罚。

这里的实现以展示算法数据流为主，未包含分布式训练、混合精度、异常样本过滤等
工程优化。更完整的数值例子见同目录 README.md。
"""

from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class GSM8KDataset(Dataset):
    """读取本地中文版 GSM8K，并返回训练所需的题目与标准答案。

    预期每条数据至少包含：
    ``question_zh-cn``（中文题目）和 ``answer_only``（不含推导的最终答案）。
    DataLoader 会把若干条字典自动整理成 ``{"prompt": [...], "answer": [...]}``。
    """

    def __init__(self, data_path, tokenizer):
        
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        # prompt = self.tokenizer.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)
        answer = sample['answer_only']
        prompt = sample['question_zh-cn']
        return {'prompt': prompt, 'answer': answer}


@dataclass
class Samples:
    """一个 prompt 对应的一组生成结果。

    每个 prompt 生成 ``num_generations`` 条回答。prompt 和 response 拼接后的
    固定总长度为 ``max_prompt_length + max_generate_length``：

    - ``prompt_response_ids``:
      ``[num_generations, max_prompt_length + max_generate_length]``，
      保存 prompt 与 response 拼接后的 token。
    - ``response_ids``: ``[num_generations, max_generate_length]``，
      只保存生成部分，末尾可能包含 padding。
    - ``attention_mask``:
      ``[num_generations, max_prompt_length + max_generate_length]``，
      是完整 prompt 和 response 序列的掩码，模型前向计算时用它屏蔽 padding。
    - ``action_mask``: ``[num_generations, max_generate_length]``，
      是 response 部分的掩码，loss 只统计实际生成的回答 token。
    - ``num_actions``: 此实现中等于 ``max_generate_length``，
      表示 response 张量的固定宽度，不是每条回答的真实长度。
    - ``response_length``: ``[num_generations]``，
      分别记录每条回答实际参与 loss 的 token 数。
    """

    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int


class GRPOArguments:
    """示例训练超参数。

    这里使用类属性而非 ``@dataclass``，所有实例会读取同一套默认值；若在运行时修改
    ``args.reward_weights``，修改后的列表也会保留在该实例后续训练过程中。
    """
    
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4 # 每道题生成 4 个候选回答，优势只在这 4 个回答之间比较
    max_prompt_length = 256 # prompt 完成 token 化和左填充后的固定宽度
    max_generate_length = 256 # 每条 response 最多生成的 token 数，也是 response 张量的固定宽度
    reward_weights : List[float] = None # 多个奖励函数的加权系数；None 表示全部取 1
    beta = 0.0 # KL 惩罚系数；为 0 时不创建参考模型，也不计算 KL 项
    clip_eps = 0.2 # PPO 比率裁剪到 [0.8, 1.2]
    gradient_accumulation_steps = 2 # 累加 2 个 micro-batch 后执行一次 optimizer.step()
    num_iterations = 1 # 同一批生成经验重复训练的轮数；大于 1 时才真正复用 old policy 概率, 采样一次 样本 训练模型轮数
    batch_size = 1

class GRPOTrainer:
    """串联采样、奖励计算、GRPO loss 和参数更新。"""

    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):

        self.args = args

        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # 参考模型是采样时策略模型的冻结副本，仅用于约束新策略不要偏离得过远。
        # beta=0 时跳过它，可以节省大约一份模型权重的显存。
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
    
        # 加载 tokenizer
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # 设置为 左 填充
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        
        for i, reward_func in enumerate(reward_funcs):
            # 如果奖励函数为字符串，表示使用的是奖励模型，则加载模型
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        
        self.reward_funcs = reward_funcs
        
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")

        # 奖励模型
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            # 奖励既可以来自普通 Python 函数，也可以来自 sequence-classification 模型。
            # 后一种情况要为奖励模型准备自己的 tokenizer 和 pad_token_id。
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer


        self.reward_tokenizers = reward_tokenizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 缓存 gradient_accumulation_steps 个已生成 micro-batch：
        # 既用于梯度累加，也使 num_iterations>1 时可以复用同一批 on-policy 样本。
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # 模型更新的次数
        self.update_steps = 0 

    def get_tokenizer(self, tokenizer):
        """改为左 padding，使同批 prompt 的最后一个有效 token 对齐。

        ``generate`` 对 decoder-only 模型通常要求左 padding，否则较短 prompt 的生成起点
        会落在 padding 后面。这里只设置 padding 方向，调用方仍需确保 pad_token_id 已定义。
        """
        tokenizer.padding_side = "left"
        return tokenizer
    
    # 生成样本，以组为单位
    def generate_samples(self, inputs):
        """
        为 batch 中每个 prompt 生成一组回答，返回 ``List[Samples]``。

        例如 batch_size=2、num_generations=4 时，列表长度为 2；列表中的每个 Samples
        含 4 条候选回答。此处先按 prompt 分组保存，便于下一步做“组内”奖励标准化。
        """
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        # 提示词 最大长度 + 回答最大长度
        max_length = self.args.max_generate_length + self.args.max_prompt_length

        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)
            
            # 将同一个 prompt 复制 num_generations 份，
            # 再通过随机采样生成 num_generations 个候选回答。
            # 注意：temperature/top_p/top_k 只有在模型的 GenerationConfig 开启
            # do_sample=True 时才生效；若配置为贪心解码，组内回答可能完全相同，
            # 此时所有优势都接近 0，模型无法从组内相对比较中学习。
            # 这里的 tokenizer 是 左填充
            inputs = self.tokenizer([input_text] * self.args.num_generations, padding='max_length', max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')
            prompt_ids = inputs['input_ids']
            with torch.no_grad():
                prompt_response_ids = self.model.generate(**inputs.to(self.args.device), 
                                    max_new_tokens = self.args.max_generate_length,
                                    temperature=0.9,
                                    top_p = 1,
                                    top_k = 50)
                
            # 将每组整理为固定形状：
            # [num_generations, max_prompt_length + max_generate_length]。
            # 这样后续才能把不同 prompt 对应的组沿第一个维度拼接起来。
            if prompt_response_ids.size(1) >= max_length: # 截断
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else: # 填充
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
          
            # attention_mask 覆盖 prompt+response；action_mask 只覆盖要优化的 response。
            # EOS 本身不作为动作计入 loss，EOS 之后的 padding 也全部屏蔽。
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            # action_mask 输出的 掩码
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
        

            # 存储的是一个group的数据
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt = prompt,
                answer = answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list
    
    # 生成经验(优势、token的概率分布)
    def generate_experiences(self, inputs):
        """
        把生成文本变成一次 GRPO 更新需要的经验张量。

        返回张量的第一个维度为 ``batch_size * num_generations``。
        旧策略和参考模型的逐 token 对数概率形状为
        ``[batch_size * num_generations, max_generate_length]``；
        回答粒度的优势形状为 ``[batch_size * num_generations]``。
        """
        
        self.model.eval()
        samples_list = self.generate_samples(inputs)
        
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len)
            response_ids = samples.response_ids # shape: (num_generations, seq_len)
            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len)
            action_mask = samples.action_mask # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt

            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            
            with torch.no_grad():

                """
                计算策略模型 输出 token 的概率
                """
                # 保存采样策略 π_old 对已生成 token 的 log 概率。
                # num_iterations>1 时，新策略会逐轮变化，重要性比率必须以它为分母基准。
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)
                
                """
                计算参考模型输出 token 的概率
                """
                # 是否使用参考模型
                if self.ref_model:
                    #计算参考模型输出token的概率
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                """
                用来存储各个奖励 在 一个 group 内的 各个响应 的 奖励
                """
                # 第一个维度表示奖励函数，第二个维度表示组内候选回答：
                # [num_funcs, num_generations]。
                # 例如 4 个奖励函数、4 个回答时就是一个 4x4 矩阵。
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                """
                将输出转换成文本 并与 prompt 拼接起来
                """
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        """
                        使用奖励模型的话，奖励模型读入完整的“题目+回答”，每条输出一个标量分数。
                        """
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    
                    else:
                        """
                        如果使用 奖励函数 的话
                        规则奖励函数批量接收同一题目的 num_generations 个回答，
                        返回包含 num_generations 个奖励值的列表。
                        """
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                
                """
                得到的 奖励 加权求和
                """
                # rewards_per_func: [num_funcs, num_generations]
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                # 每一行先乘自己的权重，再沿奖励函数维求和，得到每个回答的总奖励。
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)
                
                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                print(f'rewards: {rewards}')

                """
                均值 和 方差
                """
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # GRPO 的关键：用同一道题的其它回答作为相对基线，而不训练 value model。
                # 例：总奖励 [3.5, 1.5, 1.0, 0.0] 的均值为 1.5、样本标准差约 1.472，
                # 对应优势约 [1.359, 0, -0.340, -1.019]；组内优势之和约为 0。
                # 优势是回答/句子粒度的，compute_loss 中会广播到该回答的所有 token。
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                batch_advantages.append(advantages)
        
               
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }
    
    def compute_loss(self, model, inputs):
        """计算 PPO-clip 风格的 GRPO loss，并按每条回答的有效长度归一化。

        对回答 i 的 token t：
        ``ratio_it = exp(log π_theta(a_it) - log π_old(a_it))``；
        ``L_it = -min(ratio_it*A_i, clip(ratio_it)*A_i) + beta*KL_it``。
        最后先对每条回答的有效 token 求平均，再对所有回答求平均。
        """
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)

        """
        当前模型 的 概率分布
        """
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        """
        计算 KL 损失, K3 分布
        """
        if self.args.beta != 0.0:
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            # Schulman 的非负 k3 KL 估计：exp(x) - 1 - x，其中
            # x = log(π_ref / π_theta)。当两个模型分布相同时 x=0、惩罚也为 0。
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']
        
        # 只训练一轮时，当前模型尚未更新，detach 后正好充当 π_old，ratio 初始为 1；
        # 复用经验多轮训练时，则必须使用采样阶段缓存下来的 old log-prob。
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs) # 形状为 [batch_size * num_generations, max_generate_length]
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        # 例：旧概率 0.2、新概率 0.3，则 ratio=1.5；clip_eps=0.2 时裁剪值为 1.2。
        # 对正优势，裁剪限制概率涨得过快；对负优势，裁剪限制概率降得过快。
        # 将每条回答的一个优势值从 [batch_size * num_generations]
        # 扩展到该回答的全部 max_generate_length 个 token 位置。
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # 先让长回答和短回答各自贡献一个平均 loss，避免长回答仅因 token 多而权重更大。
        # 前提是每条回答至少有一个有效 action，否则分母为 0 会产生 NaN。
        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1) # 形状为 [batch_size * num_generations]
        loss = loss.mean()
        
        # loss = per_token_loss.sum() / action_mask.sum()
        
        return loss


    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        """
        取出模型对序列中真实 token 的 log 概率，并截取回答部分。

        自回归模型当前位置的 logits 用来预测下一个 token，所以 logits 要去掉最后一个
        位置，目标 token 要去掉第一个位置，二者才能对齐。例如序列
        ``[序列开始标记, 第一个 token, 第二个 token]`` 中，序列开始标记位置的
        logits 用来查找第一个 token 的概率，第一个 token 位置的 logits 用来查找
        第二个 token 的概率。最终只保留末尾 ``num_actions=max_generate_length``
        个回答位置。
        """
        # 模型为序列中的每个位置输出一组覆盖整个词表的未归一化分数。
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits

        # 最后一个位置没有对应的“下一个 token”标签，因此删除该位置的 logits；
        # 再转换为对数概率。结果形状为：
        # [batch_size * num_generations, 完整序列长度 - 1, 词表大小]。
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

        # input_ids[:, 1:] 删除第一个 token，使每个 logits 位置与它预测的下一个
        # 真实 token 对齐；gather 再从词表维度取出这个真实 token 的对数概率。
        # 结果形状为：
        # [batch_size * num_generations, 完整序列长度 - 1, 1]。
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))

        # 删除最后一个长度为 1 的维度，并截取末尾 num_actions 个 response 位置；
        # EOS 和 PAD 位置会在 compute_loss 中通过 action_mask 屏蔽。
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    
    
    def train_step(self, model, inputs, optimizer, step):
        """对一个 micro-batch 反向传播，并在累加周期末更新参数。"""
        model.train()
        # scaler = torch.amp.GradScaler()
        # with torch.amp.autocast(device_type='cuda'):
        loss = self.compute_loss(model, inputs)
        # 除以累加步数，使累加后的梯度尺度近似于对多个 micro-batch 取平均。
        # 因此 TensorBoard 中记录的也是除法后的 micro-batch loss。
        loss = loss / self.args.gradient_accumulation_steps
        # loss = scaler.scale(loss)
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            
            optimizer.step()
            optimizer.zero_grad()
            # scaler.unscale_(optimizer)
            # scaler.step(optimizer)
            # scaler.update()
        
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    def train(self):
        """循环执行“生成经验 -> 缓存 -> 重复训练 -> 定期保存”。

        不满一个 gradient accumulation 周期的末尾 batch 会被丢弃；这是这个教学实现
        为保持逻辑简洁所作的取舍。
        """
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                
                # 先在线采样，因此每个 accumulation 周期使用的是当前策略产生的经验。
                inputs = self.generate_experiences(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    for _ in range(self.args.num_iterations):
                        # 先遍历缓存中的 micro-batch 累加梯度；最后一个 train_step 才更新。
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                        
                del inputs
    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)           

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""
    
    args = GRPOArguments()
    
    writer = SummaryWriter('./runs')
    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    

    
    
    prompts_dataset = GSM8KDataset('/home/user/wyf/deepseek_learn/gsm8k_chinese', tokenizer)
  
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    

