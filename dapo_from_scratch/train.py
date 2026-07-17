"""从零实现的教学版 DAPO/GRPO 训练脚本。

主流程：

1. 对每个问题生成 ``num_generations`` 个回答；
2. 用多个规则奖励函数给回答打分；
3. 在同题回答组内标准化奖励，得到序列级 advantage；
4. 丢弃 advantage 全为 0 的回答组并继续采样；
5. 计算 PPO clipped surrogate loss；
6. 按同题回答组中的所有有效 token 求平均，得到 DAPO token-level loss。

与论文/官方实现的关键差异：

* ``beta=0`` 实现了移除 reference-policy KL；
* 已写出 Clip-Higher 的 0.2/0.28 非对称裁剪，但默认
  ``num_iterations=1`` 时新旧策略概率数值相同，裁剪基本不会触发；
* 动态采样按“多项总奖励是否有差异”过滤，不是只按二值准确率过滤；
* 当前 ``generate`` 没有显式传 ``do_sample=True``；
* 没有实现 Overlong Reward Shaping；
* 使用 3B Instruct、短输出和单机训练，仅用于理解算法，不是 32B 复现。

下面的注释直接使用代码中的变量名描述张量形状：

* ``batch_size``：一个训练 batch 中的 prompt group 数；
* ``num_generations``：每个 prompt 生成的回答数；
* ``num_actions``：补齐后的回答 token 数。
"""

from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class GSM8KDataset(Dataset):
    """读取中文 GSM8K，并返回训练所需的 prompt 和标准答案。

    单条样本示例：

        {
            "prompt": "小明有 3 个苹果，又买了 2 个，一共有几个？",
            "answer": "5",
        }

    ``tokenizer`` 当前只保存为成员，真正的 chat template 在生成阶段应用。
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
    """保存同一个 prompt 对应的一整组 rollout。

    设 ``num_generations=4``、prompt 最大长度为 256、回答最大长度为
    256，则典型形状为：

    * ``prompt_response_ids``: [4, 512]
    * ``response_ids``: [4, 256]
    * ``attention_mask``: [4, 512]
    * ``action_mask``: [4, 256]
    * ``response_length``: [4]，记录每条回答的有效 token 数

    ``action_mask`` 为 1 的位置才参与策略损失，padding 和 EOS 不参与。
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
    """集中保存教学实验超参数。

    这里使用类属性而不是 ``dataclass``，创建 ``GRPOArguments()`` 后仍可
    通过实例覆盖，例如 ``args.num_iterations = 2``。
    """

    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4 # 组内样本数
    max_prompt_length = 256 # 最大输入长度
    max_generate_length = 256 # 最大输出长度
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    # DAPO 移除 reference-policy KL；若设为非零值，代码会复制 ref_model。
    beta = 0.0
    # Clip-Higher：下降方向仍使用 0.2，上升方向放宽到 0.28。
    clip_eps_high = 0.28
    clip_eps_low = 0.2
    gradient_accumulation_steps = 2 # 梯度累加
    # 同一批 rollout 被重复用于多少轮策略更新。大于 1 时新旧策略才会
    # 逐渐拉开，重要性采样比率可能越过裁剪边界。
    num_iterations = 1
    batch_size = 1


class GRPOTrainer:
    """组织 rollout、奖励、优势估计和策略更新。

    这里沿用 ``GRPOTrainer`` 名称，是因为优势仍由 GRPO 的组内标准化
    得到；最终 loss reduction 和若干训练技巧则采用 DAPO 思路。
    """

    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):
        """初始化策略模型、可选参考模型、奖励函数和优化器。

        ``reward_funcs`` 中既可以放 Python 函数，也可以放奖励模型路径。
        Python 函数返回每条回答的标量奖励；奖励模型则返回一个 logit。
        """

        self.args = args
        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
    
        
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
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
            
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
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
        
        # 缓存已经生成的数据的一个批次的数据，可供模型多次训练迭代，无需重新生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # 模型更新的次数
        self.update_steps = 0 

    def get_tokenizer(self, tokenizer):
        """使用左侧 padding，让不同长度 prompt 的末尾与生成起点对齐。"""
        tokenizer.padding_side = "left"
        return tokenizer
    
    # 生成样本，以组为单位
    def generate_samples(self, inputs):
        """为 batch 中的每个 prompt 生成 ``num_generations`` 条回答。

        输入示例（DataLoader 的 batch）：

            {
                "prompt": ["1+1 等于多少？"],
                "answer": ["2"],
            }

        若 ``num_generations=4``，函数会返回一个 ``Samples``，内部含 4
        条回答。此处逐个 prompt 调用 ``generate``，因此
        ``samples_list`` 的长度等于输入 prompt 数，而不是回答总数。

        注意：代码虽然传入 temperature/top_p/top_k，却没有显式设置
        ``do_sample=True``。在常见 Transformers 默认配置下会使用贪心
        解码，4 条回答可能完全相同。这里保留原逻辑，只在注释中提示。
        """
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)
            
            # 生成一个group的输入数据
            inputs = self.tokenizer([input_text] * self.args.num_generations, padding='max_length', max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')
            prompt_ids = inputs['input_ids']
            with torch.no_grad():
                # 若希望真正进行组内随机采样，通常需要额外传入
                # do_sample=True。否则 temperature/top_p/top_k 可能被忽略。
                prompt_response_ids = self.model.generate(**inputs.to(self.args.device), 
                                    max_new_tokens = self.args.max_generate_length,
                                    temperature=0.9,
                                    top_p = 1,
                                    top_k = 50)
                
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
          
            # 完整序列 mask：
            # [num_generations, max_prompt_length + max_generate_length]。
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            # 只让真实生成 token 参与策略损失；EOS 和 PAD 的位置为 0。
            # 例：回答 token 为 [token_1, token_2, EOS, PAD]，
            # mask 为 [1, 1, 0, 0]。
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
        """把 rollout 转换成可训练的策略梯度经验。

        对每个 prompt group 执行：

        1. 计算每条回答的多项奖励并加权求和；
        2. 用 ``(reward - group_mean) / group_std`` 得到 GRPO advantage；
        3. 丢弃 advantage 全 0 的组；
        4. 保存 rollout 时策略和可选参考策略的 token log-prob。

        数值例子：若 ``num_generations=4``，总奖励为 [3, 1, 1, 1]，
        PyTorch 默认样本标准差为 1，均值为 1.5，advantage 为
        [1.5, -0.5, -0.5, -0.5]。第一条回答会被鼓励，其余会被抑制。
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
            
            with torch.no_grad():
                
                # 存储各个奖励函数在一个group内各个响应的奖励
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                # 将输出转换成文本
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    
                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                
                # rewards_per_func: [num_reward_funcs, num_generations]。
                # 例如 4 个奖励函数、每题 4 个回答时形状为 [4, 4]。
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                # 乘以各个奖励函数的权重
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)
                # rewards: [num_funcs, num_generations]
                # 沿奖励函数维求和，得到每条回答一个总奖励：
                # [num_generations]。
                # 当前四项奖励的理论最高分为 2.0+0.5+0.5+0.5=3.5。
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                print(f'rewards: {rewards}')
                
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # GRPO 的 advantage 是回答/序列粒度，而不是 token 粒度。
                # 同一条回答中的所有 token 在 compute_loss() 中共享此值。
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                # Dynamic Sampling 的教学版实现：
                # 如果所有总奖励相同，则所有 advantage 都为 0，该组无法
                # 产生策略梯度，因此跳过并由 train() 的 buffer 继续补样本。
                #
                # 与论文区别：论文主要按二值 acc 过滤全对/全错组；这里
                # 使用“正确性 + 数字 + 格式 + 标签”的总奖励。因此即使
                # 全部答错，只要格式奖励不同，该组仍可能被保留。
                nonzero_num = advantages.count_nonzero().item()
                if nonzero_num == 0:
                    continue
                
                batch_advantages.append(advantages)
                
                # 保存 rollout policy 的 token log-prob：
                # [num_generations, num_actions]。
                # 多次更新同一批经验时，它就是 PPO 比率中的分母。
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # 是否使用参考模型
                if self.ref_model:
                    #计算参考模型输出token的概率
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                    
                
                batch_prompt_response_ids.append(prompt_response_ids)
                batch_attention_mask.append(attention_mask)
                batch_action_mask.append(action_mask)
        
               
        return {
            "prompt_response_ids": batch_prompt_response_ids,
            "attention_mask": batch_attention_mask,
            "action_mask": batch_action_mask,
            "old_action_log_probs": batch_old_action_log_probs,
            "ref_action_log_probs": batch_ref_action_log_probs if self.ref_model else None,
            "advantages": batch_advantages,
        }
    
    def compute_loss(self, model, inputs):
        """计算非对称 PPO clipped loss，并使用 DAPO token-level reduction。

        对每个有效 token：

            ratio = exp(log_p_new - log_p_old)
            surrogate = min(ratio * A, clip(ratio) * A)

        代码最小化 ``-surrogate``。对于正 advantage，ratio 上升会提高
        目标；超过 ``1 + clip_eps_high`` 后不再从该样本获得额外收益。

        例：旧概率 0.01、新概率 0.015，则 ratio=1.5。使用 DAPO
        ``clip_eps_high=0.28`` 时，正优势 clipped 分支按 1.28 计算。
        这不是把实际概率强制改回 0.0128，只是截断优化目标中的收益。

        最后不是先对每条回答求平均，而是：

            group_loss = 组内所有有效 token loss 之和
                         / 组内有效 token 总数

        再对 ``batch_size`` 个 prompt group 求平均。这对应论文的
        Token-Level PG Loss。
        """
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        if self.args.beta != 0.0:
            
            ref_action_log_probs = inputs['ref_action_log_probs']
            # log_ratio = log(p_ref / p_new)。下面的 k3 是常用的低方差
            # KL(pi_new || pi_ref) 单样本估计；beta=0 时整段不会执行。
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            # k3: log_ratio.exp() - 1 - log_ratio
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']
        
        # num_iterations=1 时，用当前 log-prob 的 detach 版本作为 old 值：
        # 前向数值上 ratio=1，梯度仍可反传，但 clipping 不会触发。
        # num_iterations>1 时使用 rollout 阶段保存的 old log-prob，同一批
        # 经验重复更新后 ratio 才可能偏离 1 并越过裁剪边界。
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        # 重要性采样比率 r = p_new / p_old：
        # [batch_size * num_generations, num_actions]。
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        # DAPO Clip-Higher：下界 0.8，上界 1.28。
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps_low, 1 + self.args.clip_eps_high)
        # advantages: [batch_size * num_generations]。
        # unsqueeze 后变为 [batch_size * num_generations, 1]，
        # 再广播到每个 token。
        # 因而同一回答中的 token 共享同一个序列级 advantage。
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) # shape: [batch_size * num_generations, num_actions]
        per_token_loss = per_token_loss * action_mask  
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # 原始 GRPO 的 sample-level reduction：
        # 先对每条回答内部的 token 求平均，再对所有回答求平均。
        #
        # loss = (per_token_loss.sum(dim=1) / action_mask.sum(dim=1))
        # # shape: [batch_size * num_generations]
        # loss = loss.mean()
        
        
        # DAPO 的 group token-level reduction：
        # [batch_size * num_generations, num_actions]
        # -> [batch_size, num_generations, num_actions]，
        # 恢复“prompt group/回答/token”三层结构。
        per_token_loss = per_token_loss.view(-1, self.args.num_generations, num_actions)
        action_mask = action_mask.view(-1, self.args.num_generations, num_actions)
        # 先汇总每组 num_generations 条回答中的所有有效 token，
        # 再除以有效 token 总数。
        # 例：两条回答长度为 2 和 6，则 DAPO 中每个 token 权重均为 1/8；
        # 原始 GRPO 中短回答每个 token 权重 1/4，长回答则只有 1/12。
        loss = per_token_loss.sum(-1).sum(-1) / action_mask.sum(-1).sum(-1) # shape: [batch_size]
        loss = loss.mean()
        
        return loss


    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        """取出模型对实际生成 response token 给出的 log-prob。

        因果语言模型在位置 t 的 logits 预测位置 t+1 的 token，因此：

        * ``logits[:, :-1]`` 去掉最后一个没有目标 token 的位置；
        * ``input_ids[:, 1:]`` 是与 logits 对齐的真实下一个 token；
        * ``gather`` 只取这些真实 token 的 log-prob；
        * 最后 ``-num_actions:`` 只保留 response 部分。

        返回形状为
        ``[batch_size * num_generations, num_actions]``，
        单组输入时为 ``[num_generations, num_actions]``。
        """

        # 计算策略模型输出 token 的概率
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    
    
    def train_step(self, model, inputs, optimizer, step):
        """处理一个 micro-batch，并在累积满后执行一次参数更新。

        若 ``gradient_accumulation_steps=2``，前两个 micro-batch 的 loss
        都先除以 2 并反向传播；处理第二个后才调用 ``optimizer.step()``。
        """
        model.train()
        # scaler = torch.amp.GradScaler()
        # with torch.amp.autocast(device_type='cuda'):
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        # loss = scaler.scale(loss)
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            
            optimizer.step()
            optimizer.zero_grad()
            # scaler.unscale_(optimizer)
            # scaler.step(optimizer)
            # scaler.update()
        
            # 标签沿用历史名称 "grpo_loss"，但当前记录的是采用 DAPO
            # token-level reduction 后的 loss。因此目录中的 dapo_loss.png
            # 仍显示 grpo_loss 标签，不能仅凭标签判断所用 reduction。
            # 另外，组内标准化会让正负 advantage 经常相互抵消，loss 接近
            # 0 不代表准确率已经很高；解读两张图片时还应同时记录 reward、
            # 验证集准确率、entropy、回答长度和有效回答组比例。
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    def train(self):
        """执行 rollout 与策略更新交替进行的训练循环。

        ``buffer`` 以 prompt group 为单位缓存有效经验。举例：

        * batch_size=2；
        * 第 1 道题四个回答奖励完全相同 -> 被过滤；
        * 第 2、3 道题奖励有差异 -> 放入 buffer；
        * buffer 凑到 2 个有效 group 后，再组成
          `[batch_size * num_generations, ...]` 张量训练。

        因此这里的“动态”是实际生成 prompt 数不固定，而每次更新使用的
        有效 prompt group 数保持为 ``batch_size``。
        """
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            buffer = {'prompt_response_ids':[],
                      'attention_mask':[],
                      'action_mask':[],
                      'old_action_log_probs':[],
                      'ref_action_log_probs':[],
                      'advantages':[]}
            idx = 0
            for batch in dataloader:

                inputs = self.generate_experiences(batch)
                buffer['prompt_response_ids']+=inputs['prompt_response_ids']
                buffer['attention_mask']+=inputs['attention_mask']
                buffer['action_mask'] += inputs['action_mask']
                buffer['old_action_log_probs'] += inputs['old_action_log_probs']
                if self.ref_model is not None:
                    buffer['ref_action_log_probs'] += inputs['ref_action_log_probs']
                else:
                    buffer['ref_action_log_probs'] = None
                
                buffer['advantages'] +=inputs['advantages']
                
             
                # 如果有效 group 少于设定的 batch_size，说明有零优势组被舍弃，需要继续 rollout，直到凑够一个完整训练 batch。
                if len(buffer['prompt_response_ids']) < self.args.batch_size:
                    continue
                
                if self.ref_model is not None:
                    inputs = {k: v[:self.args.batch_size] for k, v in buffer.items()}
                    inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}
                    buffer = {k: v[self.args.batch_size:] for k, v in buffer.items()}
                    
                else:
                    inputs = {k: v[:self.args.batch_size] for k, v in buffer.items() if k != 'ref_action_log_probs'}
                    inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}
                    inputs['ref_action_log_probs'] = None
                    buffer = {k: v[self.args.batch_size:] for k, v in buffer.items() if k != 'ref_action_log_probs'}
                    buffer['ref_action_log_probs'] = None
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs

                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                
                idx += 1
                   
                del inputs
    def save_model(self):
        """保存最终策略模型和 tokenizer。"""
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)           

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # 强制模型把推理和最终答案放在不同标签中。reward_func.py 会分别
    # 检查答案正确性、是否为数字以及标签格式。
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
    # 策略模型。论文使用 Qwen2.5-32B Base；这里用 3B Instruct 方便
    # 单机教学实验，因此结果不能直接与论文 AIME 指标比较。
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-3B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-3B-Instruct')
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    

    
    
    prompts_dataset = GSM8KDataset('/home/user/wyf/deepseek_learn/gsm8k_chinese', tokenizer)
  
    # 四项奖励默认等权相加。若回答正确、为数字且格式完全正确，最高
    # 可得 3.5；组内再对总奖励做标准化，绝对分值本身不会直接作为梯度。
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    

