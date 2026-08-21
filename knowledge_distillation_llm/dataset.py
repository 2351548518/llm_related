"""训练数据集定义。

输入文件采用 Alpaca 风格的 JSON 数组，例如：
``[{"instruction": "计算 1+1", "input": "", "output": "2"}]``。

``SFTDataset`` 返回“问题+标准答案”，用于离线蒸馏；``OnPolicyDataset``
只返回问题，因为答案需要在训练过程中由当前学生模型自己生成。
"""

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig

class SFTDataset(Dataset):
    """构造监督微调/离线蒸馏样本。

    假设 chat prompt 编码后有 4 个 token，答案有 2 个 token，且
    ``max_seq_len=8``，则概念上的结果为：

    * input_ids：4 个 prompt token + 2 个答案 token + 2 个 pad；
    * labels：``[-100, -100, -100, -100, answer_id, eos_id, -100, -100]``；
    * attention_mask：``[1, 1, 1, 1, 1, 1, 0, 0]``。

    ``-100`` 是 CausalLM 默认忽略的标签值，所以监督损失只覆盖答案部分。
    """
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding_id = tokenizer.pad_token_id
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        # 取一条样本。数据为 Alpaca 风格：instruction（指令）+ input（可选补充输入）+ output（答案）
        line = self.data[index]
        instruction_text = line['instruction']
        input_text = line['input']
        output_text = line['output']
        # query = 用户输入（指令 + 输入）；answer = 答案，末尾拼接 eos 标记“回答结束”
        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token
        # 下面用 chat 模板把 query 包成对话（只放 user 一轮）；add_generation_prompt=True 会自动补上
        # assistant 的起始标记，让模型处于“该开始生成回答”的状态
        messages = []
        messages.append({'role': 'user', 'content': query})   
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        
        # prompt 与 answer 分别编码后再拼接（分开编码，避免拼接处 token 边界被合并带来的偏差）
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        
        # 输入序列 = prompt + answer，整段一起喂给模型
        input_ids = prompt_input_ids + answer_input_ids
        # 标签：prompt 部分置为 -100（忽略，不参与损失），只保留 answer 的真实 id -> 只学“怎么回答”，不学“怎么提问”
        labels = [-100] * len(prompt_input_ids) + answer_input_ids
        # attention_mask：真实 token 为 1，稍后 padding 的位置补 0
        attention_mask = [1] * len(input_ids)
        text_len = len(input_ids)
        
        # 一个 batch 中的样本长度不一致，统一 padding 到 max_seq_len
        if text_len > self.max_seq_len:
            # 超长：从右侧截断（可能会截掉 answer 的尾部）
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
        else:
            # 不足：右侧补 pad_token，labels 补 -100，attention_mask 补 0（这些位置都不参与前向/损失）
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - text_len)
            labels = labels + [-100] * (self.max_seq_len - text_len)
            attention_mask = attention_mask + [0] * (self.max_seq_len - text_len)
        
        # 下面两行被注释掉，本意是做 next-token 预测的经典错位对齐：
        #   input_ids 去掉最后一个、labels 去掉第一个，使 labels[t] 成为 input_ids[t] 的下一个 token。
        # 但这里不需要：HuggingFace 的 CausalLM 在内部已经做了 logits[:-1] 与 labels[1:] 的对齐，
        # 所以数据集直接返回对齐好的 input_ids / labels 即可。
        # input_ids = input_ids[:-1]
        # labels = labels[1:]
        # 返回张量：input_ids（模型输入）、attention_mask（1=有效，0=padding）、labels（损失目标，-100 处忽略）
        return {'input_ids': torch.tensor(input_ids), 'attention_mask':torch.tensor(attention_mask), 'labels': torch.tensor(labels)}
    



class OnPolicyDataset(Dataset):
    """构造 on-policy rollout 所需的 prompt。

    这里使用左侧 padding，因为 decoder-only 模型批量生成时，每条 prompt
    最右侧都应是真实 token。例如 prompt 为 ``[21, 22, 23]``，
    ``max_prompt_length=5``，返回 ``[pad, pad, 21, 22, 23]``，对应 mask 为
    ``[0, 0, 1, 1, 1]``。JSON 中的 output 不会读取，回答由学生实时生成。
    """
    def __init__(self, data_path, tokenizer, args=None):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.args = args
        self.padding_id = tokenizer.pad_token_id
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        if self.args is None:
            self.max_prompt_length = 512
        else:
            self.max_prompt_length = self.args.max_prompt_length
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        line = self.data[index]
        instruction_text = line['instruction']
        input_text = line['input']

        query = instruction_text + input_text

        messages = []
        messages.append({'role': 'user', 'content': query})   
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        
        prompt_input_ids = self.tokenizer.encode(prompt)
        attention_mask = [1] * len(prompt_input_ids)
        if len(prompt_input_ids) > self.max_prompt_length:
            prompt_input_ids = prompt_input_ids[-self.max_prompt_length:]
            attention_mask = attention_mask[-self.max_prompt_length:]
            
        else:
            prompt_input_ids = [self.tokenizer.pad_token_id] * (self.max_prompt_length - len(prompt_input_ids)) + prompt_input_ids
            attention_mask = [0] * (self.max_prompt_length - len(attention_mask)) + attention_mask
            
        
        

        return {'input_ids': torch.tensor(prompt_input_ids), 'attention_mask':torch.tensor(attention_mask)}
