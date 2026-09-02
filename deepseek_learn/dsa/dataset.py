"""把 DeepCtrl JSONL 样本转换成 Qwen Chat SFT 所需的固定长度张量。

一行原始数据示例：

    {
        "instruction": "请简要回答：",
        "input": "1+1等于多少？",
        "output": "2。",
        "history": [["你好", "你好！"]]
    }

最终 ``input_ids`` 包含“历史对话 + 当前问题 + 当前答案”，但 ``labels`` 会把
历史对话和当前 Prompt 对应的位置设成 -100，只监督当前答案。
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
    """按行读取 JSONL，并在 ``__getitem__`` 中动态完成 Chat Template 编码。

    返回的三个张量长度始终等于 ``max_seq_len``：

    - ``input_ids``：模型实际读取的 Token；
    - ``labels``：Prompt/Padding 为 -100，只有当前答案 Token 参与 CE Loss；
    - ``attention_mask``：真实 Token 为 1，Padding 为 0。
    """

    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            # 数据集每一行都是独立 JSON；这里一次读入内存，__getitem__ 时再 json.loads。
            self.data = f.readlines()
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        """将第 ``index`` 行样本转换为模型输入。"""

        line = self.data[index]
        line = json.loads(line)
        instruction_text = line['instruction']
        input_text = line['input']
        output_text = line['output']
        history = line['history']

        # DeepCtrl 中 instruction 是任务说明，input 是当前用户输入；二者拼成当前问题。
        # 例："请翻译：" + "hello" -> "请翻译：hello"。
        query = instruction_text + input_text

        # 在答案末尾显式添加 EOS，让模型学习何时结束本轮回复。
        answer = output_text + self.tokenizer.eos_token
        messages = []
        if history:
            # history 的每个元素都是 [user_text, assistant_text]。
            # 例：["你好", "你好！"] 会展开为一条 user 消息和一条 assistant 消息。
            for i in history:
                messages.append({'role': 'user', 'content': i[0]})
                messages.append({'role': 'assistant', 'content': i[1]})
        
        # 当前轮只加入 user 消息；add_generation_prompt=True 会在结尾追加 assistant 起始标记。
        messages.append({'role': 'user', 'content': query})   
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

        # prompt_input_ids：历史对话 + 当前问题；answer_input_ids：当前答案 + EOS。
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        input_ids = prompt_input_ids + answer_input_ids

        # -100 是 PyTorch CrossEntropyLoss 默认的 ignore_index。
        # 例：prompt=[10, 11, 12]、answer=[20, 21]，则：
        # input_ids=[10, 11, 12, 20, 21]
        # labels=   [-100, -100, -100, 20, 21]
        labels = [-100] * len(prompt_input_ids) + answer_input_ids
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            # 超长样本从右侧截断，最终长度固定为 max_seq_len。
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attention_mask = [1] * self.max_seq_len
        else:
            # 短样本右侧补 Pad；Pad 对应 label=-100，因此既不被关注，也不计算 CE Loss。
            # 例：text_len=5、max_seq_len=8，则 attention_mask=[1,1,1,1,1,0,0,0]。
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - text_len)
            labels = labels + [-100] * (self.max_seq_len - text_len)
            attention_mask = [1] * text_len + [0] * (self.max_seq_len - text_len)
        
        # attention_mask = attention_mask[:-1]
        # input_ids = input_ids[:-1]
        # labels = labels[1:]
        # DefaultDataCollator 会把多个样本堆叠成 [batch_size, max_seq_len]。
        return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels), 'attention_mask': torch.tensor(attention_mask)}
    
        
            
            
