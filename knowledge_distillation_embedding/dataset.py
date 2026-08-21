"""训练数据集：把文本候选和教师分数转换为模型输入张量。

一条蒸馏数据的结构如下：

    {
        "query": "query text",
        "positive": "relevant document",
        "negative": ["irrelevant document 1", "irrelevant document 2"],
        "label": [0.82, 0.21, 0.08]
    }

``label`` 中的元素依次对应 positive 和各个 negative 的教师余弦相似度。
"""

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from torch.utils.data import IterableDataset, Dataset
import json


class KGDataset(Dataset):
    """供 Transformers Trainer 使用的 Map-style Dataset。

    Args:
        data_path: 已经包含教师 ``label`` 的 JSON 文件路径。
        tokenizer: 学生模型的 tokenizer。
        max_seq_len: 单段文本的最大 token 长度。
    """

    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # 当前实现一次性把整个 JSON 文件加载进内存。
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        """返回一组 query/positive/negative 的 token，以及教师分数。

        例如有 2 个负样本且 max_seq_len=512 时：

        - input_texts 的顺序是 [query, positive, negative_1, negative_2]
        - input_ids 的形状是 [4, 512]
        - labels 的形状是 [3]，对应 [positive, negative_1, negative_2]

        DataLoader 再将多个样本堆叠为
        [batch_size, sample_num, sequence_length]。
        """
        sample = self.data[index]
        # 同时兼容 negative 为单个字符串或字符串列表的两种数据格式。
        if isinstance(sample['negative'], str):
            input_texts = [sample['query']] + [sample['positive']] + [sample['negative']]
        else:
            input_texts = [sample['query']] + [sample['positive']] + sample['negative']
        # 每段文本都固定 padding 到 max_seq_len，因此一个样本中的文本可以直接堆叠。
        batch_dict = self.tokenizer(
            input_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )   
        # input_ids.shape = [sample_num, seq_len]
        # sample_num = 1(query) + 1(positive) + negative_num

        # labels 不是类别编号，而是教师模型产生的浮点相似度列表。
        batch_dict['labels'] = torch.tensor(sample['label'])
        return batch_dict
