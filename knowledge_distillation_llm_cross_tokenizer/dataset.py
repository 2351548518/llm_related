"""训练数据读取与组批逻辑。

这个文件只负责保留原始 ``prompt``/``answer`` 的边界，不负责构造最终送入
模型的 chat template。学生和教师的最终输入会在 ``KGTrainer`` 中分别构造，
因为两者的 tokenizer 和 chat template 可能完全不同。
"""

import torch
from torch.utils.data import IterableDataset, Dataset
import json


class SFTDataset(Dataset):
    """读取 JSON 格式的监督微调数据。

    数据格式示例::

        [
            {"prompt": "你是谁？", "answer": "我是一个AI助手。"}
        ]

    此处暂时使用学生 tokenizer 编码，目的是让 DataLoader 能通过
    ``len(input_ids) - len(labels)`` 恢复 prompt 的结束位置。后续训练时会先
    解码回文本，再分别使用学生、教师 tokenizer 重新编码。
    """

    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        """返回一条尚未 padding 的样本。

        假设 prompt_ids=[10, 11]、answer_ids=[20, 21, 22]，则返回::

            input_ids = [10, 11, 20, 21, 22]
            labels    = [20, 21, 22]

        注意：这里的 ``labels`` 不是与 ``input_ids`` 等长的标准 CausalLM
        labels；它只用于保存答案 token 以及计算 prompt/answer 的分界点。
        """

        line = self.data[index]
        prompt = line['prompt']
        answer = line['answer']

        # 不添加 BOS/EOS 等特殊 token；最终的特殊 token 由各模型自己的
        # chat template 在 KGTrainer.get_inputs_from_texts 中生成。
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        input_ids = prompt_ids + answer_ids
        labels = answer_ids

        return {'input_ids': input_ids, 'labels': labels}


class MyDataCollator:
    """将多条样本整理为“列表的列表”，但暂时不做 padding。

    例如，两条不同长度的样本会得到::

        {
            "input_ids": [[10, 11, 20], [12, 30, 31]],
            "labels": [[20], [30, 31]],
        }

    保留原始长度后，``KGTrainer.compute_loss`` 才能准确切出 prompt。真正的
    tensor 化和 padding 会在学生、教师各自完成重新分词之后进行。
    """

    def __init__(self):
        pass

    def __call__(self, features):
        # 此处故意不使用 transformers.DefaultDataCollator：它要求同一字段的
        # 序列可以直接堆叠，而本项目需要先保留每条样本的原始可变长度。
        input_ids = [feature['input_ids'] for feature in features]
        labels = [feature['labels'] for feature in features]

        return {'input_ids': input_ids, 'labels': labels}
