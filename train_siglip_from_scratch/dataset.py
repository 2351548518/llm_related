from torch.utils.data import Dataset
import json
from PIL import Image
import torch
import pandas as pd
from io import BytesIO
import base64
from transformers import AutoTokenizer, AutoProcessor
import random

# =============================================================================
# 数据格式说明(MUGE 电商多模态数据集):
#   all_texts.jsonl  : 每行 {"text_id", "text", "image_ids":[...]} —— 一段文本可对应多张图
#   all_imgs.tsv     : 两列 image_id \t base64图片。用 pandas 当查表用。
#
# 例:
#   {"text_id":271831, "text":"女包", "image_ids":[117544, 1036646, ...]}
#   会被展开成多条样本: (text="女包", image_id=117544), (text="女包", image_id=1036646) ...
#   一段文本配多图 → 拆成多个独立图文对训练。
# =============================================================================
class SiglipDataset(Dataset):
    def __init__(self, text_data_path,
                 image_data_path,
                 tokenizer,
                 processor,
                 max_seq_length=64,        # 文本统一 padding 到 64;推理时也必须用 max_length(见 test.ipynb)
                 ):
        super().__init__()
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = max_seq_length

        # ---- 1. 读文本 jsonl,把"一段文本 + 多图 id"展开成多个 (text, image_id) 样本 ----
        # self.datas 形状:list[dict],长度 = 文本条数 × 平均图数
        #   每个元素 {'image_id':int, 'text':str}
        with open(self.text_data_path, 'r', encoding='utf-8') as f:
            self.datas = []
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                for image_id in line['image_ids']:
                    self.datas.append({'image_id': image_id, 'text': line['text']})

        # 打乱顺序,保证每个 batch 里的图文对尽量多样化(同一段文本的多个图被分散)
        random.shuffle(self.datas)

        # ---- 2. 图像表:两列 [0]=image_id, [1]=base64 字符串。只读一次常驻内存,按 id 查 ----
        # self.images 形状:DataFrame [num_imgs, 2]
        self.images = pd.read_csv(self.image_data_path, sep='\t', header=None)

    def __getitem__(self, index):
        '''
        取第 index 条样本,返回一个 dict(未做 batch 维度对齐,交给 MyDataCollator 处理)。

        数据 shape 变化:
            text(str) ──tokenizer(max_length=64, padding='max_length')──→
                input_ids: list[int]      长度 64
                attention_mask: list[int] 长度 64
            image_id(int) ──查表──→ base64(str) ──b64decode──→ bytes ──PIL──→ PIL.Image(H,W,3)
                ──processor(return_tensors='pt')──→ pixel_values: [1, 3, 224, 224]
        '''
        sample = self.datas[index]

        image_id = sample['image_id']
        text = sample['text']

        # 文本 tokenize:padding='max_length' 保证全 batch 等长,无需 collator 再动态 pad
        # str → {'input_ids': list[64], 'attention_mask': list[64]}
        tok = self.tokenizer(text, max_length=self.max_seq_length, padding='max_length', truncation=True)
        input_ids = tok['input_ids']        # list[int], 长度 64
        attention_mask = tok['attention_mask']  # list[int], 长度 64

        # 图像:按 image_id 在表里查 base64 → 解码字节 → PIL → ViT processor
        # str → bytes → PIL.Image(H,W,3) → tensor[1,3,224,224]
        image_base64 = self.images[self.images[0]==image_id][1].values[0]   # str
        image_bytes = base64.b64decode(image_base64)                        # bytes


        image = Image.open(BytesIO(image_bytes)).convert("RGB")             # PIL.Image, H×W×3
        # processor 内部:resize/crop 到 224 + 归一化;return_tensors='pt' 加上第 0 维 → [1,3,224,224]
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values']  # [1, 3, 224, 224]

        return {
            'input_ids': input_ids,           # list[int]      len=64
            'attention_mask': attention_mask, # list[int]      len=64
            'pixel_values': pixel_values      # tensor[1,3,224,224]
        }

    def __len__(self):
        return len(self.datas)

# =============================================================================
# 自定义 collator:把 __getitem__ 返回的若干单样本 dict 拼成一个 batch tensor。
#
# 数据 shape 变化:
#   features: list[dict],长度 = batch_size B
#       每个元素: input_ids list[64], attention_mask list[64], pixel_values tensor[1,3,224,224]
#
#   input_ids:      list[list[int]]  (B×64)  ─torch.tensor─→ [B, 64]
#   attention_mask: list[list[int]]  (B×64)  ─torch.tensor─→ [B, 64]
#   pixel_values:   list[[1,3,224,224]]      ─torch.cat(dim=0)─→ [B, 3, 224, 224]
# =============================================================================
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [f['input_ids'] for f in features]        # list[B] of list[64]
        attention_mask = [f['attention_mask'] for f in features]  # list[B] of list[64]
        pixel_values = [f['pixel_values'] for f in features] # list[B] of [1,3,224,224]
        return {
            'input_ids': torch.tensor(input_ids),          # [B, 64]
            'attention_mask': torch.tensor(attention_mask),# [B, 64]
            'pixel_values': torch.cat(pixel_values, dim=0) # [B, 3, 224, 224]
        }


if __name__ == '__main__':
    # 用法示例:加载一条数据看看形状
    tokenizer = AutoTokenizer.from_pretrained('/home/user/wyf/chinese-roberta-wwm-ext')
    processor = AutoProcessor.from_pretrained('/home/user/wyf/train_siglip_from_scratch/vit-base-patch16-224')

    dataset = SiglipDataset(text_data_path='/home/user/wyf/train_siglip_from_scratch/MUGE/all_texts.jsonl',
                            image_data_path='/home/user/wyf/train_siglip_from_scratch/MUGE/all_imgs.tsv',
                            tokenizer=tokenizer,
                            processor=processor,
                            max_seq_length=64)

    print(len(dataset))     # 样本总数(文本展开后的图文对数)
    print(dataset[2])       # 看第 2 条样本的 dict
