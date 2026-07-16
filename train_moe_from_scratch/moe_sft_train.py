# ============================================================
# MoE 模型的 SFT (Supervised Fine-Tuning, 监督微调) 入口
# 流程:
#   1. 注册自定义的 Config / LLM，使 AutoXxx 能识别 model_type="moe_model"
#   2. 从 ./saves/moe 加载预训练好的 MoE 权重
#   3. 用 SFTDataset 读 sft.jsonl，对"回答部分"计算 loss
#   4. HF Trainer 训练 5 个 epoch，保存到 ./saves/sft
# 与预训练的区别: 不从随机初始化训练，而是在预训练模型上继续做指令微调，
# 让模型学会按对话格式回答问题。
# ============================================================

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
from dataset import SFTDataset, LLMDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
# 复用预训练文件里定义的模型类与配置类，避免重复实现
from moe_train import LLM, Config

if __name__ == '__main__':
    # 注册自定义配置与模型，使 HF 的 AutoConfig/AutoModelForCausalLM
    # 在 from_pretrained 时能按 "moe_model" 找到对应的 Config/LLM 类
    AutoConfig.register("moe_model", Config)
    AutoModelForCausalLM.register(Config, LLM)
    # 加载预训练 (moe_train.py 训出的) 模型作为 SFT 的初始化权重
    model = AutoModelForCausalLM.from_pretrained('./saves/moe')
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./sft',
                            num_train_epochs=5,
                            do_train=True,
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=1,
                            # max_steps=15000,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,                # bfloat16 混合精度
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False)
    # SFT 数据集: 用 chat_template 拼 prompt+answer，prompt 部分标签置 0 (只对回答算 loss)
    dataset = SFTDataset('./sft.jsonl', tokenizer=tokenizer, max_seq_len=1024)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/sft')
    trainer.save_state()
