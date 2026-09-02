"""从中文 SFT 数据中筛选长序列，生成 Indexer 预热数据。

筛选标准是 Chat Template 编码后的总长度至少为 1024 Token。例如长度 800 的样本
不会写入，长度 1500 的样本会原样写入 ``warmup_data.jsonl``。选择长序列是因为
当序列长度超过 Top-K=128 后，Indexer 才需要真正区分“保留哪些历史 Token”。
"""

import json
import tqdm
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct")

# 输入文件来自 README 中的 DeepCtrl 中文 SFT 数据集。
with open('sft_data_zh.jsonl', 'r', encoding='utf-8') as f:
    # 使用追加模式分批写入，每累计 100 条长样本写盘一次。
    with open('warmup_data.jsonl', 'a', encoding='utf-8') as fw:
        
        warmup_data = []
        for line in tqdm.tqdm(f):
            # raw_line 保留原始 JSONL 文本，筛选完成后无需重新序列化。
            raw_line = line
            line = json.loads(line)
            instruction_text = line['instruction']
            input_text = line['input']
            output_text = line['output']
            history = line['history']
            query = instruction_text + input_text
            answer = output_text + tokenizer.eos_token
            messages = []
            if history:
                # 把多轮 history 展开成 Qwen Chat Template 所需的 role/content 格式。
                for i in history:
                    messages.append({'role': 'user', 'content': i[0]})
                    messages.append({'role': 'assistant', 'content': i[1]})
            
            messages.append({'role': 'user', 'content': query})   
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            prompt_input_ids = tokenizer.encode(prompt)
            answer_input_ids = tokenizer.encode(answer)
            input_ids = prompt_input_ids + answer_input_ids

            # 例：prompt 900 Token + answer 150 Token = 1050 Token，因此保留。
            if len(input_ids) >= 1024:
                warmup_data.append(raw_line)
            
            # if len(warmup_data) == 8:
            #     break
            if len(warmup_data) == 100:
                # 分批写入，避免为大量长样本长期保留额外内存。
                fw.writelines(warmup_data)
                warmup_data = []

        # 注意：当前实现没有在循环结束后 flush warmup_data，因此最后不足 100 条的部分不会写入。
