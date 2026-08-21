"""把原始检索数据整理成教师模型可以打分的候选集合。

原始数据中的一条记录通常包含一个 query、多个 positive 和多个 negative，例如：

    {
        "query": "What is knowledge distillation?",
        "positive": ["Knowledge distillation transfers knowledge ..."],
        "negative": ["A paper about image segmentation", "A database tutorial"]
    }

处理后，每个 positive 都会展开成一条独立样本，并抽取指定数量的负样本。
这些样本还没有教师分数，后续会交给 get_distillation_data_*.py 处理。
"""

from datasets import load_dataset
import random
import json
import os


def process_data(data_path, output_path, split='train', negative_num=10):
    """抽取正负候选，生成蒸馏前的 JSON 数据。

    Args:
        data_path: Hugging Face Dataset 的名称或本地数据集目录，例如 ``origin_data``。
        output_path: 输出目录，例如 ``processed_data``。该目录需要提前存在。
        split: 要处理的数据划分，默认是 ``train``。
        negative_num: 每个正样本搭配的负样本数量。

    输出示例（negative_num=2）：

        {
            "query": "What is knowledge distillation?",
            "positive": "Knowledge distillation transfers knowledge ...",
            "negative": ["negative document A", "negative document B"]
        }
    """
    datas = []
    dataset = load_dataset(data_path)
 
    data = dataset[split]
    for i in data:
        query = i['query']
        positive = i['positive']
        negative = i['negative']
        # 一条原始记录可能包含多个正确文档。每个正确文档分别构造一条训练样本，
        # 从而确保后续每条样本都满足：1 个 query + 1 个 positive + N 个 negative。
        for pos in positive:
            # 负样本足够时进行无放回抽样。例如从 20 条负样本中抽取 10 条，
            # 同一条负样本不会在当前训练样本里重复出现。
            if len(negative) >= negative_num:
                neg = random.sample(negative, negative_num)
            else:
                # 负样本不足时进行有放回抽样。例如只有 2 条负样本但需要 10 条，
                # 结果中允许同一条负样本重复出现。
                neg = random.choices(negative, k=negative_num)
            
            datas.append({'query': query, 'positive': pos, 'negative': neg})
    
    # 文件名记录负样本数量，便于区分 negative_num=1、10 等不同实验。
    with open(os.path.join(output_path, f'processed_{split}_texts_negative_num_{negative_num}.json'), 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    # 默认读取 Hugging Face 上的 SciDocs 数据，并为每个正样本搭配 1 个负样本。
    # Windows 路径使用原始字符串 r'...'，避免反斜杠被解释成转义字符。
    process_data('Bibek/scidocs-reranking-train', r'/data2/home/jiapeng2/code/LLM/llm_related/knowledge_distillation_embedding/processed_data', negative_num=1)
               
            
            
    
