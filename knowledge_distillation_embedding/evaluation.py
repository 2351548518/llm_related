"""在验证集上评估蒸馏后的 Embedding 模型。

RerankingEvaluator 会让模型编码 query、positive 和 negative，并按照余弦相似度
对候选文档重新排序。主要指标包括：

- MAP：综合衡量所有相关文档在排序列表中的位置。
- MRR@10：前 10 个结果中，第一个相关文档排名倒数的平均值。
- NDCG@10：考虑前 10 个结果的位置折损，越相关的文档排得越靠前越好。

例如第一个相关文档排名第 2，则该 query 的 reciprocal rank 为 1/2。
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import RerankingEvaluator
from datasets import load_dataset


def get_eval_data(path):
    """把 validation split 转成 RerankingEvaluator 所需的数据格式。

    单条样本示例：

        {
            "query": "query text",
            "positive": ["relevant document"],
            "negative": ["irrelevant document 1", "irrelevant document 2"]
        }
    """
    samples = []
    eval_dataset = load_dataset(path)['validation']
    for i in eval_dataset:
        samples.append({'query': i['query'], 'positive': i['positive'], 'negative': i['negative']})
        
    return samples
    

# evaluator 内部负责批量编码候选、计算相似度以及汇总排序指标。
samples = get_eval_data('origin_data')
evaluator = RerankingEvaluator(samples, show_progress_bar=True)




# print('Loading original model...')
# original_model = SentenceTransformer('/home/user/Downloads/Qwen3-Embedding-0.6B')
# original_model.cuda()
# original_model.eval()
print('Loading distillation model...')
# 这里加载 merge.py 生成的完整模型目录。
distillation_model = SentenceTransformer('merged_model/Qwen3-Embedding-0.6B')
# 将模型移动到默认 CUDA 设备，并切换到推理模式。
distillation_model.cuda()
distillation_model.eval()
# print('Loading teacher model...')
# teacher_model = SentenceTransformer('/home/user/Downloads/Qwen3-Embedding-8B')
# teacher_model.cuda()
# teacher_model.eval()

# print('Evaluating original model...')
# original_result = evaluator(original_model)
# print(original_result)
print('Evaluating distillation model...')
# 返回字典中包含 MAP、MRR@10、NDCG@10 等评估结果。
distillation_result = evaluator(distillation_model)
print(distillation_result)
# print('Evaluating teacher model...')
# teacher_result = evaluator(teacher_model)
# print(teacher_result)

    
    

