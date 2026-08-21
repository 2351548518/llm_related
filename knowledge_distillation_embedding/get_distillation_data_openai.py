"""通过 OpenAI-compatible Embeddings API 生成教师软标签。

该脚本与 get_distillation_data_local.py 生成相同的数据格式，区别只是教师
Embedding 来自 HTTP 服务。服务可以是 OpenAI API，也可以是实现了兼容接口的
本地 vLLM、Xinference 或其他推理服务。

例如输入一条包含两个负样本的数据，服务返回四个 embedding：
[query_embedding, positive_embedding, negative_1_embedding, negative_2_embedding]。
脚本再将它们转换成三个教师相似度分数。
"""

import openai
import torch
import torch.nn.functional as F
import json
import tqdm
import argparse
import os

def similarity(emb1: torch.Tensor, emb2: torch.Tensor):
    """沿最后一个维度计算余弦相似度，返回值范围理论上是 [-1, 1]。"""
    return F.cosine_similarity(emb1, emb2)

def get_embedding(client, text, model="Qwen3-Embedding-4B"):
    """批量请求文本向量。

    Args:
        client: ``openai.OpenAI`` 客户端。
        text: 字符串列表，例如 ``[query, positive, negative]``。
        model: Embeddings API 暴露的模型名称。

    Returns:
        API 响应中的 embedding data 列表，顺序与输入文本保持一致。
    """
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data

def parse_args():
    """定义 API 连接信息、教师模型名称以及输入输出路径。"""
    parser = argparse.ArgumentParser(description="Process embedding data for knowledge distillation")
    
    parser.add_argument("--base_url", type=str, default="http://0.0.0.0:8077/v1", 
                        help="OpenAI API base URL")
    parser.add_argument("--api_key", type=str, default="123", 
                        help="API key")
    parser.add_argument("--model", type=str, default="Qwen3-Embedding-4B", 
                        help="Embedding model name")
    
    parser.add_argument("--input_file", type=str, 
                        default="processed_data/processed_train_texts.json",
                        help="Input JSON file path")
    parser.add_argument("--output_file", type=str, 
                        default="train_data/train.json",
                        help="Output JSON file path")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 根据 output_file 自动创建父目录。
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # base_url 指向兼容 OpenAI /v1/embeddings 协议的服务。
    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    with open(args.input_file, "r", encoding='utf-8') as f:
        train_texts = json.load(f)
    
    train_datas = []
    
    for data in tqdm.tqdm(train_texts, total=len(train_texts)):
        q = data['query']
        pos = data['positive']
        neg = data['negative']
        # 请求文本顺序固定为 query 在前、positive 居中、negative 在后。
        if isinstance(neg, str):
            embeddings = get_embedding(client, [q, pos, neg], args.model)
        else:
            embeddings = get_embedding(client, [q, pos] + neg, args.model)
        
        # 把 SDK 返回对象转换为普通 Python 向量列表。
        embeddings = [emb.embedding for emb in embeddings]
        
        # 假设有 N 个负样本、向量维度为 H：query/positive 的形状为
        # [1, H]，negatives 的形状为 [N, H]。
        query_embedding = torch.tensor(embeddings[0], dtype=torch.float32).unsqueeze(0)
        pos_embedding = torch.tensor(embeddings[1], dtype=torch.float32).unsqueeze(0)
        neg_embedding = torch.tensor(embeddings[2:], dtype=torch.float32)
        
        # label[0] 是正样本分数，其余元素按原顺序对应各个负样本。
        pos_sim = similarity(query_embedding, pos_embedding)
        neg_sim = similarity(query_embedding, neg_embedding)
        sim = torch.cat([pos_sim, neg_sim], dim=0)
        label = sim.tolist()
        
        train_datas.append({
            "query": q, 
            "positive": pos, 
            "negative": neg, 
            "label": label
        })
    

    # 生成的 JSON 可直接传给 KGDataset。
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(train_datas, f, ensure_ascii=False, indent=4)
    
    print(f"Processing completed. Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
