"""使用本地 vLLM Embedding 模型生成蒸馏软标签。

脚本会让教师模型分别编码 query、positive 和 negative，然后计算 query 与
每个候选文本的余弦相似度。输出的相似度列表就是学生训练时使用的软标签。

输入示例：
    {"query": "q", "positive": "p", "negative": ["n1", "n2"]}

输出示例：
    {"query": "q", "positive": "p", "negative": ["n1", "n2"],
     "label": [0.87, 0.25, 0.11]}

label 的顺序始终是 [positive_score, negative_1_score, ...]。
"""

from vllm import LLM, SamplingParams
import torch
import torch.nn.functional as F
import json
import tqdm
import argparse
import os

def similarity(emb1: torch.Tensor, emb2: torch.Tensor):
    """计算两组向量的余弦相似度。

    例如 emb1.shape=[1, hidden_size]、emb2.shape=[10, hidden_size] 时，
    PyTorch 会广播 emb1，返回包含 10 个相似度的张量。
    """
    return F.cosine_similarity(emb1, emb2)

def parse_args():
    """定义教师模型、显存占用比例以及输入输出路径。"""
    parser = argparse.ArgumentParser(description="Process embedding data for knowledge distillation using vLLM")
    

    parser.add_argument("--model_path", type=str, 
                        default="Qwen3-Embedding-8B",
                        help="Path to the embedding model")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4,
                        help="GPU memory utilization ratio")
    

    parser.add_argument("--input_file", type=str, 
                        default="processed_data/processed_train_texts_negative_num_1.json",
                        help="Input JSON file path")
    parser.add_argument("--output_file", type=str, 
                        default="train_data/train_negative_num_1_8b.json",
                        help="Output JSON file path")
    

    parser.add_argument("--task", type=str, default="embed",
                        help="Task type for vLLM")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 教师输出目录不存在时自动创建。例如 output_file 为
    # train_data/train.json 时，这里会创建 train_data。
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    print(f"Loading model from {args.model_path}...")
    llm = LLM(
        model=args.model_path, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        task=args.task
    )
    

    print(f"Reading data from {args.input_file}...")
    with open(args.input_file, "r", encoding='utf-8') as f:
        train_texts = json.load(f)
    
    train_datas = []
    
    # 当前实现逐条处理训练样本；每次 embed 调用内部会同时编码当前样本的
    # query、positive 和所有 negative。
    print("Processing embeddings...")
    for data in tqdm.tqdm(train_texts, total=len(train_texts)):

        # neg 既可能是单个字符串，也可能是字符串列表。
        q = data['query']
        pos = data['positive']
        neg = data['negative']
        
        # 输入顺序非常重要。例如 neg=[n1, n2] 时，编码顺序是
        # [q, pos, n1, n2]，后面生成的 label 会沿用这一候选顺序。
        if isinstance(neg, str):
            outputs = llm.embed([q, pos, neg], use_tqdm=False)
        else:
            outputs = llm.embed([q, pos] + neg, use_tqdm=False)
        embeddings = [output.outputs.embedding for output in outputs]
        
        # 假设 embedding 维度为 H，且有 N 个负样本：
        # query_embedding.shape = [1, H]
        # pos_embedding.shape   = [1, H]
        # neg_embedding.shape   = [N, H]
        query_embedding = torch.tensor(embeddings[0], dtype=torch.float32).unsqueeze(0)
        pos_embedding = torch.tensor(embeddings[1], dtype=torch.float32).unsqueeze(0)
        neg_embedding = torch.tensor(embeddings[2:], dtype=torch.float32)
        
        # pos_sim.shape=[1]，neg_sim.shape=[N]；拼接后 label.shape=[1+N]。
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
    
    # 保存带有教师软标签的数据，供 dataset.py 和 train.py 使用。
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(train_datas, f, ensure_ascii=False, indent=4)
    
    print(f"Processing completed. Processed {len(train_datas)} samples. Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
            

            
            
            
