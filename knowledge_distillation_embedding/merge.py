"""把训练得到的 LoRA adapter 合并到学生基础模型中。

训练结束后，``saves_lora_8b_negative_1`` 主要保存的是 LoRA 增量权重，不能完全
脱离基础模型使用。本脚本执行的关系可以理解为：

    Qwen3-Embedding-0.6B 基础权重 + LoRA 增量权重 -> 合并后的完整模型

合并结果写入 ``merged_model/Qwen3-Embedding-0.6B``，供 evaluation.py 加载。
README 还说明了 Sentence Transformers 配置文件需要从原始模型目录复制过来，
以确保评估时继续使用 last-token pooling。
"""

from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


# 重新加载与训练时相同的学生基础模型及 tokenizer。
model = AutoModel.from_pretrained("Qwen3-Embedding-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen3-Embedding-0.6B")

# trainer.save_model 保存的 LoRA adapter 目录。
lora_path = 'saves_lora_8b_negative_1'

# 将 adapter 挂载到基础模型，再把低秩增量实际加回基础权重。
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# 保存后无需 PEFT，也可以按普通 Transformers 模型加载。
model.save_pretrained("merged_model/Qwen3-Embedding-0.6B")
tokenizer.save_pretrained("merged_model/Qwen3-Embedding-0.6B")
