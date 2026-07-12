from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer, AutoProcessor
from transformers import ViTImageProcessor, ViTForImageClassification

import torch.nn as nn
from transformers.utils import ModelOutput
import torch
import torch.nn.functional as F
from dataclasses import dataclass

# =============================================================================
# 模型输出容器:把 forward 的所有产出打包成一个对象返回,
# 训练时 Trainer 会自动取其中的 `loss` 字段反传,其余字段可用于推理/监控。
# 例如 test.ipynb 里就用到 outputs.logits_per_image。
# =============================================================================
@dataclass
class SiglipOutput(ModelOutput):
    loss: torch.FloatTensor = None              # []      标量,SigLIP 逐元素 sigmoid 损失
    logits_per_text: torch.FloatTensor = None  # [N, N]  文本视角相似度矩阵(= text@image.T * exp(t) + b)
    logits_per_image: torch.FloatTensor = None # [N, N]  图像视角相似度矩阵(= logits_per_text 的转置)
    text_embeds: torch.FloatTensor = None      # [N, d]  L2 归一化后的文本 embedding
    image_embeds: torch.FloatTensor = None     # [N, d]  L2 归一化后的图像 embedding



# =============================================================================
# 配置类:只保存两个骨干网络的路径。
# 用 PreTrainedConfig 是为了能 save_pretrained / from_pretrained,配合 AutoModel。
# =============================================================================
class SiglipConfig(PretrainedConfig):
    model_type = "siglip"   # 注册模型类型,save_pretrained 后可用 AutoModel 按 type 自动加载
    def __init__(
        self,
        vision_model_name_or_path: str = "vit-base-patch16-224",  # 视觉塔:ViT
        text_model_name_or_path: str = "bert-base-chinese",       # 文本塔:中文 BERT/RoBERTa
        **kwargs):
        super().__init__(**kwargs)
        self.vision_model_name_or_path = vision_model_name_or_path
        self.text_model_name_or_path = text_model_name_or_path



# =============================================================================
# SigLIP 双塔模型
#   图像塔:ViT-base      →  pooler_output([CLS])  →  L2 norm → image_embed  [N, 768]
#   文本塔:chinese-RoBERTa →  pooler_output([CLS]) →  L2 norm → text_embed  [N, 768]
#   损失:逐元素 sigmoid 损失(对角+1/非对角-1),区别于 CLIP 的 softmax InfoNCE
# =============================================================================
class SiglipModel(PreTrainedModel):
    config_class = SiglipConfig
    def __init__(self, config: SiglipConfig):
        super().__init__(config)
        # 视觉塔 & 文本塔:直接加载各自预训练权重,从它们的 pooler_output 取 [CLS] 表示
        #   vision_model 输入 [N,3,224,224] → pooler_output [N,768]
        #   text_model   输入 [N,64]        → pooler_output [N,768]
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name_or_path)
        self.text_model = AutoModel.from_pretrained(config.text_model_name_or_path)

        # 注意:这两个不是 nn.Module,不会进 state_dict / save_pretrained,
        # 推理时(test.ipynb)需另行从本地路径加载 processor/tokenizer。存这里只是为了方便。
        self.process = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)

        # SigLIP 相对 CLIP 多出的两个可学习标量(均为标量参数,形状 [1]):
        #   t: log-temperature,forward 里取 exp(t) 作为相似度的缩放(温度越大,logit 越陡)
        #   b: 偏置 bias,logit = sim * exp(t) + b
        self.t = nn.Parameter(torch.randn(1))   # [1]
        self.b = nn.Parameter(torch.randn(1))   # [1]


    def forward(self, input_ids, attention_mask, pixel_values):
        """
        输入(一个 batch,以 N=2 为例):
            input_ids:      [N, 64]            文本 token id        (padding 到 max_length=64)
            attention_mask: [N, 64]            文本 padding 掩码
            pixel_values:   [N, 3, 224, 224]   ViT 预处理后的图像张量

        输出:见 SiglipOutput(loss [], logits_per_text [N,N], logits_per_image [N,N],
                            text_embeds [N,768], image_embeds [N,768])
        """
        # ---- 1. 双塔前向,取 pooler_output(索引 [1])----
        # BERT/ViT 的输出元组:[0]=last_hidden_state(序列), [1]=pooler_output([CLS] 的汇总表示)
        text_outputs = self.text_model(input_ids, attention_mask)
        vision_outputs = self.vision_model(pixel_values)

        vision_features = vision_outputs[1] # [N, 768] 图像 [CLS] 池化表示
        text_features = text_outputs[1]     # [N, 768] 文本 [CLS] 池化表示

        # ---- 2. L2 归一化:让 text@image.T 直接等于余弦相似度,落在 [-1, 1] ----
        # 这样温度 exp(t) 才能稳定地缩放 logit 量级。CLIP 也做同样归一化。
        # norm(p=2, dim=-1, keepdim=True) → [N, 1],可广播除法,结果仍 [N, 768]
        vision_features = vision_features / vision_features.norm(p=2, dim=-1, keepdim=True) # l2标准化 → [N,768]
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)       # l2标准化 → [N,768]

        # ---- 3. 相似度矩阵 + 温度 + 偏置 ----
        # logits_per_text[i,j] = (文本i 与 图像j 的余弦相似度) * exp(t) + b
        # matmul: [N,768] @ [768,N] → [N, N];标量 exp(t)、b 广播
        # 例 N=2:
        #   logits_per_text = [[z00, z01],
        #                      [z10, z11]]   对角线 z00,z11 是配对的正样本
        logits_per_text = torch.matmul(text_features, vision_features.t()) * self.t.exp() + self.b  # [N, N]
        logits_per_image = logits_per_text.t()  # [N, N] 转置即图像视角

        # ---- 4. SigLIP 损失:把 N×N 个 (图,文) 对当成 N×N 个独立二分类 ----
        # 标签矩阵:对角线 +1(配对),非对角线 -1(不配对)。即 2*I - 1。
        # 例 N=2: labels = [[+1, -1],
        #                   [-1, +1]]
        b = logits_per_text.shape[0]                                          # 标量 N
        eye = torch.eye(b, device=logits_per_text.device) # 生成单位矩阵     # [N, N]
        labels = 2*eye - torch.ones_like(logits_per_text, device=logits_per_text.device) # [N, N] 对角线全为1，非对角线为-1，即成对的图文标签为1，非成对的为-1

        # logsigmoid(y * z):当 y=+1 且 z 大 → 接近 0(好);当 y=-1 且 z 负 → 接近 0(好)。
        # 即"正对要大、负对要小",loss 才低。这是逐格独立的,不像 CLIP 要跨行 softmax。
        # labels * logits_per_text: [N,N] 逐元素相乘 → logsigmoid 逐元素 → 仍 [N, N]
        loglik = F.logsigmoid(labels * logits_per_text)  # [N, N]

        # 内层对一行内所有 j 求和(对应公式里的 ∑_j, dim=-1 压掉最后一维),再对 batch 求平均(对应 1/N)。
        nll = -torch.sum(loglik, dim=-1)  # [N]
        loss = nll.mean()                 # [] 标量


        return SiglipOutput(loss=loss, logits_per_text=logits_per_text, logits_per_image=logits_per_image, text_embeds=text_features, image_embeds=vision_features)
