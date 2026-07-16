# ============================================================
# 从零实现一个 MoE (Mixture of Experts) 大语言模型
# 架构参考 LLaMA / DeepSeek-Mo：
#   - RMSNorm + RoPE 旋转位置编码
#   - GQA 分组注意力的 KV cache 推理
#   - SwiGLU 前馈网络 (MLP / Expert)
#   - 稀疏门控 MoE：每 token 由 top-k 个专家处理
#   - 密集层 / MoE 层 交替 (偶数层 dense，奇数层 MoE)
#   - 负载均衡辅助损失 (Switch Transformer 风格)
#   - embedding 与 output 权重 tying
# 本文件包含模型定义 (LLM/Config/DecoderLayer/...) 与预训练入口。
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


class RMSNorm(nn.Module):
    """RMS 归一化 (LLaMA 风格)。
    与 LayerNorm 的区别：不减均值，只用均方根做缩放，计算更省、效果相当。
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 可学习的缩放权重 (gamma)，初始化为 1
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 转成 float32 计算，提升数值稳定性 (bf16 下尤其重要)
        hidden_states = hidden_states.float()
        # 沿最后一维求均方 (即 E[x^2])，keepdim 以便广播
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # x / sqrt(E[x^2] + eps)，等价于 RSqrt(x^2 的均值)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 乘以可学习权重后再转回原 dtype
        return self.weight * hidden_states.float()

def rotate_half(x):
    """把张量最后一维对半切分，构造旋转所需的 (-x2, x1)。
    用于 RoPE 中 q*cos + rotate_half(q)*sin 的计算。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """对 q、k 应用旋转位置编码。
    q_embed = q*cos + rotate_half(q)*sin，让内积结果只依赖于相对位置。
    unsqueeze_dim=2 用于把 cos/sin 广播到 head 维度。
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q*cos) + (rotate_half(q)*sin)
    k_embed = (k*cos) + (rotate_half(k)*sin)

    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码。
    预计算 max_seq_len 长度的 cos/sin 缓存，forward 时按当前序列长度切片使用。
    """
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        # 逆频率：1 / 10000^(2i/d)，i 从 0 到 d/2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 位置序列 t (max_seq_len, 1) 与 inv_freq (1, d/2) 矩阵乘 -> (max_seq_len, d/2)
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        # 在最后一维拼一份得到 (max_seq_len, d)，与 q/k 的 head_dim 对齐
        freqs = torch.cat((freqs, freqs), dim=-1)

        # 缓存 cos/sin，作为不参与训练的 buffer
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        # 按当前序列长度 q.shape[1] 切片，并加一个 batch 维
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)

def repeat_kv(hidden_states, n_rep):
    """GQA 中把 KV head 复制 n_rep 份，使其数量与 Q head 对齐。
    例如 8 个 KV head 复制 2 份得到 16 个 Q head。
    n_rep==1 时直接返回，避免无谓拷贝。
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 插入一个 n_rep 维度并 expand，再 reshape 合并到 head 维
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class Attention(nn.Module):
    """分组注意力 (GQA)。
    num_heads 个 Q head，num_key_value_heads 个 KV head，靠 repeat_kv 扩展。
    支持两种实现：flash_attn=True 用 PyTorch SDPA，否则手动实现 + 显式 causal mask。
    推理时可开启 use_kv_cache 做增量解码。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # head_dim 默认为 hidden_size // num_heads，可在 config 中覆盖
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        # Q head 是 KV head 的多少倍 (用于 repeat_kv)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # KV cache：推理时缓存历史 K/V，逐 token 增量解码
        self.k_cache, self.v_cache = None, None
        self.is_causal = True
        self.flash_attn = self.config.flash_attn

        # Q 投影到 num_heads 个 head；K/V 只投影到 num_key_value_heads 个 head (更省显存)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # 输出投影把多头拼接后的结果映射回 hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, use_kv_cache=False):
        b, s = hidden_states.shape[:2]
        # ---- KV cache 增量解码分支 ----
        if use_kv_cache and self.eval():
            # 首次调用 (k_cache 为空) 或序列长度不匹配时，重新计算整段的 K/V
            if self.k_cache is None or self.k_cache.shape[1] != s-1:
                q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            else:
                # 只取最后一个新 token 投影成 q/k/v，与历史 K/V 拼接
                token = hidden_states[:, -1:, :]
                # 历史位置的 q 置零 (生成时只关心最后一个位置的输出)
                q = torch.cat((torch.zeros_like(hidden_states[:, :-1, :]), self.q_proj(token)), dim=1)
                k = torch.cat((self.k_cache, self.k_proj(token)), dim=1)
                v = torch.cat((self.v_cache, self.v_proj(token)), dim=1)
            # 更新缓存
            self.k_cache, self.v_cache = k, v

        else:
            # ---- 训练 / 全量前向分支 ----
            q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        # 拆成多头: (b, s, num_heads, head_dim)
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_key_value_heads, self.head_dim)
        v = v.view(b, s, self.num_key_value_heads, self.head_dim)

        # 给 q、k 注入旋转位置编码
        q, k = self.rotary_emb(q, k)

        # GQA：把 KV head 复制成与 Q head 相同数量
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # (b, s, n, d) -> (b, n, s, d)，把 head 维提到前面以便批量矩阵乘
        q = q.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        k = k.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        v = v.transpose(1, 2) # b, self.num_heads, s, self.head_dim

        if self.flash_attn:
            # 优先使用 PyTorch 内置 SDPA：融合算子 + 显存友好 + 自动 causal
            # 计算过程: softmax(q*k/sqrt(d)) * v -> (b, n, s, head_dim)
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                    dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=self.is_causal)
        else:
            # 手动实现注意力 (含显式上三角 causal mask)
            mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"))  # 初始化掩码
            mask = torch.triu(mask, diagonal=1)  # 生成上三角掩码
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
            scores = scores + self.mask[:, :, :s, :s]  # 应用掩码 (注: 此处应为 mask，原作者笔误，flash_attn 默认开启时不会走到这里)
            scores = F.softmax(scores.float(), dim=-1).type_as(q)  # 计算 softmax
            scores = self.attention_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, v)  # 计算输出

        # (b, n, s, d) -> (b, s, n*d) 再合并多头
        output = output.transpose(1, 2).contiguous().view(b, s, -1) # b, s, self.hidden_size

        # 输出投影 + 残差 dropout
        output = self.o_proj(output)
        output = self.residual_dropout(output)
        return output


class MLP(nn.Module):
    """密集前馈网络 (SwiGLU)。
    用于偶数层 (非 MoE 层)。
    down_proj( silu(gate_proj(x)) * up_proj(x) )
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # gate/up 投影到 intermediate 维，down 投影回 hidden 维
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        # SwiGLU: 先 gate/up 各升维，逐元素相乘后再 down 降维
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def load_balancing_loss_func(
    gate_logits,
    num_experts,
    top_k):
    """MoE 负载均衡辅助损失 (Switch Transformer 风格)。
    目标：让 token 在各专家间尽量均匀分布，避免少数专家过载 / 路由塌缩。
    计算式: L_aux = num_experts * Σ_i ( f_i * P_i )
        - f_i: 第 i 个专家被选中的 token 比例 (流量 fraction)
        - P_i: 第 i 个专家路由概率的均值 (router confidence)
    当两者同时高 (某专家又忙又被偏好) 时 loss 增大，从而被惩罚。
    """
    # 把各层的 gate_logit 沿 token 维拼到一起: [layers*b*s, num_experts]
    concatenated_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0)
    # softmax 得到每个 token 对各专家的路由概率
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    # 每个 token 取概率最大的 top_k 个专家 -> (num_tokens, top_k)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    # one-hot: 标记每个 token 选中了哪些专家 -> (num_tokens, top_k, num_experts)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # f_i: 每个 token 选该专家的比例 (在 token 与 top_k 维上求均值)，shape (num_experts,)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # P_i: 每个专家平均路由概率 (在 token 维上求均值)，shape (num_experts,)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    # 逐元素相乘并求和: Σ_i f_i * P_i，再乘以专家数做归一化缩放
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class Gating(nn.Module):
    """MoE 路由器 (Gate)。
    把 hidden 投影到 expert_num 维 logits，取 top-k 个专家，
    并把非选中位置置 -inf 后重新 softmax，得到稀疏、和为 1 的门控权重。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        # 单层线性变换即可充当路由器
        self.gate = nn.Linear(self.hidden_size, self.expert_num)

    def forward(self, x):
        # x dim: b, s, hidden_size
        logits = self.gate(x)  # gate: b, s, expert_num
        logits_topk, indices = logits.topk(self.topk, dim=-1) # 选择概率最大的两个专家，返回两个专家对每个token的概率
        zeros = torch.full_like(logits, float("-inf")) # 创建一个全为负无穷的矩阵，用于屏蔽其他专家的概率并重新归一化概率最大的两个专家
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)  # 将选择的两个专家的概率按指定索引填充
        sparse_logits = F.softmax(sparse_logits, dim=-1) # 得到一个稀疏矩阵，选择的两个专家对每个token的概率和为1
        # 同时返回原始 gate_logit (未归一化)，供 aux loss 使用
        gate_logit = logits.view(-1, self.expert_num)

        return sparse_logits, indices, gate_logit

class Expert(nn.Module):
    """单个专家 = 一个 SwiGLU MLP。结构同 MLP。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MoE(nn.Module):
    """Mixture of Experts 层。
    实现方式：遍历每个专家，用布尔 mask 取出"被该专家选中"的 token 批量送入专家，
    再按门控权重加权、散回原位置累加。这种"按专家分组"的写法避免了逐 token 调度开销。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 实例化 expert_num 个独立专家 + 一个共享路由器
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)

    def forward(self, x):
        # 路由: 得到 (sparse_logits, indices, gate_logit)
        sparse_logits, indices, gate_logit = self.gating(x)
        # 用于累加各专家的加权输出
        final_outputs = torch.zeros_like(x)
        x_flat = x.view(-1, x.shape[-1])  # (batch_size * seq_len, dim)
        sparse_logits_flat = sparse_logits.view(-1, sparse_logits.shape[-1])  # (batch_size * seq_len, export_num))

        # 遍历每个专家，把所有路由到该专家的 token 一次性批量计算
        for i, expert in enumerate(self.experts):
            # 在 topk 维上 any: 该位置只要选中过专家 i 即为 True -> (b, s)
            expert_mask = (indices == i).any(-1)  # (batch_size, seq_len)
            expert_mask_flat = expert_mask.view(-1) # (batch_size * seq_len)
            if expert_mask_flat.any():
                # 取出这些 token 送入专家 -> (n_selected, dim)
                expert_input = x_flat[expert_mask_flat]  # (seq_true, dim)
                export_output = expert(expert_input)  # (seq_true, dim)

                # 取出对应位置的专家 i 的门控权重，并升维以便广播
                gate_scores = sparse_logits_flat[expert_mask_flat, i].unsqueeze(1)  # (seq_true) --> (seq_true, 1)

                # 专家输出 × 门控权重 (逐 token 加权)
                weighted_output = export_output * gate_scores  # (seq_true, dim)

                # 散回原位置累加 (一个 token 可被多个专家处理，结果相加)
                final_outputs[expert_mask] += weighted_output


        return final_outputs, gate_logit



class DecoderLayer(nn.Module):
    """单个解码层。
    与纯 LLaMA 的区别：FFN 部分**交替**使用密集 MLP 与 MoE
        - 偶数层 (layer_idx % 2 == 0): dense MLP
        - 奇数层:                      MoE
    这样一半层是 MoE，既享受专家容量扩展，又保留密集层做通用计算。
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.moe = MoE(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.layer_idx = layer_idx
    def forward(
        self,
        hidden_states,
        use_kv_cache
    ):
        # ---- 注意力子层 (带前置 RMSNorm + 残差) ----
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            use_kv_cache=use_kv_cache
        )

        hidden_states = residual + hidden_states
        # ---- 前馈子层 (带前置 RMSNorm + 残差) ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 偶数层用密集 MLP；奇数层用 MoE
        if self.layer_idx % 2 == 0:
            hidden_states = self.mlp(hidden_states)
            gate_logit = None
        else:
            hidden_states, gate_logit = self.moe(hidden_states)
        outputs = residual + hidden_states
        return outputs, gate_logit


# 编写自定义配置时需要记住的三个重要事项如下：
# 1、必须继承自 PretrainedConfig
# 2、PretrainedConfig 的 __init__ 方法必须接受任何 kwargs
# 3、这些 kwargs 需要传递给超类的 __init__ 方法。
class Config(PretrainedConfig):
    """模型超参配置。继承 PretrainedConfig 以便接入 HF 的 save/load/Trainer 体系。
    model_type = "moe_model" 用于 AutoConfig/AutoModelForCausalLM 的注册匹配。
    """
    model_type = "moe_model"

    def __init__(self,
                hidden_size = 512,
                num_attention_heads = 16,
                num_key_value_heads = 8,
                flash_attn = True,
                attention_bias = False,
                max_seq_len = 512,
                intermediate_size = 2048,
                mlp_bias = False,
                vocab_size = 6400,
                n_layers = 8,
                dropout = 0.0,
                expert_num = 4,
                topk = 2,
                output_router_logits = True,
                aux_loss_coef = 0.01,
                **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        # 是否输出路由 logits，供计算 aux loss 使用
        self.output_router_logits = output_router_logits
        # aux loss 的系数，控制负载均衡约束的强度
        self.aux_loss_coef = aux_loss_coef
        super().__init__(**kwargs)


class LLM(PreTrainedModel):
    """完整 MoE 语言模型。
    结构: token embedding -> N 个 DecoderLayer -> RMSNorm -> output 投影。
    embedding 与 output 共享权重 (weight tying)，省参数。
    """
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.n_layers = self.config.n_layers
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk

        # 词嵌入 + embedding 后的 dropout
        self.tokon_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        # 堆叠 n_layers 个解码层
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(self.n_layers):
            self.layers.append(DecoderLayer(self.config, layer_idx))
        # 最终归一化 + 输出投影 (到词表)
        self.norm = RMSNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重 tying: embedding 与 output 共享权重
        self.tokon_embeddings.weight = self.output.weight
        # 初始化全部子模块权重
        self.apply(self._init_weights)
        self.loss = None
        self.aux_loss = None

        # 残差相关参数的缩放初始化 (注: 这里匹配的是 w3/wo 即 LLaMA 原版命名，
        # 但本工程用的是 gate_proj/down_proj，所以这段实际不会命中，等价无操作)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def _init_weights(self, module):
        # Linear: 权重 ~N(0, 0.02)，bias 置 0
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Embedding: 同样 ~N(0, 0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input_ids, labels, use_kv_cache=False):
        # 收集所有 MoE 层的门控 logits，供 aux loss 使用
        all_router_logits = () if self.config.output_router_logits else None

        # 词嵌入 + dropout -> hidden_states (b, s, hidden)
        hidden_states = self.tokon_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)
        # 逐层前向；偶数层返回 gate_logit=None，奇数层(MoE)返回真实 gate_logit
        for idx, layer in enumerate(self.layers):
            hidden_states, gate_logit = layer(hidden_states, use_kv_cache=use_kv_cache)
            if gate_logit is not None:
                all_router_logits += (gate_logit, )

        # 最终归一化
        hidden_states = self.norm(hidden_states)


        if labels is not None:
            # 训练 / 评估: 对全序列输出 logits 并算交叉熵，忽略 padding (index 0)
            logits = self.output(hidden_states)
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        else:
            # 推理: 只输出最后一个位置的 logits，用于 next-token 生成
            logits = self.output(hidden_states[:, [-1], :])
            self.loss = None

        # 计算并叠加负载均衡辅助损失
        if self.config.output_router_logits:
            self.aux_loss = load_balancing_loss_func(all_router_logits, self.expert_num, self.topk)

            if labels is not None:
                # 总 loss = 语言模型 CE + aux_loss_coef * aux_loss
                self.loss += self.config.aux_loss_coef * self.aux_loss.to(self.loss.device)

        # 返回 HF 标准输出结构 (loss, logits)，方便 Trainer 取值
        return CausalLMOutputWithPast(self.loss, logits)

    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                 use_kv_cache=True):
        """自回归生成。
        支持 temperature 采样、top_k 截断、repetition_penalty 抑制重复、流式输出。
        """
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        s = input_ids.shape[1]  # 记录原始 prompt 长度，输出只返回新生成部分
        while input_ids.shape[1] < max_new_tokens - 1:
            # 每步把当前完整序列喂入模型，取最后一位 logits
            inference_res = self(input_ids, labels, use_kv_cache=use_kv_cache)
            logits = inference_res.logits
            logits = logits[:, -1, :]

            # 重复惩罚: 对已出现过的 token 做除法降权
            for token in set(input_ids.tolist()[0]):
                logits[:, token] /= repetition_penalty

            if temperature == 0.0:
                # 贪心: 直接取最大概率 token
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  # 温度调节分布尖锐度
                if top_k is not None:
                    # top-k 截断: 只在概率最大的 k 个里采样
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)  # 归一化为概率
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  # 按概率采样一个 token

            if idx_next == eos:
                break  # 遇到结束符停止

            # 把新 token 拼到序列末尾
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if stream:
                # 流式: 每生成一个就 yield 当前已生成部分
                yield input_ids[:, s:]

        if not stream:
            yield input_ids[:, s:]  # 非流式: 最后一次性返回生成结果

if __name__ == '__main__':

    config = Config()
    model = LLM(config)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./moe',
                            num_train_epochs=10,
                            do_train=True,
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=1,
                            # max_steps=15000,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,                # 使用 bfloat16 混合精度，节省显存
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',   # cosine 学习率衰减
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True,
                            save_safetensors=False)
    dataset = LLMDataset('./train.jsonl', tokenizer=tokenizer, max_seq_len=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/moe')
    trainer.save_state()
