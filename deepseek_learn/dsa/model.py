"""在 Qwen2 Attention 上演示 DeepSeek Sparse Attention（DSA）的核心流程。

建议按下面的顺序阅读：

1. :class:`Indexer` 为每个 Query 和历史 Token 计算一个轻量级相关性分数；
2. Top-K Selector 从相关性分数中选出 K 个历史位置；
3. :class:`Qwen2Attention` 将未选位置设为负无穷，再执行正常的 Softmax 和 Value 聚合；
4. 两个训练脚本分别完成 Indexer 预热和稀疏联合训练。

张量形状示例：若 batch_size=2、seq_len=1024、num_heads=14、head_dim=64，
则 Query 为 [2, 14, 1024, 64]，Indexer Key 为 [2, 1, 1024, 64]，
Indexer 分数为 [2, 1, 1024, 1024]，Top-128 索引为 [2, 1, 1024, 128]。
"""

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from typing import Optional, Union, List, Tuple, Dict, Any
from collections.abc import Callable
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2MLP, Qwen2RMSNorm, Qwen2PreTrainedModel, Qwen2RotaryEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils.generic import check_model_inputs
from transformers.generation import GenerationMixin
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    # 每层保存 (topk_indices, raw_attn_weights, indexer_attn_scores)，供两阶段训练计算 KL Loss。
    attentions: Optional[List[tuple[torch.FloatTensor, ...]]] = None
    
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    # 注意：这里不是单个注意力矩阵，而是每层 DSA 训练所需的三个中间结果。
    attentions: Optional[List[tuple[torch.FloatTensor, ...]]] = None


class Indexer(nn.Module):
    """Lightning Indexer：为当前 Query 快速检索最相关的历史 Token。

    Indexer 只负责“候选位置检索”，不负责计算最终的注意力概率或聚合 Value。
    它为每个历史位置打分，Top-K Selector 再根据这些分数产生索引。

    例子：当前 Token 对 6 个可见位置的分数为
    ``[0.1, 2.3, 0.4, 0.2, 1.8, 2.0]``。当 ``K=3`` 时，返回的位置是
    ``[1, 5, 4]``；主注意力随后只在这 3 个位置上分配概率。
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.num_attention_heads
        self.key_value_heads = config.num_key_value_heads
        self.head_dim: int = config.hidden_size // config.num_attention_heads
        # 每个 Query 最多保留 128 个历史位置；序列不足 128 时保留全部可见位置。
        self.index_topk: int = 128

        # Indexer Key 是单头 Key：hidden_size -> head_dim。
        # 例如 [B, L, 896] -> [B, L, 64]，不是 [B, L, num_heads]。
        self.wk = nn.Linear(self.hidden_size, self.head_dim) 

        # 为每个 Query 产生 num_heads 个聚合权重，用于合并各 Query Head 的检索分数。
        # 例如 [B, L, 896] -> [B, L, 14]。
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads)

        # 增量生成时缓存 Indexer Key。persistent=False 表示保存 checkpoint 时不写入该缓存。
        self.register_buffer("k_cache", None, persistent=False)
        


    def forward(self, hidden_states: torch.Tensor, query_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask=None):
        """计算 Indexer 分数并返回每个 Query 的 Top-K 位置。

        参数形状：
            hidden_states: [B, query_len, hidden_size]
            query_states: [B, num_heads, query_len, head_dim]
            cos/sin: 当前位置使用的 RoPE 参数
            mask: [B, 1, query_len, key_len]，不可见位置通常为负无穷

        返回形状：
            topk_indices: [B, 1, query_len, K]
            attn_scores: [B, 1, query_len, key_len]
        """
        
        bsz, seqlen, _ = hidden_states.size()

        # Linear 只改变最后一维：
        # [B, query_len, hidden_size] -> [B, query_len, head_dim]。
        key_states = self.wk(hidden_states)
 
        # [B, query_len, hidden_size] -> [B, query_len, num_heads]。
        # n_heads**-0.5 用于控制跨头加权求和后的数值尺度。
        weights = self.weights_proj(hidden_states) * self.n_heads ** -0.5

        # query_states: [B, num_heads, query_len, head_dim]
        # key_states:   [B, query_len, head_dim]
        
        if seqlen > 1:
            # Prefill：一次输入整个 Prompt，直接用当前所有 Token 初始化缓存。
            # 例：输入 1024 个 Token，缓存形状为 [B, 1024, head_dim]。
            self.k_cache = key_states
        
        if seqlen == 1:
            # Decode：每次只输入一个新 Token，把新 Key 追加到历史缓存。
            # 例：[B, 1024, D] + [B, 1, D] -> [B, 1025, D]。
            key_states = torch.cat([self.k_cache, key_states], dim=1)
            self.k_cache = key_states

        # 增加“单个 Indexer Head”维度：
        # [B, key_len, head_dim] -> [B, 1, key_len, head_dim]。
        key_states = key_states.unsqueeze(1)

        # Indexer Key 也使用 RoPE，使检索分数感知 Token 的相对/绝对位置信息。
        key_states, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
    
        # [B, H, query_len, D] @ [B, 1, D, key_len]
        #     -> [B, H, query_len, key_len]。
        # ReLU 将负相关性截断为 0，对应公式中的 ReLU(q^T k)。
        attn_scores = query_states @ key_states.transpose(2,3)
        attn_scores = F.relu(attn_scores, inplace=False)
        
        # [B, H, query_len, 1] * [B, H, query_len, key_len]
        #     -> [B, H, query_len, key_len]。
        # 即每个 Query 使用自己的 w_{t,j} 给不同 Query Head 加权。
        attn_scores = weights.transpose(1,2).unsqueeze(-1) * attn_scores
        
        # 在 Head 维度求和，得到每个 Query-Key 位置对的单个索引分数：
        # [B, H, query_len, key_len] -> [B, 1, query_len, key_len]。
        attn_scores = attn_scores.sum(1, keepdim=True)
       
        if mask is not None:
            # 加入 Causal/Padding Mask，确保未来位置和 Padding 不可能进入有效 Top-K。
            attn_scores = attn_scores + mask

        # 例：key_len=1024、index_topk=128 时，每个 Query 返回 128 个位置索引。
        # topk_indices 的形状是 [B, 1, query_len, 128]。
        topk_indices = attn_scores.topk(min(self.index_topk, key_states.shape[2]), dim=-1)[1]
        return topk_indices, attn_scores


class Qwen2Attention(nn.Module):
    """加入 DSA 检索与 Top-K Mask 的 Qwen2 自注意力层。

    普通注意力直接在所有可见 Key 上执行 Softmax；这里先让 Indexer 选择位置，
    再把未选择位置加上负无穷。Softmax 后未选择位置的概率为 0。

    例：某个 Query 可见 8 个历史位置，Top-K 选择 ``[1, 4, 7]``，则 Index Mask 为
    ``[-inf, 0, -inf, -inf, 0, -inf, -inf, 0]``。
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        
        self.indexer = Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """执行标准 Q/K/V 投影、Indexer 检索和稀疏注意力计算。"""

        bsz, seqlen, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Q 使用 num_attention_heads；K/V 使用 num_key_value_heads（GQA）。
        # 以 Qwen2.5-0.5B 为例，Q 可能是 [B, 14, L, 64]，K/V 是 [B, 2, L, 64]。
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 给 Q/K 加旋转位置编码；Value 不使用 RoPE。
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            # 增量生成时把当前 K/V 写入标准 Attention Cache。
            # cache_position 用于告诉 Static Cache 当前 Token 应写入哪个位置。
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
        # GQA 中多个 Query Head 共享 K/V Head。repeat_kv 后，K/V 的 Head 数与 Q 相同。
        # 例：[B, 2, L, 64] 重复 7 组 -> [B, 14, L, 64]。
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # 主注意力的原始 logits，形状为 [B, num_heads, query_len, key_len]。
        # 这些 logits 也作为训练 Indexer 时的教师信号。
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    

        raw_attn_weights = None
        indexer_attn_scores = None
        topk_indices = None
        # 分支一：模型已经准备好 Causal/Padding Mask。
        # 训练通常进入此分支；某些带显式 Mask 的推理也可能进入此分支。
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # bool Mask 中 True 通常表示“可见”，这里转成加法 Mask：可见=0，不可见=-inf。
                attention_mask = attention_mask.logical_not()
                attention_mask = attention_mask.float().masked_fill(attention_mask, float('-inf'))

            # Indexer 负责打分，Top-K Selector 负责返回候选位置。
            topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=attention_mask)

            # 保存“没有应用 Index Mask”的主注意力 logits，供 KL 蒸馏使用。
            raw_attn_weights = attn_weights + attention_mask

            # 初始全部为 -inf，再把 Top-K 位置 scatter 成 0。
            # 例：索引 [1, 4] -> [-inf, 0, -inf, -inf, 0, ...]。
            index_mask = torch.full((bsz, 1, seqlen, seqlen), float("-inf"), device=hidden_states.device).scatter(-1, topk_indices, 0)

            # 最终允许位置 = Causal/Padding Mask 允许位置 ∩ Top-K 位置。
            index_mask = index_mask + attention_mask
            attn_weights = attn_weights + index_mask

            # 未选择位置为 -inf，Softmax 后其概率为 0。
            attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)
            
        # 分支二：没有外部 Attention Mask，代码自行区分 Prefill 和逐 Token Decode。
        else:
            # Prefill：一次处理长度大于 1 的完整 Prompt。
            if seqlen > 1:
                # 构造下三角 Causal Mask。
                # 例如第 3 个 Query 只能看到 Key 0、1、2、3，不能看到未来位置。
                mask = torch.tril(torch.ones((bsz, 1, seqlen, seqlen), device=hidden_states.device, dtype=torch.bool), diagonal=0)
                mask = mask.logical_not()
                mask = mask.float().masked_fill(mask, float('-inf'))
               
                topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=mask)

                raw_attn_weights = attn_weights + mask
                
                index_mask = torch.full((bsz, 1, seqlen, seqlen), float("-inf"), device=hidden_states.device).scatter(-1, topk_indices, 0)
                index_mask = index_mask + mask
                attn_weights = attn_weights + index_mask
                
                attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)
            
            # Decode：每次生成一个 Token，query_len=1，key_len=历史长度+1。
            else:
                topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=None)

                # Decode 的 Mask 形状必须使用完整 key_len，而不是当前 query_len=1。
                # 例：生成第 1025 个 Token 时为 [B, 1, 1, 1025]。
                index_mask = torch.full((bsz, 1, 1, key_states.shape[-2]), float("-inf"), device=hidden_states.device).scatter(-1, topk_indices, 0)
                attn_weights = attn_weights + index_mask
                attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)
        
        # 使用稀疏化后的概率聚合 Value：
        # [B, H, query_len, key_len] @ [B, H, key_len, D]
        #     -> [B, H, query_len, D]。
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 合并所有 Head，再映射回 hidden_size。
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # 三个中间结果仅在 output_attentions=True 时由上层返回，用于训练 KL Loss。
        return attn_output, (topk_indices, raw_attn_weights, indexer_attn_scores)



class Qwen2DecoderLayer(GradientCheckpointingLayer):
    """一个标准的 Pre-Norm Decoder Layer，只替换了 Self Attention 实现。"""
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # 子层一：RMSNorm -> DSA Self Attention -> 残差连接。
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attentions = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # 子层二：RMSNorm -> MLP -> 残差连接。
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attentions
  
    
class Qwen2Model(Qwen2PreTrainedModel):
    """由 Token Embedding、多个 DSA Decoder Layer 和最终 RMSNorm 组成的主干模型。"""
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            # [B, L] Token ID -> [B, L, hidden_size] Token Embedding。
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # generate(use_cache=True) 首次调用时创建可增长的标准 K/V Cache。
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        # RoPE 的 cos/sin 在所有 Decoder Layer 间复用。
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 收集每层的 (topk_indices, raw_attn_weights, indexer_attn_scores)。
        all_attentions = []
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, attentions = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                output_attentions=output_attentions,
                **kwargs,
            )
            all_attentions.append(attentions)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attentions=all_attentions if output_attentions else None,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    """在 DSA Qwen2 主干之上增加词表投影和因果语言建模损失。"""
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # 生成时可以只计算最后若干位置的 logits，以减少词表投影开销。
        # logits_to_keep=0 时 slice(0, None) 会保留所有 Token。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Transformers 的因果 LM Loss 会在内部完成 labels 的右移对齐。
            # labels=-100 的 Prompt/Padding 位置不会参与交叉熵损失。
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    
    
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')
 
    messages = [{"role": "user", "content": "你好"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize = False)
    # inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=48)
    inputs = tokenizer(text, return_tensors="pt")['input_ids']
    print(inputs)
    
    output = model.generate(inputs, do_sample=False)

    print(tokenizer.decode(output[0]))

    # for layer in model.model.layers:
    #     old_self_attn = layer.self_attn
    #     new_self_attn = Qwen2Attention(layer.self_attn.config, layer.self_attn.layer_idx)
    #     new_self_attn.load_state_dict(old_self_attn.state_dict(), strict=False)
    #     layer.self_attn = new_self_attn
    
    model = Qwen2ForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')
    output = model.generate(inputs, do_sample=False)

    print(tokenizer.decode(output[0]))
