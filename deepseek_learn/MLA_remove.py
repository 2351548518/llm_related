

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# RMS 归一化：只根据均方根缩放，不像 LayerNorm 那样减去均值。
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 先转为 float32，避免低精度下平方、求均值时数值不稳定。
        # 注意：这里没有在返回前恢复输入 dtype，混合精度场景中输出会保持 float32。
        hidden_states = hidden_states.float()
        # keepdim=True 保留最后一维，便于通过广播与原张量相乘。
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.float()
    
    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 二维旋转公式的向量化形式：x*cos(theta) + rotate_half(x)*sin(theta)。
    q_embed = (q*cos) + (rotate_half(q)*sin)
    k_embed = (k*cos) + (rotate_half(k)*sin)
    
    return q_embed, k_embed

# 旋转位置编码（Rotary Position Embedding，RoPE）。
class RotaryEmbedding(nn.Module):
    """预先缓存每个位置对应的 cos/sin，forward 时按序列长度切片。"""

    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        # dim=4 时会得到两个角频率：10000^0 和 10000^(-2/4)。
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # t 的形状为 [max_seq_len, 1]，表示位置 0,1,...,max_seq_len-1。
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        # [max_seq_len, 1] @ [1, dim/2] -> [max_seq_len, dim/2]
        freqs = t @ inv_freq.unsqueeze(0)
        # 与 rotate_half 的“前半/后半”布局对应，复制为 [max_seq_len, dim]。
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        # 例：q 的形状为 [B, S, H, Dr]，此处取得位置 [0, S) 的编码。
        # 注意：增量解码 start_pos>0 时，正确做法通常应取 [start_pos, start_pos+S)，
        # 当前接口没有接收 start_pos，因此会重复使用从位置 0 开始的编码。
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)    

# ====================================================================================================

class MLA(nn.Module):

    def __init__(self,
                dim,
                n_heads,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                max_seq_len,
                max_batch_size,
                mode):
        super().__init__()
        self.dim = dim # 隐藏层维度
        self.n_heads = n_heads  #总头数
        self.q_lora_rank = q_lora_rank # q低秩压缩到的维度
        self.kv_lora_rank = kv_lora_rank # kv低秩压缩到的维度

        # nope = No Positional Encoding，不施加旋转位置编码的部分。
        self.qk_nope_head_dim = qk_nope_head_dim
        # rope = 使用旋转位置编码的部分；MLA 只对较小的这部分应用 RoPE。
        self.qk_rope_head_dim = qk_rope_head_dim

        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim # qk的总维度，不带旋转位置编码的维度加上带旋转位置编码的维度

        self.v_head_dim = v_head_dim # value的维度，等于不带旋转位置编码的k维度

        self.mode = mode
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        
        
        # ------------------------- Query 的低秩投影 -------------------------
        # 例：[B,S,4096] -> [B,S,128]，先压缩以减少 Query 投影参数量。
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank) # q的降维矩阵
        self.q_norm = RMSNorm(self.q_lora_rank)
        # 例：[B,S,128] -> [B,S,16*(256+48)] = [B,S,4864]。
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim) # q的升维矩阵

        
        # ---------------------- Key/Value 的联合压缩 
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim) # kv的降维矩阵 生成“压缩 KV + 共享的位置 Key”
        # nn.Linear(self.dim, self.kv_lora_rank)
        # nn.Linear(self.dim, self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)) # kv的升维矩阵 生成“每个注意力头的内容 Key + Value”
        
        # 拼接 H 个头的注意力结果后，再投影回模型隐藏维度。
        # 例：[B,S,16*256] -> [B,S,4096]。
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        
        # 注意：此处未传入 MLA 的 max_seq_len，RotaryEmbedding 会使用默认值 1024。
        self.rotary_emb = RotaryEmbedding(self.qk_rope_head_dim) # 旋转旋转位置编码

        # ============================================================================================
        if self.mode == 'naive':
            # naive 模式保存每个头完整的 K 和 V：
            # k_cache=[max_B,max_T,H,Dn+Dr]，v_cache=[max_B,max_T,H,Dv]。
            # 多头 注意力 使用 的 方法
            self.register_buffer('k_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.n_heads, self.qk_head_dim), persistent=False)
            self.register_buffer('v_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.n_heads, self.v_head_dim), persistent=False)
            
        else:
            # 压缩模式只保存所有头共享的 latent KV 和位置编码 K：
            # kv_cache=[max_B,max_T,Ckv]，pe_cache=[max_B,max_T,Dr]。
            # persistent=False 表示缓存不会被写入 state_dict（模型权重文件）。
            """
            低维 的 压缩 KV
            """
            self.register_buffer('kv_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer('pe_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim), persistent=False)
            
        
    def forward(self, x, start_pos: int, mask=None):
        
        bs, seq_len, _ = x.shape
        end_pos = start_pos + seq_len

        # ====== 构造 Query ======
        q = self.wq_a(x)  # [bs, seq_len, q_lora_rank] 降维
        q = self.q_norm(q) # [bs, seq_len, q_lora_rank]
        q = self.wq_b(q) # [bs, seq_len, n_heads * qk_head_dim]
        q = q.view(bs, seq_len, self.n_heads, self.qk_head_dim) # [bs, seq_len, n_heads, qk_head_dim (qk_nope_head_dim + qk_rope_head_dim) ]
        # 示例形状分别为 [4,100,16,256] 和 [4,100,16,48]。
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) 
        
        # ====== 构造压缩 KV 和 K_rope ======
        kv = self.wkv_a(x) # [bs, seq_len, kv_lora_rank + qk_rope_head_dim] 降维
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) 
        k_pe = k_pe.unsqueeze(2) # k_pe shape:[bs, seq_len, 1, qk_rope_head_dim]

        # ====== 旋转位置编码 ======
        # 只对 q_pe/k_pe 应用 RoPE；q_nope/k_nope 不进行旋转。
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)


        if self.mode == 'naive':
            # ====== 方式一：展开完整的 K/V ======
            q = torch.cat([q_nope, q_pe], dim=-1)
            
            # 把 latent KV 从 Ckv 维解压为 H 个头各自的 K_nope 和 V。
            kv = self.kv_norm(kv) # [bs, seq_len, kv_lora_rank)]
            kv = self.wkv_b(kv) # [bs, seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]
            kv = kv.view(bs, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # 结果 K=[B,S,H,Dn+Dr]，示例为 [4,100,16,304]。
            k = torch.cat([k_nope, k_pe.expand(-1,-1,self.n_heads,-1)], dim=-1) 

            """
            保存KV cache
            K     : [4,100,16,304]
            V     : [4,100,16,256]
            """
            self.k_cache[:bs, start_pos:end_pos, :, :] = k
            self.v_cache[:bs, start_pos:end_pos, :, :] = v

            """
            [4,16,100,304] x [4,16,304,100]
            -> [4,16,100,100]
            -> transpose(1,2)
            -> scores: [4,100,16,100]
            """
            scores = torch.matmul(q.transpose(1, 2), self.k_cache[:bs, :seq_len, :, :].transpose(1, 2).transpose(2, 3) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
            scores = scores.transpose(1, 2)
        else:
            # ====== 方式二：在 latent 空间计算注意力 ======
            k_pe = k_pe.squeeze(2) # [4,100,48]

            # PyTorch Linear 的 weight 是 [out_features, in_features]
            wkv_b = self.wkv_b.weight  # [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
            """
            前 256 行: W_K [16,256,64]
            后 256 行: W_V [16,256,64]
            """
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)


            """
            # 权重吸收的核心：
            #   k_nope = W_K @ c
            #   q_nope @ k_nope^T = q_nope @ (W_K @ c)^T = (q_nope @ W_K) @ c^T
            # [4,100,16,256] × [16,256,64] -> [4,100,16,64]
            """
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # 缓存归一化后的 latent KV 和应用 RoPE 后的共享 K_pe。
            kv = self.kv_norm(kv)
            self.kv_cache[:bs, start_pos:end_pos, :] = kv # kv shape:[bs, seq_len, kv_lora_rank]
            self.pe_cache[:bs, start_pos:end_pos, :] = k_pe # k_pe shape:[bs, seq_len, qk_rope_head_dim]
            
            scores_nope = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bs, :seq_len, :])
            scores_pe = torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bs, :seq_len, :])
            # 两部分分数相加后，仍按完整 Q/K 头维度 进行缩放。
            scores = (scores_nope + scores_pe) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim) # [bs, seq_len, n_heads, seq_len]
        
        if mask is not None:
            # mask shape:[bs, seq_len, seq_len]
            scores += mask.unsqueeze(2)
        
        scores = scores.softmax(dim=-1)
       
        if self.mode == 'naive':
            # 标准注意力加权求和：AttentionWeights @ V
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bs, :seq_len])
        else:
            """
            A(cW_V^T)=(Ac)W_V^T

            [4,100,16,100] × [4,100,64]
            -> latent context：[4,100,16,64]

            [4,100,16,64] × [16,256,64]
            -> context：[4,100,16,256]
            """
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bs, :seq_len])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

    
        # 将 H 个头拼接：[B,S,H,Dv] -> [B,S,H*Dv]，再投影回 dim。
        x = x.contiguous ().view(bs, seq_len, -1)
        x = self.wo(x)
        
        return x

if __name__ == '__main__':
    # 构造一批示例输入：4 个样本，每个样本 100 个 token，每个 token 4096 维。
    x = torch.randn(4, 100, 4096)
    
    # 以下参数只用于演示张量如何在 MLA 中流动，并非某个正式模型的完整配置。
    dim = 4096
    n_heads = 16
    q_lora_rank = 128
    kv_lora_rank = 64
    qk_nope_head_dim = 256
    qk_rope_head_dim = 48
    v_head_dim = 256
    max_seq_len = 512
    max_batch_size = 16
    # 只有 mode == 'naive' 才走完整 K/V 缓存分支；其他字符串都会走压缩分支。
    mode = 'none'

    mla = MLA(dim=dim, 
            n_heads=n_heads, 
            q_lora_rank=q_lora_rank, 
            kv_lora_rank=kv_lora_rank, 
            qk_nope_head_dim=qk_nope_head_dim, 
            qk_rope_head_dim=qk_rope_head_dim, 
            v_head_dim=v_head_dim, 
            max_seq_len=max_seq_len, 
            max_batch_size=max_batch_size, 
            mode=mode)
    # 注意：MLA.forward 要求传入 start_pos，这里的原始调用没有传入，运行时会
    # 先报缺少 start_pos 参数。即使补为 mla(x, start_pos=0)，forward 中的
    # seqlen 拼写也仍会触发 NameError。本文件此次仅增加讲解注释，不修改原逻辑。
    print(mla(x,start_pos=0))
    # 压缩模式下，kv_cache 的完整预分配形状为 [16, 512, 64]。
    print(mla.kv_cache)
