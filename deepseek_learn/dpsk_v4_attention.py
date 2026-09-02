"""“局部精确注意力 + 压缩历史注意力”的简化教学实现。

本文件包含两种配置：

1. CSA（``compress_ratio=4``）：每 4 个 token 构成一个块，相邻块之间使用
   overlap 表示，再由 Indexer 为每个 Query 选择 Top-K 个压缩块。
2. HCA（``compress_ratio=128``）：每 128 个 token 构成一个块，Query 可以
   使用所有已经完整结束的历史块。

两种配置都会对最近 ``window_size`` 个 token 保留精确的 token 级注意力。
本文统一使用下面的符号：

    B = batch size，S = 序列长度，D = 输入维度，
    H = head 维度，R = 压缩率，N = floor(S / R)。

整体数据流如下：

    x [B, S, D]
      |-- 线性投影 ----------> 局部 q/k/v [B, S, H]
      `-- Compressor --------> 压缩块 [B, N, H]

    output[t] = Attention(q[t], local_window[t] + selected_blocks[t])

这是用于说明算法的数据流原型，不是可直接替换生产模型的注意力层：当前实现
只有一个头，使用 Python 循环，没有 padding mask 和稀疏 Kernel，并且输出维度
是 ``head_dim``，没有再投影回 ``dim``。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn



class RMSNorm(nn.Module):
    """使用均方根对最后一个维度归一化。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 例：x.shape == [2, 8, 64] 时，会分别为 2*8 个 token 向量计算
        # 一个 RMS 标量，最后的 64 维保持不变。
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


@dataclass
class AttentionConfig:
    """CompressedAttention 使用的维度配置和历史选择策略。"""

    # 输入隐藏状态 x 的维度。
    dim: int = 256
    # 当前单头教学实现的 head 维度。
    head_dim: int = 64
    # 每个 Query 可以精确看到的最近 token 数量，其中包含 Query 自己。
    window_size: int = 128
    # 一个压缩块代表多少个 token；本文件中 4 对应 CSA，128 对应 HCA。
    compress_ratio: int = 4
    # 仅供 CSA 使用：Indexer 从压缩历史中选择的块数。
    history_topk: int = 16


class Compressor(nn.Module):
    """每 ``compress_ratio`` 个 token 池化成一个可学习的历史向量。

    不使用 overlap 时，假设 R=4、S=10：

        输入 token：       [0 1 2 3] [4 5 6 7] [8 9]
        完整块：              block 0    block 1
        压缩结果：              c0         c1       # shape [B, 2, H]
        剩余 token：                                  [8 9]

    只有完整块才会被压缩。最后不足一个块的 token 单独返回，调用方仍可以用
    token 级分辨率处理它们。

    在 CSA 分支（R=4）中，``wkv`` 和 ``wgate`` 会为每个 token 分别产生两组
    H 维通道。前 H 维传递给下一块作为 overlap 候选，后 H 维描述当前块。
    因此 block 1 的池化输入来自：

        token 0..3 的前 H 维 + token 4..7 的后 H 维

    第一个块没有前驱块，因此用 -inf 填充虚假的 overlap 分数；经过 softmax
    后，这些位置的概率为 0。
    """

    def __init__(self, dim: int, head_dim: int, compress_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        # 只有 CSA 配置会把前一个块用作 overlap。
        self.overlap = compress_ratio == 4

        # Python 中 True 等于 1，False 等于 0：
        #   CSA -> coeff=2 -> 两组 H 维通道
        #   HCA -> coeff=1 -> 一组 H 维通道
        coeff = 1 + self.overlap
        # 虽然名字是 wkv，但它输出的是待压缩的候选特征，而不是常规意义上相互
        # 独立的 Key 和 Value。wgate 为候选特征的每个坐标生成池化 logit。
        self.wkv = nn.Linear(dim, coeff * head_dim)
        self.wgate = nn.Linear(dim, coeff * head_dim)
        # APE 是可学习的块内位置偏置。以 R=4 为例，它的四行可以让块内
        # 第 0、1、2、3 个位置具有不同的池化权重。
        self.ape = nn.Parameter(torch.zeros(compress_ratio, coeff * head_dim))
        self.norm = RMSNorm(head_dim)

        # 为可能的流式实现预留的变量。当前 forward 没有把 current_kv 和
        # current_score 用作持久缓存，并且每次调用都会重置 prev_block_*。
        self.current_kv = []
        self.current_score = []
        self.prev_block_kv = None
        self.prev_block_score = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(完整块的压缩结果, 未压缩的剩余 token)``。

        参数：
            x：形状为 ``[B, S, D]`` 的隐藏状态。

        返回：
            compressed：形状为 ``[B, floor(S / R), H]``。
            remainder：形状为 ``[B, S % R, D]``。
        """

        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        # 例：S=10、R=4 -> cutoff=8。token 0..7 构成两个完整块，
        # token 8..9 暂时不压缩。
        cutoff = (seqlen // ratio) * ratio

        # 序列长度不足一个块时，还没有可以使用的压缩历史。
        if cutoff == 0:
            return x.new_zeros(bsz, 0, self.head_dim), x

        # [B, cutoff, D] -> [B, cutoff, coeff * H]
        kv = self.wkv(x[:, :cutoff])
        score = self.wgate(x[:, :cutoff])

        # 把 token 轴切分成 N 个完整块，每个块包含 R 个 token：
        # [B, cutoff, coeff*H] -> [B, N, R, coeff*H].
        kv = kv.view(bsz, cutoff // ratio, ratio, -1)
        score = score.view(bsz, cutoff // ratio, ratio, -1)

        # 通过广播，为每个 batch、每个块加上同一组可学习位置偏置：
        # [1, 1, R, coeff*H]。
        score = score + self.ape.view(1, 1, ratio, -1)

        outputs = []
        self.prev_block_kv = None
        self.prev_block_score = None
        for block_idx in range(kv.size(1)):
            # 两个张量的形状都是 [B, R, coeff*H]。
            kv_block = kv[:, block_idx]
            score_block = score[:, block_idx]

            if not self.overlap:
                # softmax 沿 R 个 token 位置计算。由于 logit 中仍保留 H 轴，
                # 输出的每个坐标都会学习自己的一组 R 维池化权重，而不是所有
                # 坐标共享一个标量权重。
                # [B, R, H] --沿 R 求和--> [B, H]
                weights = torch.softmax(score_block, dim=1)
                pooled = (kv_block * weights).sum(dim=1)
                outputs.append(self.norm(pooled))

            else:
                # 后 H 维表示当前块自身。
                # main_kv/main_score: [B, R, H]
                main_kv = kv_block[..., self.head_dim :]
                main_score = score_block[..., self.head_dim :]

                if self.prev_block_kv is None:
                    # block 0 没有前驱块。-inf 可以保证 R 个虚假候选位置经过
                    # softmax 后的概率严格为 0。
                    overlap_kv = torch.zeros_like(main_kv)
                    overlap_score = torch.full_like(main_score, float("-inf"))
                else:
                    # 对于 block i>0，取 block i-1 产生的前 H 维，作为
                    # block i 的 overlap 部分。
                    overlap_kv = self.prev_block_kv[..., : self.head_dim]
                    overlap_score = self.prev_block_score[..., : self.head_dim]

                # [前一块的 R 个候选, 当前块的 R 个候选] -> [B, 2R, H]
                mixed_kv = torch.cat([overlap_kv, main_kv], dim=1)
                mixed_score = torch.cat([overlap_score, main_score], dim=1)
                weights = torch.softmax(mixed_score, dim=1)
                pooled = (mixed_kv * weights).sum(dim=1)

                # detach() 让 overlap 只传递数值：压缩块 i 无法经由 overlap
                # 向块 i-1 反向传播梯度。这里的 ``-ratio:`` 表示整个 R-token 块。
                self.prev_block_kv = kv_block[:, -ratio:].detach()
                self.prev_block_score = score_block[:, -ratio:].detach()
                outputs.append(self.norm(pooled))

        # N 个 [B, H] 张量堆叠为 [B, N, H]。
        compressed = torch.stack(outputs, dim=1)

        # 不足以填满一个块的 token 不做压缩，直接返回。
        remainder = x[:, cutoff:]
        return compressed, remainder


class Indexer(nn.Module):
    """为每个 CSA Query 选择相关性最高的压缩块。

    假设 ``B=2, S=32, N=8, H=8, topk=2``：

        q:                    [2, 32, 8]
        compressed_history:  [2,  8, 8]
        scores:               [2, 32, 8]
        返回的索引：           [2, 32, 2]

    某一行索引为 ``[6, 1]``，表示该 Query 会关注第 6 和第 1 个压缩块。
    这是一个简化的单头 Indexer；与仓库中的 DSA 文档不同，它没有使用 ReLU，
    也没有使用可学习权重聚合多个头。

    注意：这个类会对传入的所有块排序，它自身不会应用因果/可见性掩码。
    如果用于因果语言模型，调用方必须先隐藏尚未完成的块或未来块。
    """

    def __init__(self, dim: int, head_dim: int, topk: int = 16):
        super().__init__()
        self.topk = topk
        self.wq = nn.Linear(dim, head_dim)

    def forward(self, x: torch.Tensor, compressed_history: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, S, min(topk, N)]`` 的 Top-K 块编号。"""

        # 没有完整块时，也就没有可以选择的历史编号。此时最后一维长度为 0，
        # -1 只是为了与其他分支的哨兵值约定保持一致。
        if compressed_history.size(1) == 0:
            return torch.full(
                (x.size(0), x.size(1), 0),
                -1,
                device=x.device,
                dtype=torch.long,
            )

        # 把每个 token 投影到与压缩块相同的 H 维比较空间：
        # [B, S, D] -> [B, S, H]。
        q = self.wq(x)
        # 每个 Query 与每个压缩块做点积：
        # [B, S, H] x [B, N, H] -> [B, S, N].
        scores = torch.einsum("bsd,bnd->bsn", q, compressed_history)
        k = min(self.topk, compressed_history.size(1))

        # topk().indices 保存的是块编号，而不是注意力概率。
        return scores.topk(k, dim=-1).indices


class CompressedAttention(nn.Module):
    """把精确的局部注意力与低分辨率的压缩历史注意力结合起来。

    对于位置 ``t`` 上的 Query，精确局部记忆的范围是：

        left = max(0, t + 1 - window_size)
        token 编号 = left, ..., t

    选中的压缩块会追加到局部记忆之后，Query 再对合并后的序列执行普通的
    Scaled Dot-Product Attention。

    本层有意简化为单头，返回 ``[B, S, head_dim]``。完整的 Transformer 层
    通常还需要多头拆分与合并、投影回 ``dim``、dropout 和残差连接。
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        # 用于精确局部窗口的 token 级投影。
        self.wq = nn.Linear(config.dim, config.head_dim)
        self.wk = nn.Linear(config.dim, config.head_dim)
        self.wv = nn.Linear(config.dim, config.head_dim)
        self.compressor = Compressor(
            dim=config.dim,
            head_dim=config.head_dim,
            compress_ratio=config.compress_ratio
        )
        self.indexer = None
        if config.compress_ratio == 4:
            # 在这个教学实现中，只有 CSA 会执行 Top-K 块检索；HCA 会直接
            # 暴露所有满足因果约束、已经结束的块。
            self.indexer = Indexer(config.dim, config.head_dim, config.history_topk)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """计算注意力，并返回便于观察的历史中间结果。

        参数：
            x：形状为 ``[B, S, D]`` 的输入隐藏状态。

        返回：
            output：形状为 ``[B, S, H]`` 的注意力结果。
            辅助字典：
                ``compressed_history``：形状为 ``[B, floor(S/R), H]``；
                ``selected_history``：每个 Query 选中的压缩块编号。
        """

        bsz, seqlen, _ = x.shape
        # q、k、v 的形状都是 [B, S, H]。
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 剩余 token 已经由精确局部注意力覆盖，因此此处只取完整块的压缩结果。
        compressed_history, _ = self.compressor(x)
        if self.config.compress_ratio == 4:
            # CSA：为每个 Query 选择最多 history_topk 个块编号。
            # 注意：当前 Indexer 没有因果块掩码。例如，t=0 的 Query 也可能
            # 选中包含未来 token 的 block 3。此处保留原代码行为，只用注释
            # 明确指出这一点。
            selected_history = self.indexer(x, compressed_history)
        elif self.config.compress_ratio == 128:
            # HCA：构造包含所有“已经因果可见的完整块”的编号列表。
            n_blocks = compressed_history.size(1)
            if n_blocks == 0:
                selected_history = torch.full((bsz, seqlen, 0), -1, device=x.device, dtype=torch.long)
            else:
                block_ids = torch.arange(n_blocks, device=x.device)
                token_positions = torch.arange(seqlen, device=x.device)

                # 当前 token 只能使用在它之前已经结束的块。
                # 为便于展示，下面用 R=4、S=10 举例（实际 HCA 配置使用 R=128）：
                # token_positions.unsqueeze(-1) = [[0],[1],...,[9]]
                # 完整块的右边界                         = [4, 8]
                # [
                # [False, False],  # t=0
                # [False, False],  # t=1
                # [False, False],  # t=2
                # [False, False],  # t=3
                # [ True, False],  # t=4
                # [ True, False],  # t=5
                # [ True, False],  # t=6
                # [ True, False],  # t=7
                # [ True,  True],  # t=8
                # [ True,  True],  # t=9
                # ]

                visible_blocks = token_positions.unsqueeze(-1) >= (
                    (block_ids + 1) * self.config.compress_ratio
                )

                # 为了方便批处理，这里保留矩形张量：可见位置存块编号，不可见
                # 位置存 -1。在 R=4 的例子中，selected_history[0, 5]
                # 等于 [0, -1]。
                selected_history = torch.full((bsz, seqlen, n_blocks), -1, device=x.device, dtype=torch.long)
                expanded_ids = block_ids.view(1, 1, n_blocks).expand(bsz, seqlen, n_blocks)
                mask = visible_blocks.unsqueeze(0).expand(bsz, seqlen, n_blocks)
                # 只在通过因果可见性检查的位置写入块编号。
                selected_history[mask] = expanded_ids[mask]
        else:
            raise ValueError(
                "compress_ratio must be in {4, 128}."
            )

        outputs = []
        # 标准缩放点积因子，避免点积数值随着 head_dim 增大而明显变大。
        scale = 1.0 / math.sqrt(self.config.head_dim)
        for t in range(seqlen):
            # 例：t=20、window_size=16 -> left=5，因此精确局部记忆包含
            # token 5..20，共 16 个 token，其中包含当前 token 20。
            left = max(0, t + 1 - self.config.window_size)
            recent_k = k[:, left : t + 1]
            recent_v = v[:, left : t + 1]

            # batch 中的每个样本可能选择不同的压缩块，因此逐样本收集。
            idx = selected_history[:, t]
            chosen_k = []
            for b in range(bsz):
                valid = idx[b][idx[b] >= 0]
                if len(valid) == 0:
                    chosen_k.append(compressed_history.new_zeros(0, self.config.head_dim))
                else:
                    chosen_k.append(compressed_history[b, valid])

            max_hist = max(tensor.size(0) for tensor in chosen_k)
            # 把可能具有不同长度的块选择结果装入矩形张量。
            hist_k = compressed_history.new_zeros(bsz, max_hist, self.config.head_dim)
            hist_v = compressed_history.new_zeros(bsz, max_hist, self.config.head_dim)
            for b in range(bsz):
                n = chosen_k[b].size(0)
                if n > 0:
                    # 当前原型把同一个压缩向量同时当作 Key 和 Value。更完整的
                    # 实现可以分别学习两种不同的历史表示。
                    hist_k[b, :n] = chosen_k[b]
                    hist_v[b, :n] = chosen_k[b]

            # Memory 的排列顺序是：[最近的精确 token, 压缩历史块]。
            memory_k = torch.cat([recent_k, hist_k], dim=1)
            memory_v = torch.cat([recent_v, hist_v], dim=1)

            # q[:, t]: [B, H]，memory_k: [B, M, H]
            # -> score/prob: [B, M]。
            score = torch.einsum("bd,bnd->bn", q[:, t], memory_k) * scale
            prob = score.softmax(dim=-1)
            # 对 M 个记忆向量加权求和，得到位置 t 上的一个 [B, H] 输出。
            out_t = torch.einsum("bn,bnd->bd", prob, memory_v)
            outputs.append(out_t)

        # S 个 [B, H] 张量堆叠为 [B, S, H]。辅助张量可以用于检查创建了哪些
        # 压缩块，以及每个 Query 最终选择了哪些块。
        return torch.stack(outputs, dim=1), {
            "compressed_history": compressed_history,
            "selected_history": selected_history,
        }


if __name__ == "__main__":
    # 两种配置共享一份输入，便于直接比较结果形状。
    torch.manual_seed(0)
    x = torch.randn(2, 256, 32)

    # CSA 示例：
    #   S=32、R=4 -> 得到 8 个压缩块；
    #   Top-K=2   -> 32 个 Query 中的每一个都会返回 2 个块编号。
    csa_cfg = AttentionConfig(
        dim=32,
        head_dim=8,
        window_size=16,
        compress_ratio=4,
        history_topk=2,
    )
    csa = CompressedAttention(csa_cfg)
    csa_output, csa_aux = csa(x[:, :32])
    print("csa output shape:", tuple(csa_output.shape))
    print("csa compressed history shape:", tuple(csa_aux["compressed_history"].shape))
    print("csa history indices shape:", tuple(csa_aux["selected_history"].shape))

    # HCA 示例：
    #   S=256、R=128 -> 得到 2 个压缩块；
    #   selected_history 为每个 Query 保留两个槽位，尚不可见的块填 -1。
    #   例如 t=0 对应 [-1, -1]，t=128 对应 [0, -1]。
    hca_cfg = AttentionConfig(
        dim=32,
        head_dim=8,
        window_size=16,
        compress_ratio=128
    )
    hca = CompressedAttention(hca_cfg)
    hca_output, hca_aux = hca(x)
    print("heavy output shape:", tuple(hca_output.shape))
    print("heavy compressed history shape:", tuple(hca_aux["compressed_history"].shape))
    print("heavy history indices shape:", tuple(hca_aux["selected_history"].shape))
