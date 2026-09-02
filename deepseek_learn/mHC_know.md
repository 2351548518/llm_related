# 为什么需要 mHC？

mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）要解决的核心矛盾是：

> 既希望像 Hyper-Connections（HC）一样，用多条残差流提高信息容量；又希望像普通残差连接一样，让信号和梯度稳定地穿过深层网络。

## 1. 为什么普通残差连接很重要？

普通残差连接可以写为：

$$
\boldsymbol{x}_{l+1}
=
\boldsymbol{x}_l
+
\mathcal{F}_l(\boldsymbol{x}_l).
$$

将它递归展开到更深的第 $L$ 层：

$$
\boldsymbol{x}_L
=
\boldsymbol{x}_l
+
\sum_{i=l}^{L-1}
\mathcal{F}_i(\boldsymbol{x}_i).
$$

这里最重要的不只是“相加”，而是存在一条不经过复杂变换的直接路径：

$$
\boldsymbol{x}_l\longrightarrow\boldsymbol{x}_L.
$$

即使某些 Transformer 层暂时没有学习好，浅层信号仍然可以沿残差路径传到深层。反向传播时同样存在单位梯度路径：

$$
\frac{\partial\boldsymbol{x}_{l+1}}
{\partial\boldsymbol{x}_l}
=
\boldsymbol{I}
+
\frac{\partial\mathcal{F}_l}
{\partial\boldsymbol{x}_l}.
$$

其中的 $\boldsymbol{I}$ 是稳定深层网络的关键，这种性质称为 identity mapping（恒等映射）。

可以将它理解成：每一层都可以在原始笔记上补充内容，但不能直接丢掉原始笔记。

## 2. 普通残差连接的局限

普通 Transformer 始终只有一条 $C$ 维残差流：

```text
x0 -> x1 -> x2 -> ... -> xL
```

Attention、FFN、知识特征、语法特征和推理状态等内容，都需要写入同一个向量，可能产生以下问题：

- 不同类型的信息相互干扰；
- 新层写入的信息覆盖旧信息；
- 残差流容量与 Transformer 隐藏维度绑定；
- 如果通过增大隐藏维度提升容量，Attention 和 FFN 的计算量也会显著增加。

## 3. HC：将一条残差流扩展为多条

Hyper-Connections（HC）将一条残差流扩展为 $n$ 条：

$$
\boldsymbol{X}_l\in\mathbb{R}^{n\times C}.
$$

可以将它理解成从“一个共享笔记本”扩展为“多个并行笔记本”。

每层使用三个映射：

$$
\boldsymbol{X}_{l+1}
=
\mathcal{H}_l^{\mathrm{res}}\boldsymbol{X}_l
+
\left(\mathcal{H}_l^{\mathrm{post}}\right)^\top
\mathcal{F}_l\left(
\mathcal{H}_l^{\mathrm{pre}}\boldsymbol{X}_l
\right).
$$

三个映射的职责分别是：

| 映射 | 作用 |
| --- | --- |
| $\mathcal{H}^{\mathrm{pre}}$ | 从多条残差流读取信息，并聚合成一个 $C$ 维子层输入。 |
| $\mathcal{H}^{\mathrm{post}}$ | 将 Attention 或 FFN 的输出写回不同残差流。 |
| $\mathcal{H}^{\mathrm{res}}$ | 在残差旁路中混合不同残差流的信息。 |

这样，残差流容量扩展为 $nC$，但昂贵的 Attention 或 FFN 仍然只处理一个 $C$ 维输入，而不是直接处理完整的 $nC$ 维输入。

这给模型增加了一个新的扩展维度：

```text
传统扩展：增加网络深度、隐藏维度或训练数据
HC 扩展：增加残差流的宽度
```

参考：[Hyper-Connections 原始论文](https://arxiv.org/abs/2409.19606)。

## 4. 为什么不能直接使用无约束 HC？

问题主要出在 $\mathcal{H}_l^{\mathrm{res}}$。

普通残差连接的旁路相当于固定的单位矩阵 $\boldsymbol{I}$，而 HC 将它变成了任意可学习矩阵。信号穿过很多层后，残差路径会连续乘上：

$$
\mathcal{H}_{L-1}^{\mathrm{res}}
\mathcal{H}_{L-2}^{\mathrm{res}}
\cdots
\mathcal{H}_l^{\mathrm{res}}.
$$

只要每层稍微放大或缩小信号，经过深层累积后，就可能指数爆炸或消失。

### 4.1 信号爆炸示例

假设有两条残差流：

$$
\boldsymbol{X}_0
=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$

无约束 HC 学到如下残差映射：

$$
\mathcal{H}^{\mathrm{res}}
=
\begin{bmatrix}
1.2 & 0.3\\
0.4 & 1.1
\end{bmatrix}.
$$

一次传播后：

$$
\mathcal{H}^{\mathrm{res}}\boldsymbol{X}_0
=
\begin{bmatrix}
1.5\\
1.5
\end{bmatrix}.
$$

为了直观说明，假设每层都产生相同的 $1.5$ 倍放大。经过 20 层后：

$$
1.5^{20}\approx3325.
$$

信号从 1 被放大到三千多。反向传播需要连续乘以这些矩阵的转置，因此梯度也可能爆炸。

### 4.2 信号消失示例

如果每层都将信号缩放为原来的 $0.8$：

$$
0.8^{20}\approx0.0115.
$$

经过 20 层后，信号和梯度就可能变得非常小。

这不是纯理论问题。mHC 论文报告，在 27B 模型实验中，无约束 HC 的组合映射增益曾接近 3000，并伴随 loss 和梯度范数异常。

参考：[mHC 论文的数值稳定性分析](https://arxiv.org/html/2512.24880#S3.SS1)。

## 5. mHC 如何约束 HC？

mHC 不允许 $\mathcal{H}^{\mathrm{res}}$ 成为任意矩阵，而是将它约束为双随机矩阵：

$$
\mathcal{H}_{ij}^{\mathrm{res}}\geq0,
$$

$$
\sum_j\mathcal{H}_{ij}^{\mathrm{res}}=1,
\qquad
\sum_i\mathcal{H}_{ij}^{\mathrm{res}}=1.
$$

也就是说：

- 所有元素非负；
- 每一行的元素之和为 1；
- 每一列的元素之和为 1。

所有 $n\times n$ 双随机矩阵组成的集合称为 Birkhoff polytope。论文将这种受约束的残差连接空间称为“流形”。

在代码中，可以通过 Sinkhorn-Knopp 迭代近似完成投影：

```python
K = torch.exp(matrix)

for _ in range(num_iter):
    # 每一行归一化为 1
    K = K / K.sum(dim=-1, keepdim=True)

    # 每一列归一化为 1
    K = K / K.sum(dim=-2, keepdim=True)
```

参考：[mHC 的流形约束定义](https://arxiv.org/html/2512.24880#S4.SS1)。

## 6. 双随机矩阵为什么能够稳定传播？

假设残差映射为：

$$
\mathcal{H}^{\mathrm{res}}
=
\begin{bmatrix}
0.8 & 0.2\\
0.2 & 0.8
\end{bmatrix},
$$

输入的两条残差流为：

$$
\boldsymbol{X}
=
\begin{bmatrix}
8\\
2
\end{bmatrix}.
$$

经过残差混合：

$$
\begin{aligned}
\mathcal{H}^{\mathrm{res}}\boldsymbol{X}
&=
\begin{bmatrix}
0.8 & 0.2\\
0.2 & 0.8
\end{bmatrix}
\begin{bmatrix}
8\\
2
\end{bmatrix}\\
&=
\begin{bmatrix}
0.8\times8+0.2\times2\\
0.2\times8+0.8\times2
\end{bmatrix}\\
&=
\begin{bmatrix}
6.8\\
3.2
\end{bmatrix}.
\end{aligned}
$$

从这个例子可以观察到：

1. 第一条流的一部分进入第二条流，分支之间发生了信息交换。
2. 所有输出都是输入的凸组合，不会产生任意倍数的放大。
3. 两条残差流的总量保持不变：

   $$
   8+2=6.8+3.2=10.
   $$

4. 双随机矩阵的谱范数不超过 1，因此残差旁路不会放大信号的二范数。
5. 双随机矩阵相乘后仍然是双随机矩阵，因此稳定性质可以跨层保持。
6. 矩阵的转置仍然是双随机矩阵，所以反向传播也获得类似约束。

mHC 并不是强制：

$$
\mathcal{H}^{\mathrm{res}}=\boldsymbol{I}.
$$

它是在单位矩阵附近提供一个更大的安全空间：

```text
普通残差：只能原样传递
无约束 HC：可以任意混合，但可能失控
mHC：可以灵活混合，同时受到稳定性约束
```

当 $n=1$ 时，唯一的双随机矩阵就是：

$$
\mathcal{H}^{\mathrm{res}}=[1].
$$

因此 mHC 会自然退化为普通残差旁路。

## 7. 一个完整的两分支计算示例

假设当前有两条残差流：

$$
\boldsymbol{X}_l
=
\begin{bmatrix}
8\\
2
\end{bmatrix}.
$$

### 7.1 Pre Mapping：读取信息

设：

$$
\mathcal{H}^{\mathrm{pre}}
=
\begin{bmatrix}
0.75 & 0.25
\end{bmatrix}.
$$

子层输入为：

$$
\begin{aligned}
\boldsymbol{h}^{\mathrm{pre}}
&=
\mathcal{H}^{\mathrm{pre}}\boldsymbol{X}_l\\
&=
0.75\times8+0.25\times2\\
&=6.5.
\end{aligned}
$$

Attention 或 FFN 只需要执行一次。假设其输出为：

$$
\boldsymbol{h}^{\mathrm{out}}
=
\mathcal{F}(6.5)
=1.
$$

### 7.2 Res Mapping：混合残差流

$$
\boldsymbol{h}^{\mathrm{res}}
=
\begin{bmatrix}
0.8 & 0.2\\
0.2 & 0.8
\end{bmatrix}
\begin{bmatrix}
8\\
2
\end{bmatrix}
=
\begin{bmatrix}
6.8\\
3.2
\end{bmatrix}.
$$

### 7.3 Post Mapping：把新信息写回各分支

设：

$$
\mathcal{H}^{\mathrm{post}}
=
\begin{bmatrix}
0.2 & 1.4
\end{bmatrix}.
$$

那么：

$$
\begin{aligned}
\boldsymbol{h}^{\mathrm{post}}
&=
\left(\mathcal{H}^{\mathrm{post}}\right)^\top
\boldsymbol{h}^{\mathrm{out}}\\
&=
\begin{bmatrix}
0.2\\
1.4
\end{bmatrix}.
\end{aligned}
$$

这说明模型认为：本层产生的新信息对于第二条残差流更加重要。

### 7.4 合并两条路径

$$
\begin{aligned}
\boldsymbol{X}_{l+1}
&=
\boldsymbol{h}^{\mathrm{res}}
+
\boldsymbol{h}^{\mathrm{post}}\\
&=
\begin{bmatrix}
6.8\\
3.2
\end{bmatrix}
+
\begin{bmatrix}
0.2\\
1.4
\end{bmatrix}\\
&=
\begin{bmatrix}
7.0\\
4.6
\end{bmatrix}.
\end{aligned}
$$

需要注意：mHC 约束的是残差旁路 $\mathcal{H}^{\mathrm{res}}\boldsymbol{X}_l$ 的传播稳定性，而不是要求整个层的输出总量完全不变。Attention 或 FFN 会写入新信息，因此完整隐藏状态本来就应该发生变化。

## 8. 为什么 Pre Mapping 和 Post Mapping 也要约束？

mHC 使用：

$$
\mathcal{H}^{\mathrm{pre}}
=
\sigma\left(\widetilde{\mathcal{H}}^{\mathrm{pre}}\right),
$$

$$
\mathcal{H}^{\mathrm{post}}
=
2\sigma\left(\widetilde{\mathcal{H}}^{\mathrm{post}}\right).
$$

因此：

$$
\mathcal{H}^{\mathrm{pre}}\in(0,1),
\qquad
\mathcal{H}^{\mathrm{post}}\in(0,2).
$$

这样设计主要是为了：

- 避免大量正负权重相互抵消；
- 防止读取和写入系数没有边界；
- 由于 $2\sigma(0)=1$，让 $\mathcal{H}^{\mathrm{post}}$ 在接近零初始化时具有自然的单位写入尺度。

不过，mHC 最核心的传播稳定性仍然来自对 $\mathcal{H}^{\mathrm{res}}$ 的双随机约束。

## 9. 对应到代码实现

仓库中的 [`mHC.ipynb`](./mHC.ipynb) 将一次 mHC 计算拆成 `width_connection` 和 `depth_connection` 两个阶段。

### 9.1 生成映射并读取残差流

```python
h_pre, h_res, H_post = mhc.width_connection(hidden_states)
```

其中：

```python
h_pre = H_pre @ hidden_states
h_res = H_res @ hidden_states
```

- `h_pre` 是多条残差流聚合后的子层输入；
- `h_res` 是经过双随机矩阵混合的残差旁路；
- `H_post` 决定子层输出如何写回各条残差流。

### 9.2 执行 Attention 或 FFN

```python
h_out = layer(h_pre)
```

### 9.3 写回并合并残差

```python
h_post = H_post.unsqueeze(-1) @ h_out.unsqueeze(-2)
output = h_res + h_post
```

完整数据流为：

```text
从多条残差流中读取
        |
        v
执行 Attention / FFN
        |
        v
将结果写回多条残差流

同时，残差流之间通过受约束的 H_res 安全混合
```

## 10. 什么时候值得使用 mHC？

mHC 并不是所有模型都必须使用。普通残差连接已经足以稳定训练大量模型。

mHC 更适合以下场景：

- 希望扩展残差流的信息容量；
- 希望增加跨分支、跨层连接的表达能力；
- 网络很深，或者模型训练规模很大；
- 使用无约束 HC 时出现信号或梯度不稳定；
- 能够通过算子融合、重计算等方式控制额外的显存和通信成本。

多条残差流会增加激活、显存访问和并行通信开销。mHC 论文在经过定制算子融合等基础设施优化、残差流扩展率 $n=4$ 时，报告了约 6.7% 的额外训练时间开销。

参考：[mHC 原始论文](https://arxiv.org/html/2512.24880)。

## 11. 总结

一句话概括：

> 普通残差连接提供稳定但单一的信息通道；HC 提供丰富但可能失控的多通道；mHC 用双随机约束将 HC 限制在一个既能交换信息、又不会任意放大信号的安全空间中。

mHC 的核心价值不是简单地“增加几条残差分支”，而是实现以下平衡：

| 目标 | mHC 的处理方式 |
| --- | --- |
| 增加残差容量 | 将一条 $C$ 维残差流扩展为 $n$ 条，总容量为 $nC$。 |
| 控制主干计算量 | 通过 $\mathcal{H}^{\mathrm{pre}}$ 聚合后，Attention/FFN 仍只处理 $C$ 维输入。 |
| 允许分支交换信息 | 使用可学习的 $\mathcal{H}^{\mathrm{res}}$ 混合不同残差流。 |
| 防止信号和梯度爆炸 | 将 $\mathcal{H}^{\mathrm{res}}$ 约束为双随机矩阵。 |
| 保持深层稳定性 | 利用双随机矩阵在矩阵乘法下的封闭性。 |
| 降低读写抵消 | 对 $\mathcal{H}^{\mathrm{pre}}$ 和 $\mathcal{H}^{\mathrm{post}}$ 施加非负约束。 |

更完整的公式、参数形状和模型集成代码参见 [`mHC_Note.md`](./mHC_Note.md)。
