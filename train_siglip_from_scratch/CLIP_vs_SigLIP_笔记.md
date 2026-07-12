# CLIP 与 SigLIP 对比笔记

> 结合本文件夹的三张图(`clip.png` / `siglip.png` / `伪代码.png`)和 `model.py` 的实际代码,逐公式拆解两种图文对比损失,并给出一个可手算的小 batch 例子。

---

## 0. 网络结构

> 一句话先说结论:**CLIP 和 SigLIP 的网络结构几乎完全一样,都是"图像塔 + 文本塔"的双塔结构。两者的区别不在网络结构,而在损失函数。**SigLIP 是一篇"训练目标改进"的论文,结构上仅多一个偏置 `b` 来适配新损失。

### 0.1 通用双塔结构(两者共有)

```
        ┌──────────────────┐                        ┌──────────────────┐
图像 ──→│  Image Encoder   │──→ img_feat ─→ L2 norm │                  │
        │  (ViT / ResNet)  │                        │                  │
        └──────────────────┘                        │   相似度矩阵      │
                                                  │  S = I · Tᵀ       │──→ 损失
        ┌──────────────────┐                        │  (余弦相似度)     │
文本 ──→│  Text Encoder    │──→ txt_feat ─→ L2 norm │                  │
        │  (Transformer)   │                        │                  │
        └──────────────────┘                        └──────────────────┘
```

对应 `model.py` 的前向部分(model.py:46-60):

```python
text_features   = text_model(input_ids, attention_mask)[1]      # pooler_output
vision_features = vision_model(pixel_values)[1]                  # pooler_output

# L2 归一化,让内积 = 余弦相似度
vision_features = vision_features / vision_features.norm(p=2, dim=-1, keepdim=True)
text_features   = text_features   / text_features.norm(p=2, dim=-1, keepdim=True)

logits_per_text  = torch.matmul(text_features, vision_features.t()) * self.t.exp() + self.b
logits_per_image = logits_per_text.t()
```

两者都要把 embedding 做 **L2 归一化**,这样 `text @ image.T` 就是余弦相似度,取值落在 `[-1, 1]`,再乘温度 `exp(t)`(SigLIP 还多加一个偏置 `b`)得到 logit。

### 0.2 组件对照(通用结构 → 本项目实现)

| 组成部分 | CLIP/SigLIP 通用 | 本项目实现 |
|---|---|---|
| 图像编码器 | ViT(或 ResNet) | `vit-base-patch16-224` |
| 文本编码器 | Transformer(BERT 或 GPT 风格) | `chinese-roberta-wwm-ext` |
| 特征向量 | 取 [CLS] / pooler_output | `vision_outputs[1]`、`text_outputs[1]` |
| 归一化 | L2,使内积 = 余弦相似度 | `features / features.norm(...)` |
| 相似度 | `S = I·Tᵀ`(N×N 矩阵) | `matmul(text, vision.t())` |
| 温度/偏置 | 温度 τ | SigLIP 多一个偏置 b:`S·exp(t)+b` |

### 0.3 三个结构关键细节

1. **双塔各自独立编码**:图像和文本**不共享参数、不交互**,只在最后算一次内积。这也是为什么推理时可以分别给图库和文本库预建索引、做极速检索(把图文匹配退化为最近邻搜索)。
2. **投影到同一维度空间**:原始 CLIP 在 backbone 后接一个线性投影头(`W_proj`),把图像和文本向量都映射到同一个 d 维空间(如 512/768)。本项目省略了显式投影头,直接用 ViT/BERT 的 `pooler_output`——它们本来就同是 768 维,可直接对齐,这是简化写法,本质等价。
3. **L2 归一化后,内积就是余弦相似度**,取值 `[-1,1]`,再用温度 `exp(t)` 放大到合理 logit 量级。

### 0.4 CLIP 的结构(见 `clip.png`)

```
图1 图2 图3 ...  →  Image Encoder  →  I1 I2 I3 ...  ┐
                                                   ├→ N×N 相似度矩阵 → 对称 softmax 损失
文1 文2 文3 ... →  Text Encoder  →  T1 T2 T3 ...  ┘
```

- **图像塔**:论文用 ViT-L/14 或 ResNet-50/ResNet-x。
- **文本塔**:因果(自回归)Transformer(类似 GPT-2),取 **[EOS] token** 的表示,再用线性头投影。
- 结构要点:没有跨模态融合层,纯双塔。
- 损失:对相似度矩阵的行和列各做一次 softmax(对称 InfoNCE),标签是"对角线"——即第 i 张图配第 i 段文本。

### 0.5 SigLIP 的结构(见 `siglip.png`)

```
图1 图2 图3 ...  →  Image Encoder  →  I1 I2 I3 ...  ┐
                                                   ├→ N×N 相似度矩阵 → 逐元素 sigmoid 损失
文1 文2 文3 ... →  Text Encoder  →  T1 T2 T3 ...  ┘
```

- **网络结构与 CLIP 几乎一致**:双塔、投影头、L2 归一化、相似度矩阵。
- **结构层面的两处差异**:
  1. **多一个可学习偏置 `b`**:`logit = sim · exp(t) + b`。CLIP 只有温度 τ,SigLIP 多了 bias——因为 sigmoid 损失对 logit 的绝对偏移敏感(softmax 对平移不敏感),需要 bias 来校准。
  2. **文本塔多用 Encoder 风格**(BERT/[CLS]),而非 CLIP 的因果 Transformer。但这是实现选择,不是强制。
- 损失:对相似度矩阵的**每一个元素**单独做 `logsigmoid(y·z)` 二分类,标签矩阵 `2I-1`(对角 +1,非对角 -1)。

对应 `model.py:59`:
```python
logits_per_text = torch.matmul(text_features, vision_features.t()) * self.t.exp() + self.b
```
这就是 SigLIP 结构的"图像塔 + 文本塔 + 温度 t + 偏置 b"。

### 0.6 结构对比表

| | CLIP | SigLIP |
|---|---|---|
| 整体架构 | 双塔 | 双塔(相同) |
| 图像编码器 | ViT / ResNet | ViT / ResNet(相同) |
| 文本编码器 | 因果 Transformer + [EOS] | 常用 Encoder + [CLS](非强制) |
| 投影头 | 线性投影到共享 d 维 | 线性投影到共享 d 维(相同) |
| L2 归一化 | ✅ | ✅(相同) |
| 相似度矩阵 | `I·Tᵀ` | `I·Tᵀ`(相同) |
| 温度参数 τ | ✅(仅温度) | ✅ 温度 t |
| 偏置参数 b | ❌ | ✅ ← 结构上唯一的实质新增 |
| **损失函数** | **对称 softmax / InfoNCE** | **逐元素 sigmoid + BCE** |
| 跨模态融合层 | ❌ | ❌ |

> **为什么"结构一样"值得强调**:很多初学者以为 SigLIP 是"新模型",其实它只是把 CLIP 双塔末端的"N 选 1 softmax 对比"换成了"N×N 个独立二分类的 sigmoid"。结构上仅多一个偏置 `b` 适配新损失。所以本项目 `model.py` 才能用几乎同一份前向代码承载 SigLIP——`t`/`b` 两个参数 + `logsigmoid` 损失,就是它相对 CLIP 的全部"结构"差异。要在本项目实现 CLIP,只需把 `model.py` 损失段(model.py:62-67)替换为对称交叉熵、并去掉 `b`,塔结构完全不动。

---

## 1. CLIP:对称的 InfoNCE / softmax 损失

### 公式(见 `clip.png`)

设 batch 内有 `N` 个图文对,`s_ij = <I_i, T_j>` 是第 i 张图与第 j 段文本的相似度。

**对称交叉熵:**

$$
L_{\text{text}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)},\qquad
L_{\text{image}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ji}/\tau)}
$$

$$
L_{\text{CLIP}} = \frac{L_{\text{text}} + L_{\text{image}}}{2}
$$

其中标签是 `labels = arange(N)`(即对角线那一个正样本)。

### 关键特征

- **softmax 跨整个 batch 归一化**:每一行(每段文本)在所有 N 张图上做 softmax,要求“正样本对的得分要打过 batch 内所有负样本”。
- 因此 **batch 越大,负样本越多,对比信号越强**;CLIP 训练常需 8k~32k 的大 batch。
- 每行的梯度互相耦合(分母是所有样本之和),分布式训练时要跨卡 all-gather 整个 batch,工程复杂。

> ⚠️ 本项目 `model.py` **没有**实现 CLIP 损失,这里只是作为对照。若要实现,把 `logsigmoid` 那段换成 `F.cross_entropy(logits / t, torch.arange(b))` 的对称版本即可。

---

## 2. SigLIP:逐元素 sigmoid 损失

### 符号约定:$\sigma$ 是什么

$\sigma$ 是 **sigmoid(逻辑斯蒂)函数**,把任意实数压到 $(0,1)$ 当"概率"用:

$$
\sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^x}{1+e^x}
$$

几个关键值:

| 输入 $x$ | $\sigma(x)$ | 含义 |
|---|---|---|
| $+\infty$ | $\to 1$ | $x$ 越大,越接近 1("是"的概率) |
| $0$ | $=0.5$ | 中间值,模棱两可 |
| $-\infty$ | $\to 0$ | $x$ 越小,越接近 0("否"的概率) |

形状是过 $(0,0.5)$ 的 **S 形单调递增曲线**。

在 SigLIP 里它的作用是把"logit"(可正可负的分数)翻译成"匹配概率":

$$
\text{logit}_{ij}=t\,\mathbf{x}_i\cdot\mathbf{y}_j+b \;\xrightarrow{\;\sigma\;}\; \text{匹配概率}\in(0,1)
$$

- logit 大正数 → $\sigma\to1$ → "这对图文匹配";
- logit 大负数 → $\sigma\to0$ → "这对图文不匹配";
- logit≈0 → $\sigma=0.5$ → 模型没把握。

> `test.ipynb` 里 `probs = torch.sigmoid(logits_per_image)` 就是这一步——把 logit 经 $\sigma$ 变成 50.2%/51.0% 这种概率。接近 0.5 正说明 logit≈0、模型没训练好。
>
> 代码里 `F.logsigmoid` = $\log\sigma$,所以 $\sigma$ 这个符号藏在 `logsigmoid` 的后半段名字里(详见后文"代码里的 sigmoid 藏在哪")。

### 公式(见 `siglip.png` / `伪代码.png`)

SigLIP 把“N 选 1 的 softmax 分类”改写成“**N×N 个独立的二分类**”:每一对 (i, j) 单独判断“图 i 与文本 j 是否匹配”。

标签矩阵:`y_ij = +1`(配对,对角线)、`y_ij = -1`(不配对,非对角线)。

$$
L_{\text{SigLIP}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\log\sigma\bigl(y_{ij}\cdot z_{ij}\bigr),\qquad z_{ij} = s_{ij}\,e^{t} + b
$$

### 对应代码(model.py:62-67)

```python
b = logits_per_text.shape[0]
eye = torch.eye(b, device=logits_per_text.device)
labels = 2*eye - torch.ones_like(logits_per_text, device=logits_per_text.device)
#  对角线 = +1,非对角线 = -1
loglik = F.logsigmoid(labels * logits_per_text)   # logsigmoid(y * z)
nll    = -torch.sum(loglik, dim=-1)               # 对一行内所有 j 求和
loss   = nll.mean()                                 # 再对 batch 平均
```

逐行对应公式:
- `2*eye - 1` → `y_ij ∈ {+1, -1}`;
- `logsigmoid(labels * logits)` → `logσ(y·z)`;
- `-sum(...)` → 内层 ∑_j;
- `.mean()` → 外层 `/N`。

### 公式符号 ↔ 代码逐项对应

你贴的公式(SigLIP 论文形式):

$$
\mathcal{L} = -\frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\sum_{j=1}^{|\mathcal{B}|}\underbrace{\log \frac{1}{1+e^{z_{ij}}(-t\mathbf{x}_i\cdot\mathbf{y}_j+\bar{b})}}_{\mathcal{L}_{ij}}
$$

> ⚠️ 上式里 `e^{z_{ij}}(-t·sim+b)` 这段的指数/括号位置有混叠。按 SigLIP 论文标准形式,它应理解为:

$$
\mathcal{L}_{ij} = -\log \frac{1}{1+e^{-z_{ij}\,(t\,\mathbf{x}_i\cdot\mathbf{y}_j + b)}} = -\log\sigma\!\bigl(z_{ij}\cdot\underbrace{(t\,\mathbf{x}_i\cdot\mathbf{y}_j + b)}_{\text{logit}_{ij}}\bigr)
$$

即"对 logit $\text{logit}_{ij}=t\,\mathbf{x}_i\cdot\mathbf{y}_j+b$ 做 sigmoid 二分类,标签 $z_{ij}\in\{+1,-1\}$"。代码就是按这个干净形式实现的。下面逐符号对应:

| 公式符号 | 含义 | 代码对应(model.py) |
|---|---|---|
| $\|\mathcal{B}\|$ | batch 大小 $N$ | `b = logits_per_text.shape[0]` |
| $\mathbf{x}_i$ | 第 $i$ 个**文本** embedding(L2 归一化后) | `text_features[i]`  shape `[768]` |
| $\mathbf{y}_j$ | 第 $j$ 个**图像** embedding(L2 归一化后) | `vision_features[j]`  shape `[768]` |
| $\mathbf{x}_i\cdot\mathbf{y}_j$ | 余弦相似度 $s_{ij}$ | `matmul(text_features, vision_features.t())[i,j]` |
| $t$ | 温度(论文里 $t=e^{\bar t}$,即 log-temperature 取指数) | `self.t.exp()` —— `self.t` 存的是 $\bar t$ |
| $b$ ($\bar b$) | 偏置 bias | `self.b` |
| $z_{ij}$ | ±1 标签(+1 配对/-1 不配对) | `labels[i,j]` = `2*eye - 1` |
| $\text{logit}_{ij}$ | 单格 logit | `logits_per_text[i,j]` |
| $\mathcal{L}_{ij}$ | 单格损失 $-\log\sigma(z_{ij}\cdot\text{logit}_{ij})$ | `-loglik[i,j]` = `-F.logsigmoid(labels*logits)[i,j]` |

**三步拼装:**

1. **logit**(model.py:59):
   $$\text{logit}_{ij} = (\mathbf{x}_i\cdot\mathbf{y}_j)\,e^{\bar t} + b$$
   ```python
   logits_per_text = torch.matmul(text_features, vision_features.t()) * self.t.exp() + self.b   # [N,N]
   ```

2. **单格损失** $\mathcal{L}_{ij} = -\log\sigma(z_{ij}\cdot\text{logit}_{ij})$(model.py:65):
   $$-\log\sigma(z_{ij}\cdot\text{logit}_{ij}) \;\longleftrightarrow\; \texttt{-F.logsigmoid(labels * logits\_per\_text)[i,j]}$$
   ```python
   loglik = F.logsigmoid(labels * logits_per_text)   # [N,N] = log σ(z·logit),逐格独立
   ```

3. **聚合**(model.py:66-67):内层 $\sum_j$ + 外层 $\tfrac{1}{N}\sum_i$:
   $$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\mathcal{L}_{ij}\;\longleftrightarrow\;\texttt{nll = -sum(loglik, dim=-1); loss = nll.mean()}$$
   ```python
   nll  = -torch.sum(loglik, dim=-1)   # [N]  内层 ∑_j:每行对所有 j 求和
   loss = nll.mean()                    # []   外层 1/N:对 batch 平均
   ```

**关于符号约定的两点澄清:**
- 公式里 $z_{ij}$ 是 **±1 标签**;而本笔记 §2 上方的简写式用 $y_{ij}$ 表示标签、$z_{ij}$ 表示 logit —— 两套写法符号相反,别混淆。**以代码为准**:`labels` 是 ±1 标签,`logits_per_text` 是 logit。
- 公式里 $\mathbf{x}_i\cdot\mathbf{y}_j$ 的下标 $i$ 对应 `logits_per_text` 的**行(文本)**、$j$ 对应**列(图像)**;即 $\mathbf{x}$=文本、$\mathbf{y}$=图像。这只是约定,换成 $\mathbf{x}$=图像也对称等价(矩阵转置即可,代码里 `logits_per_image = logits_per_text.t()`)。

### 那个等式怎么来的 & 代码里的 sigmoid 藏在哪

**(a) 等式就是 sigmoid 的定义代入,不是额外推导。** sigmoid 定义 $\sigma(a)=\frac{1}{1+e^{-a}}$,令 $a=z_{ij}\cdot\text{logit}_{ij}=z_{ij}(t\,\mathbf{x}_i\cdot\mathbf{y}_j+b)$:

$$
\sigma(z_{ij}\cdot\text{logit}_{ij})=\frac{1}{1+e^{-z_{ij}\cdot\text{logit}_{ij}}}=\frac{1}{1+e^{-z_{ij}(t\,\mathbf{x}_i\cdot\mathbf{y}_j+b)}}
$$

两边取 $-\log$ 即 $\mathcal{L}_{ij}$。

**(b) 为什么是 $z_{ij}\cdot\text{logit}_{ij}$ 这个形式(±1 标签统一正负样本)。** 二分类"这对图文是否匹配":
- 正样本 $z=+1$:希望 logit 大 → $\sigma(\text{logit})\to1$,匹配概率 $=\sigma(+1\cdot\text{logit})=\sigma(\text{logit})$;
- 负样本 $z=-1$:希望 logit 小 → $\sigma(\text{logit})\to0$,不匹配概率 $=1-\sigma(\text{logit})=\sigma(-\text{logit})=\sigma(-1\cdot\text{logit})$。

两种情况被 $\pm1$ 标签统一成 $\sigma(z_{ij}\cdot\text{logit}_{ij})$="分类正确的概率",取负对数似然即损失。**这是 SigLIP 用 ±1 标签的妙处**:正负样本共享一个公式,不用像 BCE 那样分支写。

**(c) 代码里的 sigmoid 在 `F.logsigmoid` 名字里。**

```python
loglik = F.logsigmoid(labels * logits_per_text)   # [N,N]
```

`torch.nn.functional.logsigmoid(x)` = $\log\sigma(x)$,**内部同时干了 sigmoid 和 log 两件事**(且数值稳定,大 logit 不溢出)。对应拆解:

| 代码片段 | 公式 |
|---|---|
| `labels * logits_per_text` | $z_{ij}\cdot\text{logit}_{ij}$ |
| `F.logsigmoid(...)` | $\log\sigma(z_{ij}\cdot\text{logit}_{ij})$ |
| `nll = -torch.sum(loglik, ...)` 的负号 | $-\log\sigma(\cdot)=\mathcal{L}_{ij}$ |

若硬拆开 `logsigmoid`(等价但数值不稳定,别真这么用):
```python
loglik = torch.log(torch.sigmoid(labels * logits_per_text))   # 等价,但大 logit 时 exp 会溢出
```
所以 **sigmoid 没有消失,只是被 `logsigmoid` 和 log 打包一起算了** —— 这就是代码里只见 `logsigmoid`、不见单独 `sigmoid` 的原因。

### 关键特征

- **没有 softmax 归一化分母**,每个 (i,j) 对的损失和梯度只依赖它自己的 logit,彼此解耦。
- 因此 **小 batch 也能训**(SigLIP 论文用 8k~16k,但效果在小 batch 下比 CLIP 鲁棒得多),不再需要“靠大 batch 堆负样本”。
- 多了一个可学习偏置 `b`,以及温度 `t`(代码里是 `t.exp()`,所以 `t` 学的是 log-temperature)。
- 因为是独立二分类,负样本是“逐格”提供的,天然支持非对称 batch(图和文本数量不必相等)。

---

## 3. 一个可手算的小例子

设 `batch = 3`,三段文本与三张图,**第 1 个对是正样本**(图0↔文0),其余都不配对。为方便手算,假设 L2 归一化后的相似度矩阵和参数已给出:

```
        图0    图1    图2
文0  [ 0.8 ,  0.1 ,  0.0 ]     ← 文0 与 图0 配对
文1  [ 0.2 ,  0.7 ,  0.1 ]     ← 文1 与 图1 配对(本例暂当负样本看)
文2  [ 0.0 ,  0.2 ,  0.6 ]     ← 文2 与 图2 配对(本例暂当负样本看)
```

设 `exp(t)=1, b=0`(训练初期近似),则 `z_ij = s_ij`。

**标签矩阵**(`2*I - 1`):

```
labels = [[+1, -1, -1],
          [-1, +1, -1],
          [-1, -1, +1]]
```

**逐项计算** `logσ(y·z)`(用 `σ(x)=1/(1+e^-x)`,`logσ(x)= -softplus(-x)`):

| (i,j) | y·z = labels·s | σ(y·z) | logσ(y·z) |
|-------|----------------|--------|-----------|
| (0,0) | +0.8 | 0.690 | -0.371 |
| (0,1) | -0.1 | 0.475 | -0.744 |
| (0,2) |  0.0 | 0.500 | -0.693 |
| (1,0) | -0.2 | 0.450 | -0.799 |
| (1,1) | +0.7 | 0.668 | -0.403 |
| (1,2) | -0.1 | 0.475 | -0.744 |
| (2,0) |  0.0 | 0.500 | -0.693 |
| (2,1) | -0.2 | 0.450 | -0.799 |
| (2,2) | +0.6 | 0.646 | -0.437 |

**第一行求和**(nll[0])= -( -0.371 -0.744 -0.693 ) = 0.371 + 0.744 + 0.693 = **1.808**(负对数似然,越大越差)
同理 nll[1] ≈ 0.799+0.403+0.744 = 1.946;nll[2] ≈ 0.693+0.799+0.437 = 1.929。

**最终 loss = mean(nll) ≈ (1.808 + 1.946 + 1.929) / 3 ≈ 1.894**。

物理含义:对角线正样本 `y·z > 0`(如 (0,0) 的 +0.8)让 `σ→1`、`logσ→0`,贡献小;非对角线负样本 `y·z<0`(如 (0,1) 的 -0.1)让 `σ→0.5 以下`,`logσ` 为负、被取负后变正,贡献大。**训练就是要推高对角线 logit、压低非对角线 logit**,直到所有非对角项的 `y·z` 都很负(→ -∞),loss 才趋于 0。

### 对照 CLIP 在同一矩阵上的行为

CLIP 第 0 行的 loss = `-log( softmax([0.8,0.1,0.0])[0] )` = `-log( e^0.8/(e^0.8+e^0.1+e^0) )` ≈ `-log(2.225/(2.225+1.105+1.0))` ≈ `-log(0.512)` ≈ **0.669**。它**只关心正样本能否“赢过”行内其余负样本的相对排序**,而 SigLIP 还要求负样本绝对值被压到足够负。这正是两者本质差异:**CLIP 是相对的 N 选 1,SigLIP 是绝对的逐格二分类**。

---

## 4. 推理时的差别(对应 `test.ipynb`)

```python
probs = torch.sigmoid(logits_per_image)
# 50.2% 与"两只猫在玩耍", 51.0% 与"一颗树"
```

- **SigLIP**:推理直接对每个 logit 做 `sigmoid`,得到“这对图文匹配的概率”,是**独立、可逐格解读**的概率(两张图之间无需比较)。这也是为什么 `test.ipynb` 能逐条打印概率。
- **CLIP**:通常用 `softmax(logits/T)` 得到“这张图最像哪段文本”的**相对分布**,或直接比 `logits` 大小排序(检索场景)。概率是“行内归一化”的,不独立。

> 顺带说明:`test.ipynb` 输出 50% / 51% 接近随机,说明当前 `outputs/` 里的 checkpoint **训练严重不足**——对角线 logit 没被推高、非对角没被压低,`sigmoid` 自然落在 0.5 附近。可查 `outputs/trainer_state.json` 的 loss 曲线确认是否收敛。

---

## 5. 一句话总结

| | CLIP | SigLIP |
|---|---|---|
| 损失形式 | 对称 softmax / InfoNCE | 逐元素 sigmoid + BCE |
| 归一化 | 跨 batch softmax(分母耦合) | 无(逐格独立) |
| 标签 | `arange(N)`(N 选 1) | `2I-1` ∈ {+1,-1}(每格二分类) |
| 额外参数 | 温度 τ | 温度 t(=log τ)+ 偏置 b |
| 推理概率 | softmax 相对分布 | sigmoid 独立概率 |
| 对 batch 大小 | 敏感,需大 batch | 鲁棒,小 batch 也能训 |

**核心一句话**:CLIP 让正样本在行内“赢过”所有负样本(相对排序);SigLIP 让每个正对足够大、每个负对足够小(绝对二分类)。本项目 `model.py` 实现的就是后者。
