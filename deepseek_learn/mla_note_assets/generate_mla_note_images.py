"""为 MLA.md 生成矩阵板书风格的教学插图。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent

GREEN = "#d9ead3"
GREEN_DARK = "#70ad47"
GREEN_LIGHT = "#eef7e9"
ORANGE = "#f28c18"
RED = "#ef4444"
BLUE = "#4f81bd"
PURPLE = "#8064a2"
GRAY = "#666666"
LIGHT_GRAY = "#f5f5f5"
INK = "#202124"


# Matplotlib 自带字体不包含中文。显式注册系统中的 Noto CJK 字体，避免中文被
# 渲染成方框；该 TTC 的内部 family 名称是 Noto Sans CJK JP，但覆盖简体中文。
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK_FONT_PATH)
CJK_FONT_FAMILY = font_manager.FontProperties(fname=CJK_FONT_PATH).get_name()


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [CJK_FONT_FAMILY, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
    }
)


def make_canvas(title, subtitle=None):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.add_patch(Rectangle((0.15, 0.15), 15.7, 8.7, fill=False, lw=1.2, ec="#777777"))
    ax.text(0.45, 8.55, title, fontsize=23, weight="bold", color="#111111", va="top")
    if subtitle:
        ax.text(0.48, 8.13, subtitle, fontsize=11.5, color=GRAY, va="top")
    return fig, ax


def rounded_box(ax, x, y, w, h, text, fc="white", ec=GRAY, fontsize=12, lw=1.3, color=INK):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color)
    return patch


def arrow(ax, start, end, color="#444444", lw=1.5, style="-|>", rad=0.0, label=None, label_offset=(0, 0)):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=10.5, color=color, ha="center", va="center")
    return patch


def matrix_block(
    ax,
    x,
    y,
    w,
    h,
    label,
    shape=None,
    rows=8,
    cols=8,
    face=GREEN,
    edge=GREEN_DARK,
    label_size=12,
    sections=None,
):
    """绘制带网格的矩阵；sections=[(比例, 颜色, 标签), ...]。"""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=1.4))

    if sections:
        cursor = x
        for ratio, color, section_label in sections:
            sw = w * ratio
            ax.add_patch(Rectangle((cursor, y), sw, h, facecolor=color, edgecolor="none", alpha=0.92))
            ax.text(cursor + sw / 2, y + h / 2, section_label, ha="center", va="center", fontsize=10.5)
            cursor += sw

    for i in range(1, rows):
        yy = y + h * i / rows
        ax.plot([x, x + w], [yy, yy], color=edge, lw=0.42, alpha=0.75)
    for j in range(1, cols):
        xx = x + w * j / cols
        ax.plot([xx, xx], [y, y + h], color=edge, lw=0.42, alpha=0.75)

    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=label_size, weight="bold")
    if shape:
        ax.text(x + w / 2, y - 0.18, shape, ha="center", va="top", fontsize=10.5, color=GRAY)


def stacked_matrix(ax, x, y, w, h, label, shape, layers=3, face=GREEN):
    for layer in range(layers - 1, -1, -1):
        dx = layer * 0.11
        dy = layer * 0.09
        matrix_block(
            ax,
            x + dx,
            y + dy,
            w,
            h,
            label if layer == 0 else "",
            shape if layer == 0 else None,
            rows=7,
            cols=7,
            face=face,
        )


def red_note(ax, x, y, text, rotation=-2, size=12, ha="left"):
    ax.text(x, y, text, color=RED, fontsize=size, style="italic", rotation=rotation, ha=ha, va="center")


def section_title(ax, x, y, text, color=ORANGE, size=27):
    ax.text(x, y, text, color=color, fontsize=size, weight="bold", ha="center", va="center")


def save(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def draw_core_idea():
    fig, ax = make_canvas(
        "1. MLA 的核心：把每个 token 的大 K/V，压缩成一个小型“信息胶囊”",
        "实际示例：H=16，Dn=256，Dr=48，Dv=256，Ckv=64；绿色网格表示需要保存的数据。",
    )

    ax.plot([8, 8], [0.7, 7.75], color="#bbbbbb", lw=1.2, ls="--")
    ax.text(3.95, 7.6, "普通完整 K/V 缓存", fontsize=18, weight="bold", ha="center")
    ax.text(12.0, 7.6, "MLA 压缩缓存", fontsize=18, weight="bold", ha="center")

    rounded_box(ax, 3.15, 6.55, 1.6, 0.65, "token 表示 $h_t$\n4096 维", fc="#fff7ed", ec=ORANGE)
    arrow(ax, (3.95, 6.53), (2.2, 5.95), color=ORANGE)
    arrow(ax, (3.95, 6.53), (5.7, 5.95), color=ORANGE)
    section_title(ax, 1.25, 5.65, "K")
    section_title(ax, 4.72, 5.65, "V")
    stacked_matrix(ax, 1.65, 2.1, 2.0, 3.65, "完整 K", "[H, Dn+Dr] = [16,304]", layers=3)
    stacked_matrix(ax, 5.10, 2.1, 2.0, 3.65, "完整 V", "[H, Dv] = [16,256]", layers=3)
    ax.text(4.0, 1.42, "每个 token 缓存", fontsize=12, ha="center", color=GRAY)
    ax.text(4.0, 1.05, "16 × (304 + 256) = 8960 个数", fontsize=16, ha="center", weight="bold")
    red_note(ax, 0.72, 3.15, "每个头都存一份！", rotation=5, size=14)
    arrow(ax, (2.0, 3.25), (1.63, 3.5), color=RED, lw=1.4, rad=-0.2)

    rounded_box(ax, 11.1, 6.55, 1.7, 0.65, "token 表示 $h_t$\n4096 维", fc="#fff7ed", ec=ORANGE)
    arrow(ax, (11.95, 6.52), (11.95, 6.02), color=ORANGE)
    rounded_box(ax, 10.95, 5.42, 2.0, 0.58, "$W_{kv\_a}$\n一次联合降维", fc="#fff7ed", ec=ORANGE)
    arrow(ax, (11.95, 5.4), (10.35, 4.85), color=ORANGE)
    arrow(ax, (11.95, 5.4), (13.45, 4.85), color=ORANGE)

    section_title(ax, 9.3, 4.62, "$c^{KV}$", size=24)
    matrix_block(ax, 9.7, 2.45, 1.45, 2.35, "压缩\nKV", "[Ckv] = [64]", rows=8, cols=5)
    section_title(ax, 12.45, 4.62, "$k^{R}$", size=24)
    matrix_block(ax, 12.85, 2.95, 1.15, 1.85, "位置\nKey", "[Dr] = [48]", rows=7, cols=4, face="#e2f0d9")
    ax.add_patch(
        FancyBboxPatch(
            (9.35, 1.75), 5.05, 3.45, boxstyle="round,pad=0.08", fill=False, edgecolor=RED, lw=2.0, linestyle="--"
        )
    )
    red_note(ax, 11.9, 1.48, "KV Cache 只保存这两个小块", rotation=-2, size=14, ha="center")
    ax.text(11.9, 1.05, "每个 token：64 + 48 = 112 个数", fontsize=16, ha="center", weight="bold")
    ax.text(8.0, 0.48, "8960 ÷ 112 = 80×", color=RED, fontsize=24, weight="bold", ha="center")
    red_note(ax, 13.7, 6.42, "需要计算时再恢复，\n或把恢复矩阵吸收到 Q 里", rotation=2, size=12)
    arrow(ax, (13.65, 6.3), (13.0, 5.15), color=RED, lw=1.3, rad=0.15)

    save(fig, "mla_01_core_idea.png")


def draw_q_kv_decomposition():
    fig, ax = make_canvas(
        "2. 一个 token 如何被拆成 Q、压缩 KV 和位置 Key",
        "关键观察：Q 每个头都不同；K 的位置部分在所有头间共享；K_nope 与 V 从同一个 64 维 c_KV 解压。",
    )

    rounded_box(ax, 7.25, 7.2, 1.5, 0.6, "$x_t$\n4096 维", fc="#fff7ed", ec=ORANGE, fontsize=13)
    arrow(ax, (7.8, 7.18), (4.0, 6.62), color=ORANGE, label="Query 路径", label_offset=(0, 0.2))
    arrow(ax, (8.2, 7.18), (12.0, 6.62), color=ORANGE, label="KV 路径", label_offset=(0, 0.2))

    rounded_box(ax, 2.9, 5.95, 2.0, 0.62, "$wq_a$：4096 → 128", fc="#fff7ed", ec=ORANGE)
    matrix_block(ax, 3.35, 4.42, 1.15, 1.25, "$q^C$", "[Cq]=[128]", rows=7, cols=5)
    arrow(ax, (3.9, 5.94), (3.9, 5.7), color=BLUE)
    ax.text(4.77, 4.98, "RMSNorm", fontsize=11, color=BLUE, rotation=90, va="center")
    arrow(ax, (3.9, 4.38), (3.9, 4.04), color=BLUE)
    rounded_box(ax, 2.75, 3.38, 2.3, 0.62, "$wq_b$：128 → 16×304", fc="#eef2ff", ec=BLUE)
    arrow(ax, (3.9, 3.36), (3.9, 3.04), color=BLUE)
    matrix_block(
        ax,
        1.45,
        1.38,
        5.0,
        1.58,
        "",
        "Q：[H, Dn+Dr] = [16,304]",
        rows=6,
        cols=12,
        sections=[(0.84, "#c6e0b4", "$q_{nope}$\n256 维"), (0.16, "#ffe699", "$q_{pe}$\n48 维")],
    )
    section_title(ax, 0.82, 2.15, "Q", size=30)
    red_note(ax, 5.55, 3.35, "先升维，再按 256 | 48 切开", rotation=-3, size=12)
    arrow(ax, (5.8, 3.2), (5.45, 2.92), color=RED, rad=0.15)

    rounded_box(ax, 10.85, 5.95, 2.3, 0.62, "$wkv_a$：4096 → 64+48", fc="#fff7ed", ec=ORANGE)
    arrow(ax, (12.0, 5.93), (10.45, 5.35), color=GREEN_DARK)
    arrow(ax, (12.0, 5.93), (13.65, 5.35), color=ORANGE)
    matrix_block(ax, 9.65, 3.85, 1.55, 1.45, "$c^{KV}$", "[64]", rows=7, cols=5)
    matrix_block(ax, 13.15, 4.18, 1.15, 1.12, "$k_{pe}$", "[48]", rows=6, cols=4, face="#ffe699", edge="#bf9000")
    red_note(ax, 13.7, 5.72, "所有头共享", rotation=3, size=12, ha="center")

    arrow(ax, (10.43, 3.82), (10.43, 3.43), color=GREEN_DARK)
    rounded_box(ax, 9.45, 2.78, 1.95, 0.6, "kv_norm + $wkv_b$", fc="#ecfdf5", ec=GREEN_DARK)
    arrow(ax, (10.43, 2.75), (9.0, 2.58), color=GREEN_DARK)
    arrow(ax, (10.43, 2.75), (11.8, 2.58), color=GREEN_DARK)
    matrix_block(ax, 7.85, 1.20, 2.3, 1.35, "$k_{nope}$", "[H,Dn]=[16,256]", rows=6, cols=9)
    matrix_block(ax, 10.75, 1.20, 2.3, 1.35, "$V$", "[H,Dv]=[16,256]", rows=6, cols=9)
    section_title(ax, 7.35, 1.90, "K", size=30)
    section_title(ax, 10.35, 1.90, "V", size=30)

    arrow(ax, (13.72, 4.15), (14.25, 3.43), color=ORANGE, rad=0.12)
    rounded_box(ax, 13.35, 2.78, 1.8, 0.6, "RoPE 旋转", fc="#fff7ed", ec=ORANGE)
    arrow(ax, (14.25, 2.75), (14.25, 2.58), color=ORANGE)
    matrix_block(ax, 13.65, 1.20, 1.2, 1.35, "$k_{pe}^{R}$", "[Dr]=[48]", rows=6, cols=4, face="#ffe699", edge="#bf9000")

    ax.text(9.0, 0.38, "$K = [k_{nope},\;k_{pe}^{R}]$，每个头 256+48=304 维", fontsize=12.5, weight="bold", ha="center")
    red_note(ax, 6.92, 5.0, "RoPE 只碰黄色的 48 维！", rotation=-3, size=14, ha="center")
    arrow(ax, (7.7, 4.82), (13.3, 4.75), color=RED, lw=1.4, rad=0.18)
    arrow(ax, (6.2, 4.8), (5.95, 2.75), color=RED, lw=1.4, rad=-0.15)

    save(fig, "mla_02_q_kv_decomposition.png")


def draw_naive_attention():
    fig, ax = make_canvas(
        "3. naive 模式：解压完整 K/V，再做标准多头注意力",
        "为了能看清矩阵，图中使用玩具维度：B=1，H=2，S=T=4，Dq=Dn+Dr=5，Dv=3。代码中的运算完全相同。",
    )

    section_title(ax, 0.65, 6.85, "Q", size=32)
    stacked_matrix(ax, 1.15, 5.25, 1.65, 2.15, "Q", "[H,S,Dq] = [2,4,5]", layers=2, face="#c6e0b4")
    ax.text(3.1, 6.25, "×", fontsize=32, color=ORANGE, weight="bold")
    section_title(ax, 3.75, 7.5, "$K^T$", size=27)
    stacked_matrix(ax, 3.55, 5.25, 2.2, 2.15, "$K^T$", "[H,Dq,T] = [2,5,4]", layers=2, face="#c6e0b4")
    ax.text(6.05, 6.25, "÷ $\sqrt{Dq}$", fontsize=19, color=ORANGE, weight="bold")
    arrow(ax, (6.8, 6.3), (7.35, 6.3), color="#444444")
    stacked_matrix(ax, 7.55, 5.25, 2.15, 2.15, "scores", "[H,S,T] = [2,4,4]", layers=2, face="#e2f0d9")

    ax.text(9.95, 6.25, "+", fontsize=30, color=ORANGE, weight="bold")
    matrix_block(ax, 10.45, 5.25, 2.15, 2.15, "causal\nmask", "[S,T] = [4,4]", rows=4, cols=4, face="#a9d18e")
    for row in range(4):
        for col in range(row + 1, 4):
            x = 10.45 + 2.15 * (col + 0.5) / 4
            y = 5.25 + 2.15 * (3 - row + 0.5) / 4
            ax.text(x, y, "$-\infty$", fontsize=8, color="#8b0000", ha="center", va="center")
    arrow(ax, (12.82, 6.3), (13.35, 6.3), color="#444444")
    rounded_box(ax, 13.5, 5.78, 1.65, 0.95, "softmax\n沿 T 归一化", fc="#fff7ed", ec=ORANGE)

    red_note(ax, 8.6, 7.65, "每个 head 得到一张 4×4 注意力表", rotation=-2, size=13, ha="center")
    arrow(ax, (8.65, 7.48), (8.65, 7.28), color=RED)

    arrow(ax, (14.3, 5.74), (12.8, 4.32), color=ORANGE, rad=0.08)
    stacked_matrix(ax, 10.85, 2.15, 2.15, 2.15, "A", "softmax(scores+mask)", layers=2, face="#ffe699")
    ax.text(9.9, 3.15, "=", fontsize=28, color=ORANGE, weight="bold")
    ax.text(13.3, 3.15, "×", fontsize=30, color=ORANGE, weight="bold")
    section_title(ax, 13.95, 4.58, "V", size=30)
    stacked_matrix(ax, 13.7, 2.15, 1.65, 2.15, "V", "[H,T,Dv] = [2,4,3]", layers=2, face="#c6e0b4")

    arrow(ax, (10.65, 3.15), (9.55, 3.15), color="#444444", style="<|-")
    stacked_matrix(ax, 7.1, 2.15, 2.15, 2.15, "输出 O", "[H,S,Dv] = [2,4,3]", layers=2, face="#e2f0d9")
    ax.text(5.95, 3.15, "→ 拼接 H 个头 → $w_o$ → [B,S,dim]", fontsize=15, weight="bold", ha="right")

    red_note(ax, 2.5, 4.45, "naive 的“贵”不在公式，\n而在缓存了完整 K 和 V", rotation=2, size=14, ha="center")
    arrow(ax, (3.2, 4.35), (4.4, 5.12), color=RED, rad=-0.2)
    arrow(ax, (3.3, 4.15), (14.1, 4.32), color=RED, rad=0.18)

    ax.text(8, 0.73, "$O=softmax(QK^T/\sqrt{Dq}+M)V$", fontsize=24, ha="center", weight="bold")
    save(fig, "mla_03_naive_attention.png")


def draw_weight_absorption():
    fig, ax = make_canvas(
        "4. 权重吸收：不恢复完整 K/V，也能得到同一个注意力结果",
        "矩阵乘法满足结合律。红色括号表示改变计算顺序；虚线框表示真正需要放入 KV Cache 的内容。",
    )

    ax.text(0.55, 7.5, "① 计算分数：把 $W_K$ 从 Key 一侧“搬”到 Query 一侧", fontsize=17, weight="bold")
    ax.text(0.7, 6.88, "普通顺序", fontsize=14, color=BLUE, weight="bold")
    matrix_block(ax, 1.0, 5.5, 1.5, 0.8, "$q$", "[1,Dn]", rows=2, cols=7)
    ax.text(2.72, 5.9, "×", fontsize=24, color=ORANGE, weight="bold")
    matrix_block(ax, 3.15, 5.15, 1.45, 1.5, "$W_K$", "[Dn,Ckv]", rows=7, cols=5, face="#ffe699", edge="#bf9000")
    ax.text(4.85, 5.9, "×", fontsize=24, color=ORANGE, weight="bold")
    matrix_block(ax, 5.3, 5.35, 2.25, 1.1, "$c^T$", "[Ckv,T]", rows=5, cols=9)
    ax.text(7.82, 5.9, "=", fontsize=24, color=ORANGE, weight="bold")
    matrix_block(ax, 8.35, 5.5, 2.4, 0.8, "scores_nope", "[1,T]", rows=2, cols=10, face="#e2f0d9")
    ax.add_patch(FancyBboxPatch((3.0, 4.92), 4.72, 1.95, boxstyle="round,pad=0.06", fill=False, ec=RED, lw=2, ls="--"))
    red_note(ax, 5.35, 4.66, "普通做法：先恢复完整 K^T = W_K · c^T", rotation=-2, size=12.5, ha="center")

    ax.text(11.0, 6.88, "MLA 顺序", fontsize=14, color=GREEN_DARK, weight="bold")
    matrix_block(ax, 11.15, 5.5, 1.3, 0.8, "$q$", "[1,Dn]", rows=2, cols=7)
    ax.text(12.67, 5.9, "×", fontsize=24, color=ORANGE, weight="bold")
    matrix_block(ax, 13.05, 5.15, 1.35, 1.5, "$W_K$", "[Dn,Ckv]", rows=7, cols=5, face="#ffe699", edge="#bf9000")
    ax.add_patch(FancyBboxPatch((10.98, 4.94), 3.56, 1.9, boxstyle="round,pad=0.06", fill=False, ec=RED, lw=2))
    arrow(ax, (14.48, 5.9), (15.0, 5.9), color=ORANGE)
    ax.text(14.95, 5.42, "再与 $c^T$ 点积", fontsize=10.5, color=GRAY, ha="right")
    red_note(ax, 12.8, 4.62, "先得到压缩维 Query：q·W_K", rotation=2, size=12.5, ha="center")
    ax.text(8.0, 7.02, "$q(W_Kc^T)=(qW_K)c^T$", fontsize=24, color=ORANGE, weight="bold", ha="center")

    ax.plot([0.55, 15.45], [4.25, 4.25], color="#bbbbbb", lw=1.0, ls="--")
    ax.text(0.55, 3.85, "② 聚合 Value：先对小 c 加权，再通过 $W_V$ 解压", fontsize=17, weight="bold")

    matrix_block(ax, 1.05, 2.25, 2.0, 0.75, "$A$", "[S,T]", rows=3, cols=9, face="#ffe699", edge="#bf9000")
    ax.text(3.3, 2.62, "×", fontsize=24, color=ORANGE, weight="bold")
    matrix_block(ax, 3.75, 1.98, 1.7, 1.3, "$c$", "[T,Ckv]", rows=7, cols=5)
    ax.text(5.72, 2.62, "×", fontsize=24, color=ORANGE, weight="bold")
    matrix_block(ax, 6.15, 1.82, 1.5, 1.62, "$W_V^T$", "[Ckv,Dv]", rows=5, cols=7, face="#ffe699", edge="#bf9000")
    ax.text(7.95, 2.62, "=", fontsize=24, color=ORANGE, weight="bold")
    matrix_block(ax, 8.45, 2.05, 2.0, 1.15, "$O$", "[S,Dv]", rows=6, cols=7, face="#e2f0d9")
    ax.add_patch(FancyBboxPatch((0.88, 1.63), 4.73, 1.98, boxstyle="round,pad=0.06", fill=False, ec=RED, lw=2))
    red_note(ax, 3.1, 1.38, "先算 A·c，只聚合 64 维 latent", rotation=-2, size=12.5, ha="center")

    ax.add_patch(FancyBboxPatch((11.15, 1.22), 3.75, 2.35, boxstyle="round,pad=0.08", fc=GREEN_LIGHT, ec=GREEN_DARK, lw=1.5))
    ax.text(13.02, 3.2, "真正缓存", fontsize=15, color=GREEN_DARK, weight="bold", ha="center")
    matrix_block(ax, 11.55, 1.85, 1.25, 0.9, "$c^{KV}$", "64 维", rows=5, cols=5)
    matrix_block(ax, 13.45, 2.0, 0.95, 0.75, "$k^R$", "48 维", rows=5, cols=4, face="#ffe699", edge="#bf9000")
    ax.text(13.0, 1.48, "112 数/token，而不是 8960", fontsize=12.5, color=RED, weight="bold", ha="center")

    ax.text(8.0, 0.58, "MLA 没有删掉 K/V 的表达能力；它只是改变乘法顺序，避免把完整 K/V 长期放进缓存。", fontsize=15, weight="bold", ha="center")
    save(fig, "mla_04_weight_absorption.png")


def draw_incremental_cache():
    fig, ax = make_canvas(
        "5. start_pos 与 KV Cache：prefill 后逐 token 解码",
        "绿色格子是历史 token，橙色格子是当前 token。T 表示参与本次注意力的 Key/Value 总长度。",
    )

    def cache_row(y, title, active_until, current=None, wrong_only_first=False):
        ax.text(0.55, y + 0.35, title, fontsize=15, weight="bold", va="center")
        start_x = 3.0
        labels = ["0", "1", "2", "3", "…", "97", "98", "99", "100"]
        for i, label in enumerate(labels):
            x = start_x + i * 1.15
            if wrong_only_first:
                fc = "#fecaca" if i == 0 else "white"
                ec = RED
            elif current is not None and label == str(current):
                fc = "#f9cb9c"
                ec = ORANGE
            elif label == "…" or (label.isdigit() and int(label) <= active_until):
                fc = GREEN
                ec = GREEN_DARK
            else:
                fc = "white"
                ec = "#aaaaaa"
            ax.add_patch(Rectangle((x, y), 0.92, 0.72, facecolor=fc, edgecolor=ec, lw=1.2))
            ax.text(x + 0.46, y + 0.36, label, ha="center", va="center", fontsize=11)
        return start_x, start_x + (len(labels) - 1) * 1.15 + 0.92

    x0, x1 = cache_row(6.65, "Prefill", active_until=99)
    ax.text(3.0, 7.62, "输入 S=100 个 token：start_pos=0，end_pos=100", fontsize=13, color=BLUE)
    prefill_end = x0 + 7 * 1.15 + 0.92  # 图中 token 99 的右边界
    ax.plot([x0, x0, prefill_end, prefill_end], [6.56, 6.51, 6.51, 6.56], color=RED, lw=1.6)
    arrow(ax, ((x0 + prefill_end) / 2, 6.51), ((x0 + prefill_end) / 2, 6.05), color=RED, lw=1.4)
    ax.text((x0 + prefill_end) / 2, 5.87, "写入 cache[0:100]", ha="center", color=RED, fontsize=13)

    x0, x1 = cache_row(4.65, "第 101 个 token", active_until=99, current=100)
    ax.text(3.0, 5.62, "输入 S=1：start_pos=100，end_pos=101", fontsize=13, color=BLUE)
    ax.plot([x0, x0, x1, x1], [4.56, 4.51, 4.51, 4.56], color=RED, lw=1.6)
    arrow(ax, ((x0 + x1) / 2, 4.51), ((x0 + x1) / 2, 4.05), color=RED, lw=1.4)
    ax.text((x0 + x1) / 2, 3.87, "Query 应读取 cache[0:101]，所以 T=101", ha="center", color=RED, fontsize=13)
    red_note(ax, 13.4, 5.2, "RoPE 位置应是 100", rotation=-3, size=13)
    arrow(ax, (13.65, 5.05), (12.48, 4.98), color=RED, rad=0.15)

    x0, x1 = cache_row(2.25, "当前错误切片", active_until=0, wrong_only_first=True)
    ax.text(3.0, 3.22, "代码使用 cache[:seq_len]；此时 seq_len=1", fontsize=13, color="#991b1b")
    red_note(ax, 8.2, 1.85, "结果只读到 token 0，历史 1…100 全丢了！", rotation=1, size=15, ha="center")

    rounded_box(ax, 0.62, 0.55, 4.3, 0.92, "正确区间\n写入 [start_pos:end_pos]\n读取 [:end_pos]", fc=GREEN_LIGHT, ec=GREEN_DARK, fontsize=12.5)
    rounded_box(ax, 5.78, 0.55, 4.3, 0.92, "正确 RoPE\n位置 [start_pos:end_pos]\n而不是每次从 0 开始", fc="#fff7ed", ec=ORANGE, fontsize=12.5)
    rounded_box(ax, 10.95, 0.55, 4.3, 0.92, "先修拼写\nend_pos = start_pos + seq_len\n原代码 seqlen 未定义", fc="#fef2f2", ec=RED, fontsize=12.5)

    save(fig, "mla_05_incremental_cache.png")


if __name__ == "__main__":
    draw_core_idea()
    draw_q_kv_decomposition()
    draw_naive_attention()
    draw_weight_absorption()
    draw_incremental_cache()
