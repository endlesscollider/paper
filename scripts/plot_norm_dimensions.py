#!/usr/bin/env python3
"""
一次性脚本：画出 BatchNorm / LayerNorm / InstanceNorm / GroupNorm 在
一个 [N=2, C=4, L=2] 的示例张量上，分别在哪些 (N, C) 格子间共享均值/方差统计量。

每个小方块代表一个 (样本 n, 通道 c) 对（内部还有 L=2 个空间位置，颜色相同代表
这些格子的所有数值被合并到同一组去算均值和方差）。

用法：
    uv run --with matplotlib --with numpy python scripts/plot_norm_dimensions.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

N, C = 2, 4  # 2 个样本，4 个通道

# 每种方法的分组矩阵：group_id[c, n]
bn_groups = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])          # 按通道分组（跨 N）
ln_groups = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])          # 按样本分组（跨 C）
in_groups = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])          # 每个 (n,c) 各自一组
gn_groups = np.array([[0, 1], [0, 1], [2, 3], [2, 3]])          # 每个样本内，通道两两分组

configs = [
    ("BatchNorm\n(group by channel, across samples)", bn_groups),
    ("LayerNorm\n(group by sample, across channels)", ln_groups),
    ("InstanceNorm\n(each sample x channel alone)", in_groups),
    ("GroupNorm, G=2\n(within sample, channels paired)", gn_groups),
]

cmap = plt.get_cmap("tab10")

fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))

for ax, (title, groups) in zip(axes, configs):
    for c in range(C):
        for n in range(N):
            gid = groups[c, n]
            color = cmap(gid % 10)
            rect = mpatches.Rectangle((n, C - 1 - c), 1, 1, facecolor=color,
                                       edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(n + 0.5, C - 1 - c + 0.5, f"n={n} c={c}",
                     ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xlim(0, N)
    ax.set_ylim(0, C)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle("Which (sample, channel) cells share mean/variance statistics?", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
output = "public/norm_dimension_comparison.png"
plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"✅ {output}")
