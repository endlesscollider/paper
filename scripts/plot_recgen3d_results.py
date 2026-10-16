#!/usr/bin/env python3
"""
为 RecGen3D 精读生成结果图。

数据来源：RecGen3D（arXiv 2604.01479，SIGGRAPH Asia 2026）Table 1。
左图：Chamfer-L2（越低越好）；右图：Voxel IoU（越高越好）。
在 Toys4K 与 GSO 两个基准上对比几何生成质量，RecGen3D（Ours）用强调色。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["LucidFusion", "Hunyuan3D-MV", "TRELLIS-M", "TRELLIS-S",
           "SAM 3D", "ReconViaGen", "RecGen3D\n(Ours)"]
labels = [m.replace("\n", " ") for m in methods]

# Chamfer-L2 (↓)
cd_toys = [0.1333, 0.0513, 0.0343, 0.0309, 0.0354, 0.0281, 0.0175]
cd_gso = [0.1203, 0.0667, 0.0318, 0.0391, 0.0296, 0.0290, 0.0192]
# Voxel IoU (↑)
iou_toys = [0.1035, 0.6311, 0.6318, 0.6583, 0.6061, 0.7229, 0.8030]
iou_gso = [0.1317, 0.5980, 0.6489, 0.6486, 0.6339, 0.6918, 0.8155]

x = np.arange(len(methods))
w = 0.38
colors = ["#2196F3", "#FF9800"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
b1 = ax.bar(x - w / 2, cd_toys, w, label="Toys4K", color=colors[0])
b2 = ax.bar(x + w / 2, cd_gso, w, label="GSO", color=colors[1])
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Chamfer-L2 (lower is better)")
ax.set_title("Geometry error: Chamfer-L2", fontsize=11.5)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9)

ax = axes[1]
b1 = ax.bar(x - w / 2, iou_toys, w, label="Toys4K", color=colors[0])
b2 = ax.bar(x + w / 2, iou_gso, w, label="GSO", color=colors[1])
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Voxel IoU (higher is better)")
ax.set_ylim(0, 0.95)
ax.set_title("Volumetric overlap: Voxel IoU", fontsize=11.5)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9)

fig.suptitle("RecGen3D vs 3D generation/reconstruction baselines on sparse-view objects", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("public/recgen3d_results_bar.png", dpi=150)
print("saved public/recgen3d_results_bar.png")
