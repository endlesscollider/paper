#!/usr/bin/env python3
"""
为 OVOW 精读生成"迭代细化轮数 vs 实例几何精度"折线图。

数据来源：OVOW 论文 Table 3 超参鲁棒性（held-out 验证集，指标为 IoU-B）。
N_iter ∈ {1,2,3,5} → IoU-B ∈ {0.58, 0.72, 0.78, 0.79}。
这张图解释为什么默认取 N_iter=3：精度在第 3 轮饱和，第 5 轮只涨 0.01 却要
多跑两轮渲染-匹配-优化循环（每轮都含 FoundationPose + 渲染 + RoMaV2 匹配）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n_iter = [1, 2, 3, 5]
iou_b = [0.58, 0.72, 0.78, 0.79]

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
ax.plot(n_iter, iou_b, marker="o", markersize=7, linewidth=2.2, color="#1E88E5")

# 标注默认值所在的拐点
ax.scatter([3], [0.78], s=180, facecolors="none", edgecolors="#E53935", linewidths=2.2, zorder=5)
ax.annotate("default $N_{iter}=3$\n(accuracy saturates)",
            xy=(3, 0.78), xytext=(3.25, 0.64),
            fontsize=10, color="#E53935",
            arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.6))

for x, y in zip(n_iter, iou_b):
    ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=9.5, color="#333333")

ax.set_title("Instance Geometry vs. Scale-Refinement Iterations", fontsize=12)
ax.set_xlabel("$N_{iter}$ (render-match-optimize rounds)")
ax.set_ylabel("IoU-B (higher is better)")
ax.set_xticks(n_iter)
ax.set_ylim(0.5, 0.9)
ax.grid(linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("public/ovow_iter_saturation.png")
print("saved to public/ovow_iter_saturation.png")
