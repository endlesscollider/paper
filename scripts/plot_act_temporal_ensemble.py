#!/usr/bin/env python3
"""
一次性脚本：为 ACT / ALOHA 深度解析系列生成"时间集成（temporal ensemble）指数权重"图。

公式来源：Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware (arXiv:2304.13705) 第 IV-A 节：w_i = exp(-m * i)。对某个固定的目标时刻 t，
缓冲区里堆着多个来自不同预测时刻的动作；按"预测发生的顺序"编号，i = 0 是最早
（最旧）的那次预测，权重最大（exp(0) = 1），i 越大越新、权重越小。m 控制新观测
被纳入的速度：m 越小曲线越平坦，较新的预测仍保留可观权重，新观测生效越快。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

i = np.arange(0, 40, 1)
ms = [0.01, 0.05, 0.1]
colors = ["#2196F3", "#4CAF50", "#F44336"]

fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
for m, c in zip(ms, colors):
    w = np.exp(-m * i)
    w = w / w.sum()
    ax.plot(i, w, marker="o", markersize=3, linewidth=2, color=c,
            label=f"m = {m}")

ax.set_xlabel("Chunk age index i  (0 = oldest prediction for timestep t)", fontsize=11)
ax.set_ylabel("Normalized weight  w_i / sum(w_i)", fontsize=12)
ax.set_title("Temporal Ensemble: Exponential Weighting", fontsize=12)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out = "public/act_temporal_ensemble_weights.png"
fig.savefig(out)
print(f"saved to {out}")
