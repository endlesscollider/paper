#!/usr/bin/env python3
"""
一次性脚本：为 ACT / ALOHA 深度解析系列生成"仿真任务成功率对比"分组柱状图。

数据来源：Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware (arXiv:2304.13705) Table I，两个仿真任务在 **人类演示数据** 上的最终
成功率（右半列）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["BC-ConvMLP", "BeT", "RT-1", "VINN", "ACT (ours)"]
cube = [3, 16, 4, 17, 82]
insert = [1, 13, 2, 11, 60]

x = np.arange(len(methods))
width = 0.38
fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
b1 = ax.bar(x - width / 2, cube, width, label="Transfer Cube", color="#2196F3")
b2 = ax.bar(x + width / 2, insert, width, label="Bimanual Insertion", color="#FF9800")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{int(b.get_height())}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("Simulated Tasks Trained on Human Demos", fontsize=12)
ax.set_ylim(0, 95)
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out = "public/act_success_rates.png"
fig.savefig(out)
print(f"saved to {out}")
