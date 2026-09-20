#!/usr/bin/env python3
"""
一次性脚本：为 S30 综述生成"不同量化位宽下 7B 模型显存占用对比"柱状图。
数据来源：7B(=7e9)参数 × 每参数字节数，字节数分别为
FP32=4, FP16/BF16=2, INT8=1, INT4=0.5。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["FP32", "FP16 / BF16", "INT8", "INT4"]
bytes_per_param = [4, 2, 1, 0.5]
params = 7e9
gib = [(params * b) / (1024 ** 3) for b in bytes_per_param]

colors = ["#F44336", "#FF9800", "#4CAF50", "#2196F3"]

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
bars = ax.bar(labels, gib, color=colors, width=0.6)

for bar, val in zip(bars, gib):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.2f} GiB",
             ha="center", va="bottom", fontsize=10, color="#333333")

ax.set_title("7B Model Memory Footprint by Weight Precision", fontsize=12)
ax.set_ylabel("Memory (GiB)")
ax.set_ylim(0, max(gib) * 1.2)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("public/quant_memory_bitwidth_bar_7b.png")
print("saved to public/quant_memory_bitwidth_bar_7b.png")
