#!/usr/bin/env python3
"""
一次性脚本：为 S30 综述生成"RTN / GPTQ / AWQ 在 LLaMA-7B 上的 WikiText2 困惑度对比"分组柱状图。
数据来源：AWQ 论文（Lin et al., arXiv:2306.00978）Table 4，LLaMA-7B 一列；
FP16 基线困惑度取自公开复现报告（约 5.68，与 GPTQ 官方仓库一致）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

groups = ["INT4 (g128)", "INT3 (g128)"]
methods = ["RTN", "GPTQ", "AWQ"]
data = {
    "RTN": [5.73, 6.66],
    "GPTQ": [5.69, 6.43],
    "AWQ": [5.60, 6.24],
}
colors = {"RTN": "#607D8B", "GPTQ": "#FF9800", "AWQ": "#2196F3"}
fp16_baseline = 5.68

x = np.arange(len(groups))
width = 0.25

fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=150)
for i, m in enumerate(methods):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, data[m], width, label=m, color=colors[m])
    for b, v in zip(bars, data[m]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8.5)

ax.axhline(fp16_baseline, color="#F44336", linestyle="--", linewidth=1.3,
            label=f"FP16 baseline ({fp16_baseline})")

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("WikiText2 Perplexity (lower is better)")
ax.set_title("LLaMA-7B: RTN vs GPTQ vs AWQ", fontsize=12)
ax.set_ylim(5.4, 7.0)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("public/quant_awq_gptq_rtn_llama7b.png")
print("saved to public/quant_awq_gptq_rtn_llama7b.png")
