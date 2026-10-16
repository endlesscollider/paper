#!/usr/bin/env python3
"""
为 RecGen 精读生成两张结果图。

数据来源：RecGen 论文（arXiv 2604.27106）Table 1 与正文 4.1 节 / 附录 A.1。
图 1（recgen_results_bar.png）：形状重建 CD_norm（越低越好）与位姿精度 ADD-SB@0.05（越高越好）
    在 HB / ReOcS / LMO / ArtVIP 四个基准上，对比 SAM3D 与 RecGen 单视图/双视图。
图 2（recgen_occlusion_robustness.png）：按遮挡程度分箱的 ADD-SB（越低越好），
    对比 RecGen 与 SAM3D 在 0–3% 与 40–70% 两个极端遮挡区间上的退化幅度。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- 图 1：跨数据集形状/位姿对比 ----------------
datasets = ["HB", "ReOcS", "LMO", "ArtVIP\n(parts)"]
methods = ["Any6D\n(TRELLIS)", "SAM3D", "RecGen\n(1-view)", "RecGen\n(2-view)"]
colors = ["#B0BEC5", "#607D8B", "#2196F3", "#F44336"]

# CD_norm (↓)
cd = np.array([
    [0.106, 0.068, 0.116, 0.090],  # Any6D (TRELLIS)
    [0.033, 0.026, 0.057, 0.056],  # SAM3D
    [0.032, 0.019, 0.050, 0.026],  # RecGen 1-view
    [0.029, 0.018, 0.056, 0.024],  # RecGen 2-view
])
# ADD-SB @0.05 (↑)
adds = np.array([
    [33.8, 47.4, 16.9, 37.7],
    [54.6, 43.6, 17.6, 45.8],
    [73.8, 89.5, 50.0, 84.0],
    [74.2, 91.1, 55.6, 86.4],
])

x = np.arange(len(datasets))
w = 0.2

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i, m in enumerate(methods):
    bars = ax.bar(x + (i - 1.5) * w, cd[i], w, label=m.replace("\n", " "), color=colors[i])
    for b, v in zip(bars, cd[i]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Normalized Chamfer Distance (lower is better)")
ax.set_title("Shape quality: CD_norm", fontsize=11.5)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=8.5, ncol=2)

ax = axes[1]
for i, m in enumerate(methods):
    bars = ax.bar(x + (i - 1.5) * w, adds[i], w, label=m.replace("\n", " "), color=colors[i])
    for b, v in zip(bars, adds[i]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("ADD-SB @0.05 (%, higher is better)")
ax.set_title("6-DoF pose accuracy: ADD-SB@0.05", fontsize=11.5)
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=8.5, ncol=2)

fig.suptitle("RecGen vs baselines across object- and part-level benchmarks", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("public/recgen_results_bar.png", dpi=150)
print("saved public/recgen_results_bar.png")

# ---------------- 图 2：遮挡鲁棒性 ----------------
groups = ["Objects\n0-3% occ.", "Objects\n40-70% occ.", "Parts\n0-3% occ.", "Parts\n40-70% occ."]
recgen = [0.044, 0.073, 0.030, 0.044]
sam3d = [0.053, 0.116, 0.059, 0.092]

x = np.arange(len(groups))
w = 0.36
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w / 2, recgen, w, label="RecGen", color="#F44336")
b2 = ax.bar(x + w / 2, sam3d, w, label="SAM3D", color="#607D8B")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, color="#333333")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("ADD-SB (lower is better)")
ax.set_title("Occlusion robustness: pose error vs. occlusion severity", fontsize=11.5)
# SAM3D 的退化幅度更大：用两侧误差框标出增长
for i, (r, s) in enumerate(zip(recgen, sam3d)):
    ax.annotate("", xy=(x[i] + w / 2, s), xytext=(x[i] + w / 2, r),
                arrowprops=dict(arrowstyle="<->", color="#455A64", lw=1.0))
    ax.text(x[i] + w / 2 + 0.05, (r + s) / 2, f"+{s - r:.3f}", fontsize=8, color="#455A64")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
ax.set_ylim(0, 0.135)
fig.tight_layout()
fig.savefig("public/recgen_occlusion_robustness.png", dpi=150)
print("saved public/recgen_occlusion_robustness.png")
