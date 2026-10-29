import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 数据来自 PhysTwin 论文 Table 1 与 Table 3
methods = ["Spring-Gauss", "GS-Dynamics", "PhysTwin"]
colors = ["#9E9E9E", "#FF9800", "#2196F3"]

# ---------- Panel A/B: 重建与重仿真 / 未来预测 ----------
recon_cd = [0.041, 0.014, 0.005]
recon_iou = [57.6, 72.1, 84.4]
future_cd = [0.062, 0.041, 0.012]
future_iou = [46.4, 49.8, 72.5]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.suptitle("PhysTwin vs Baselines on Two Tasks", fontsize=14, fontweight="bold")

def grouped(ax, values, title, ylabel, ylim, fmt="{:.3f}"):
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (ylim[1] - ylim[0]) * 0.015,
                fmt.format(v), ha="center", va="bottom", fontsize=9)

grouped(axes[0, 0], recon_cd, "Reconstruction & Resimulation: CD", "Chamfer Distance (lower is better)", (0, 0.047))
grouped(axes[0, 1], recon_iou, "Reconstruction & Resimulation: IoU", "IoU % (higher is better)", (0, 95), "{:.1f}")
grouped(axes[1, 0], future_cd, "Future Prediction: CD", "Chamfer Distance (lower is better)", (0, 0.071))
grouped(axes[1, 1], future_iou, "Future Prediction: IoU", "IoU % (higher is better)", (0, 82), "{:.1f}")

plt.tight_layout()
plt.savefig("public/phystwin_task_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- Panel C: 分层优化消融 ----------
variants = ["Zero-order only", "First-order only", "Hierarchical (full)"]
abl_cd = [0.007, 0.008, 0.005]
abl_track = [0.012, 0.012, 0.009]
abl_iou = [80.2, 82.7, 84.4]
abl_colors = ["#FF9800", "#9C27B0", "#2196F3"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
fig.suptitle("Ablation: Sparse-to-Dense Optimization", fontsize=14, fontweight="bold")

metrics = [
    (abl_cd, "Chamfer Distance", "CD (lower is better)", (0, 0.0095), "{:.3f}"),
    (abl_track, "Tracking Error", "Track Error (lower is better)", (0, 0.0145), "{:.3f}"),
    (abl_iou, "IoU", "IoU % (higher is better)", (0, 90), "{:.1f}"),
]
for ax, (vals, title, ylabel, ylim, fmt) in zip(axes, metrics):
    x = np.arange(len(variants))
    bars = ax.bar(x, vals, color=abl_colors, width=0.62, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (ylim[1] - ylim[0]) * 0.02,
                fmt.format(v), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("public/phystwin_ablation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved")
