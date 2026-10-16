#!/usr/bin/env python3
"""
为「VGGT 前馈式多视图三维重建」前置知识生成配图。

数据来源：VGGT（arXiv 2503.11651）Table 1，相机位姿估计（RealEstate10K，10 帧，AUC@30）。
横轴为运行时间（秒，对数轴），纵轴为位姿精度 AUC@30（越高越好）。
展示"前馈一次前向"在精度与速度上同时超越需要后优化的传统方法，以及再叠加 BA 仍可提升。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (name, AUC@30, time_seconds, is_ours, is_optimization)
data = [
    ("Colmap+SPSG", 45.2, 15.0, False, True),
    ("PixSfM", 49.4, 25.0, False, True),
    ("PoseDiff", 48.0, 7.0, False, True),
    ("DUSt3R", 67.7, 7.0, False, True),
    ("MASt3R", 76.4, 9.0, False, True),
    ("VGGSfM v2", 78.9, 10.0, False, True),
    ("MV-DUSt3R", 71.3, 0.6, False, False),
    ("CUT3R", 75.3, 0.6, False, False),
    ("FLARE", 78.8, 0.5, False, False),
    ("Fast3R", 72.7, 0.2, False, False),
    ("VGGT (feed-forward)", 85.3, 0.2, True, False),
    ("VGGT + BA", 93.5, 1.8, True, True),
]

fig, ax = plt.subplots(figsize=(8, 5.5))
for name, auc, t, ours, opt in data:
    color = "#F44336" if ours else ("#607D8B" if opt else "#2196F3")
    marker = "*" if ours else "o"
    size = 260 if ours else 60
    ax.scatter(t, auc, s=size, c=color, marker=marker, zorder=3,
               edgecolors="white", linewidths=1.0)
    ax.annotate(name, (t, auc), textcoords="offset points", xytext=(6, 4),
                fontsize=8.5, color="#333333")

ax.set_xscale("log")
ax.set_xlabel("Runtime per reconstruction (seconds, log scale)")
ax.set_ylabel("Camera pose AUC@30 on RealEstate10K (higher is better)")
ax.set_title("VGGT: feed-forward geometry vs optimization-based pipelines", fontsize=11.5)
ax.grid(linestyle="--", alpha=0.4, which="both")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# 图例
ax.scatter([], [], s=200, c="#F44336", marker="*", label="VGGT (ours)")
ax.scatter([], [], s=60, c="#607D8B", label="optimization-based")
ax.scatter([], [], s=60, c="#2196F3", label="feed-forward")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(40, 100)
fig.tight_layout()
fig.savefig("public/vggt_pose_accuracy_vs_speed.png", dpi=150)
print("saved public/vggt_pose_accuracy_vs_speed.png")
