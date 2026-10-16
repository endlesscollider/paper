#!/usr/bin/env python3
"""
为 OVOW 精读生成"4D 场景基准上精度-速度对比"图。

数据来源：OVOW 论文 Table 2（OVOW-4D-Scene-Bench，120 个动态场景）。
左图：Object-IoU（越高越好）—— 实例分离与几何正确性。
右图：单帧耗时（秒，对数轴，越低越好）—— 计算成本。
两者并排是为了展示"又准又快"，单独看任一张都会漏掉一半结论。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["VIGA", "TabletopGen", "CAST", "MIDI", "SceneGen", "SAM3D", "OVOW"]
object_iou = [0.016, 0.034, 0.080, 0.174, 0.051, 0.050, 0.210]
time_per_frame = [788, 638, 365, 234, 152, 103, 3.35]

# OVOW 用强调色，其余统一灰色，读者一眼看到本文位置
colors = ["#B0BEC5"] * 6 + ["#E53935"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：Object-IoU
ax = axes[0]
bars = ax.bar(methods, object_iou, color=colors, width=0.62)
for bar, val in zip(bars, object_iou):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9.5, color="#333333")
ax.set_title("Object-IoU on OVOW-4D-Scene-Bench (higher is better)", fontsize=11.5)
ax.set_ylabel("Object-IoU (OBB)")
ax.set_ylim(0, max(object_iou) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", rotation=25)

# 右：单帧耗时（对数轴）
ax = axes[1]
bars = ax.bar(methods, time_per_frame, color=colors, width=0.62)
for bar, val in zip(bars, time_per_frame):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.12, f"{val:g}",
            ha="center", va="bottom", fontsize=9.5, color="#333333")
ax.set_yscale("log")
ax.set_title("Per-frame Runtime on OVOW-4D-Scene-Bench (lower is better, log scale)", fontsize=11.5)
ax.set_ylabel("Seconds per frame (log)")
ax.set_ylim(1, 2000)
ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", rotation=25)

fig.tight_layout()
fig.savefig("public/ovow_4d_bench_quality_vs_speed.png", dpi=150)
print("saved to public/ovow_4d_bench_quality_vs_speed.png")
