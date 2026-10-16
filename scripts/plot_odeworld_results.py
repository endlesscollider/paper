#!/usr/bin/env python3
"""
一次性脚本：为 ODEWorld 深度解析系列生成两张结果图。

图 1（odeworld_policy_libero.png）：LIBERO-LONG 基准上的平均成功率（论文 Table 2，
3 个随机种子平均）。

图 2（odeworld_policy_realworld.png）：真机机器人实验结果（论文 Table 3），
X-VLA 基线对比 ODEWorld（顺序子目标）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- 图 1：LIBERO-LONG ----------------
methods = ["GLCBC", "SuSIE", "Seer", "VPP",
           "ODEWorld\n(velocity)", "ODEWorld\n(single subgoal)",
           "ODEWorld\n(sequential)"]
acc = [78.0, 76.3, 78.6, 81.0, 82.3, 82.6, 83.6]
colors = ["#607D8B"] * 4 + ["#2196F3"] * 3

fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
bars = ax.bar(np.arange(len(methods)), acc, 0.6, color=colors)
for b, v in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}",
            ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(np.arange(len(methods)))
ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel("Average Success Rate (%)", fontsize=12)
ax.set_title("LIBERO-LONG Policy Learning", fontsize=12)
ax.set_ylim(70, 86)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out1 = "public/odeworld_policy_libero.png"
fig.savefig(out1)
print(f"saved to {out1}")
plt.close(fig)

# ---------------- 图 2：真机 ----------------
tasks = ["Shrimp", "Mouse", "Pen", "Rearrangement", "Average"]
xvla = [80, 60, 30, 50, 55]
ours = [90, 80, 70, 80, 80]

x = np.arange(len(tasks))
width = 0.38
fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
b1 = ax.bar(x - width / 2, xvla, width, label="X-VLA", color="#607D8B")
b2 = ax.bar(x + width / 2, ours, width,
            label="ODEWorld (sequential subgoals)", color="#4CAF50")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{int(b.get_height())}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=9.5)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("Real-World Robot Experiments", fontsize=12)
ax.set_ylim(0, 100)
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out2 = "public/odeworld_policy_realworld.png"
fig.savefig(out2)
print(f"saved to {out2}")
