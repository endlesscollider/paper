#!/usr/bin/env python3
"""
一次性脚本：为 FPO 精读生成两张柱状图。
数据来源：McAllister et al., "Flow Matching Policy Gradients", arXiv:2507.21053, Table 1.

图 1：Gaussian PPO(tuned) vs DPPO vs FPO 在 MuJoCo Playground 10 个任务上的平均 reward 对比
图 2：FPO 在不同 Monte Carlo 采样数 N_mc 下的平均 reward（验证"多采样降低偏差"的效果）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)

# ---- 子图 1：主对比 ----
ax = axes[0]
labels = ["Gaussian\nPPO (tuned)", "DPPO", "FPO\n(Nmc=8)"]
means = [667.8, 652.5, 759.3]
errs = [66.0, 83.7, 45.3]
colors = ["#607D8B", "#FF9800", "#2196F3"]

bars = ax.bar(labels, means, yerr=errs, capsize=5, color=colors, width=0.55)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + errs[means.index(val)] + 15,
             f"{val:.1f}", ha="center", va="bottom", fontsize=10, color="#333333")

ax.set_title("Gaussian PPO vs DPPO vs FPO", fontsize=12)
ax.set_ylabel("Average Reward (10 tasks)")
ax.set_ylim(0, 900)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---- 子图 2：N_mc 消融 ----
ax2 = axes[1]
labels2 = ["Nmc=1", "Nmc=4", "Nmc=8"]
means2 = [691.6, 731.2, 759.3]
errs2 = [50.3, 58.2, 45.3]
colors2 = ["#9C27B0", "#3F51B5", "#2196F3"]

bars2 = ax2.bar(labels2, means2, yerr=errs2, capsize=5, color=colors2, width=0.55)
for bar, val, e in zip(bars2, means2, errs2):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + e + 15,
              f"{val:.1f}", ha="center", va="bottom", fontsize=10, color="#333333")

ax2.set_title("FPO Reward vs. MC Sample Count", fontsize=12)
ax2.set_ylabel("Average Reward (10 tasks)")
ax2.set_ylim(0, 900)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("public/fpo_mujoco_comparison_bar.png")
print("saved to public/fpo_mujoco_comparison_bar.png")
