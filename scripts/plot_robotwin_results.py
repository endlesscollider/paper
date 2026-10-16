#!/usr/bin/env python3
"""
一次性脚本：为 RoboTwin 2.0 深度解析系列生成两张结果对比图。

图 1（robotwin_codegen_asr.png）：专家代码生成成功率。数据取自论文 Table 1，
对比 RoboTwin 1.0 / 2.0 在 Vanilla、FB（执行日志反馈）、MM FB（多模态反馈）
三种设置下的 ASR 与 Top5-ASR。

图 2（robotwin_realworld_gains.png）：真机 sim-to-real 结果。数据取自论文 Table 4，
对比"10 条真机演示""10 条真机 + 1k 域随机化合成数据""仅合成数据零样本"三种
设置在四个评测配置下的平均成功率。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- 图 1：代码生成成功率 ----------------
groups = ["R1.0\nVanilla", "R1.0\n+FB", "R1.0\n+MM FB",
          "R2.0\nVanilla", "R2.0\n+FB", "R2.0\n+MM FB"]
asr = [47.4, 60.4, 63.9, 62.1, 66.7, 71.3]
top5 = [57.6, 71.4, 74.2, 68.0, 73.6, 78.6]

x = np.arange(len(groups))
width = 0.38
fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
b1 = ax.bar(x - width / 2, asr, width, label="ASR", color="#2196F3")
b2 = ax.bar(x + width / 2, top5, width, label="Top5-ASR", color="#FF9800")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=9)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("Expert Code Generation: ASR and Top5-ASR", fontsize=12)
ax.set_ylim(0, 90)
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out1 = "public/robotwin_codegen_asr.png"
fig.savefig(out1)
print(f"saved to {out1}")
plt.close(fig)

# ---------------- 图 2：真机 sim-to-real ----------------
configs = ["Seen\nClean", "Seen\nCluttered", "Unseen\nClean", "Unseen\nCluttered"]
real10 = [29.5, 14.0, 15.5, 9.0]
real10_syn = [43.0, 41.5, 39.0, 42.0]
zero_shot = [np.nan, np.nan, 36.5, 29.5]

x = np.arange(len(configs))
width = 0.26
fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150)
b1 = ax.bar(x - width, real10, width, label="10 clean real", color="#607D8B")
b2 = ax.bar(x, real10_syn, width, label="10 real + 1k RoboTwin 2.0", color="#4CAF50")
b3 = ax.bar(x + width, zero_shot, width, label="1k RoboTwin 2.0 (zero-shot)", color="#9C27B0")
for bars in (b1, b2, b3):
    for b in bars:
        h = b.get_height()
        if not np.isnan(h):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("Average Success Rate (%)", fontsize=12)
ax.set_title("Real-World Sim-to-Real Transfer (RDT backbone)", fontsize=12)
ax.set_ylim(0, 52)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out2 = "public/robotwin_realworld_gains.png"
fig.savefig(out2)
print(f"saved to {out2}")
