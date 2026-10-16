#!/usr/bin/env python3
"""空载响应辨识验证：末端误差分解与重力下垂两联图。

数据来自 2871 空载标定（835327-835331）的 servo 回放 trace，
数值为仿真 link6 世界坐标与真机记录 end_pose 三维欧氏 RMSE（单位 mm）。
恒定偏置 = 逐帧误差向量的均值模长；动态分量 = 误差围绕其均值的 RMSE。
两者满足 RMSE^2 = 偏置^2 + 动态^2。
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (标签, 恒定偏置 [mm], 动态分量 [mm], 总 RMSE [mm])
DECOMP = [
    ("835327 tail\nleft arm", 4.143, 3.075, 5.159),
    ("835328 tail\nright arm", 4.784, 3.648, 6.016),
    ("835330 whole\nright arm", 3.661, 4.753, 5.999),
    ("835331 whole\nleft arm", 3.692, 3.594, 5.153),
]

# 静止侧：关重力(response) vs 开重力(servo) [mm]
STATIC = [
    ("835330 static left", 1.700, 6.319),
    ("835331 static right", 1.635, 3.613),
]

C_OFFSET = "#607D8B"
C_DYN = "#2196F3"
C_TOTAL = "#F44336"
C_NOG = "#4CAF50"
C_G = "#FF9800"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = [row[0] for row in DECOMP]
    x = np.arange(len(labels))
    w = 0.26
    offset = np.array([row[1] for row in DECOMP])
    dyn = np.array([row[2] for row in DECOMP])
    total = np.array([row[3] for row in DECOMP])

    b1 = ax1.bar(x - w, offset, w, label="Constant offset", color=C_OFFSET)
    b2 = ax1.bar(x, dyn, w, label="Dynamic residual", color=C_DYN)
    b3 = ax1.bar(x + w, total, w, label="Total RMSE (servo)", color=C_TOTAL)
    for bars in (b1, b2, b3):
        ax1.bar_label(bars, fmt="%.2f", fontsize=8, padding=1)
    ax1.set_ylabel("End-effector position error [mm]")
    ax1.set_title("(a) RMSE = sqrt(offset^2 + dynamic^2)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 7.4)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")

    slabels = [row[0] for row in STATIC]
    sx = np.arange(len(slabels))
    nog = np.array([row[1] for row in STATIC])
    withg = np.array([row[2] for row in STATIC])
    c1 = ax2.bar(sx - w, nog, w, label="Gravity off (response)", color=C_NOG)
    c2 = ax2.bar(sx, withg, w, label="Gravity on (servo)", color=C_G)
    for bars in (c1, c2):
        ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=1)
    ax2.set_ylabel("End-effector position error [mm]")
    ax2.set_title("(b) Motionless arm: gravity droop")
    ax2.set_xticks(sx)
    ax2.set_xticklabels(slabels, fontsize=8)
    ax2.set_ylim(0, 7.4)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle("ALOHA empty-load response validation (2871)", fontsize=11)
    fig.tight_layout()
    fig.savefig("public/aloha_response_error_decomposition.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
