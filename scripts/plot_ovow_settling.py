#!/usr/bin/env python3
"""
为 OVOW 精读生成"重力轴吸附函数"图。

数据来源：OVOW 论文 Eq. 15（Stage 4 物理接地）。
d_min = 物体底部接触点到地面的最小有符号距离；
Δz  = 沿重力轴施加的平移量；ε = 接触容差（场景对角线 0.2%）；
δ_max = 下沉上限（场景对角线 2%）。

映射关系（论文定义，ε 取 1、δ_max 取 10 便于看清形状）：
  d_min < -ε        → Δz = -(d_min + ε) = -d_min - ε   （穿模：抬起，落到 -ε 处）
  |d_min| <= ε      → Δz = 0                            （死区：不动，避免抖动）
  d_min > +ε        → Δz = -min(d_min - ε, δ_max)        （悬空：下沉，但封顶）
对照"无限下沉"（无 δ_max 的朴素做法）说明封顶的作用。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

eps = 1.0
delta_max = 10.0


def settle(d_min):
    if d_min < -eps:
        return -(d_min + eps)
    if d_min > eps:
        return -min(d_min - eps, delta_max)
    return 0.0


xs = np.linspace(-14, 26, 2001)
ys = np.array([settle(x) for x in xs])
# 朴素做法：不设下沉上限，悬空多远就往下拉多远
ys_naive = np.where(xs < -eps, -(xs + eps), np.where(xs > eps, -(xs - eps), 0.0))

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

ax.plot(xs, ys_naive, linestyle="--", linewidth=1.8, color="#90A4AE",
        label="no settling cap (naive)")
ax.plot(xs, ys, linewidth=2.4, color="#1E88E5", label="OVOW Eq. 15")

# 死区阴影：|d_min| <= eps 时不平移
ax.axvspan(-eps, eps, color="#FFCDD2", alpha=0.55, zorder=0)
ax.annotate("dead zone $|d_{min}|\\leq\\epsilon$\n(no move, avoids jitter)",
            xy=(0, 0), xytext=(-12.5, -9.5), fontsize=9.5, color="#C62828",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.4))
# 封顶注释
ax.annotate("settling capped at $\\delta_{max}$",
            xy=(24, -delta_max), xytext=(6.5, -16.5), fontsize=9.5, color="#1E88E5",
            arrowprops=dict(arrowstyle="->", color="#1E88E5", lw=1.4))
ax.text(-13.5, 12.5, "penetration\n(push up)", fontsize=9.5, color="#333333")
ax.text(19, 8.5, "floating\n(settle down)", fontsize=9.5, color="#333333")

ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
ax.set_title("Gravity-Axis Settling: Shift vs. Bottom-Contact Distance", fontsize=11.5)
ax.set_xlabel("$d_{min}$ (signed distance of bottom contact set to ground)")
ax.set_ylabel("$\\Delta z$ applied along gravity axis")
ax.legend(loc="lower right", fontsize=9.5)
ax.grid(linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("public/ovow_settling_function.png")
print("saved to public/ovow_settling_function.png")
