"""
PD 控制文章配图：二阶闭环系统的阶跃响应。

左图：固定自然频率 omega_n，改变阻尼比 zeta（对应固定 Kp、改变 Kd）
右图：固定临界阻尼 zeta=1，改变自然频率 omega_n（对应改变 Kp，Kd 同步放大）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 统一风格
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

t = np.linspace(0, 1.2, 2000)


def second_order_step(t, wn, zeta):
    """标准二阶系统单位阶跃响应（目标值为 1）。"""
    if zeta < 1.0:
        wd = wn * np.sqrt(1.0 - zeta**2)
        return 1.0 - np.exp(-zeta * wn * t) * (
            np.cos(wd * t) + (zeta / np.sqrt(1.0 - zeta**2)) * np.sin(wd * t)
        )
    if abs(zeta - 1.0) < 1e-9:
        return 1.0 - np.exp(-wn * t) * (1.0 + wn * t)
    # 过阻尼：两个实极点
    sq = np.sqrt(zeta**2 - 1.0)
    s1 = -wn * (zeta - sq)
    s2 = -wn * (zeta + sq)
    return 1.0 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# ---------- 左图：固定 wn=10，改变 zeta ----------
ax = axes[0]
wn = 10.0
cases = [
    (0.0, "zeta = 0  (no damping)", "#F44336", "-"),
    (0.2, "zeta = 0.2  (underdamped)", "#FF9800", "-"),
    (0.5, "zeta = 0.5  (underdamped)", "#2196F3", "-"),
    (1.0, "zeta = 1.0  (critical)", "#4CAF50", "-"),
    (2.0, "zeta = 2.0  (overdamped)", "#607D8B", "-"),
]
for zeta, label, color, ls in cases:
    ax.plot(t, second_order_step(t, wn, zeta), label=label, color=color, linewidth=2.0, linestyle=ls)

ax.axhline(1.0, color="#9E9E9E", linewidth=1.2, linestyle="--")
ax.set_title(r"Fixed stiffness $K_p$ ($\omega_n=10$ rad/s), varying damping", fontsize=12)
ax.set_xlabel("time (s)")
ax.set_ylabel("joint angle  q(t) / target")
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 2.0)
ax.legend(loc="lower right", fontsize=9)
ax.annotate(
    "no damping: rings forever",
    xy=(0.32, 1.95), xytext=(0.45, 1.78),
    fontsize=9, color="#F44336",
    arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.2),
)
ax.annotate(
    "overdamped: sluggish",
    xy=(0.55, 0.62), xytext=(0.62, 0.30),
    fontsize=9, color="#607D8B",
    arrowprops=dict(arrowstyle="->", color="#607D8B", lw=1.2),
)

# ---------- 右图：固定 zeta=1，改变 wn ----------
ax = axes[1]
zeta = 1.0
for wn_val, label, color in [
    (4.0, r"$\omega_n = 4$ rad/s  (soft)", "#FF9800"),
    (10.0, r"$\omega_n = 10$ rad/s", "#2196F3"),
    (25.0, r"$\omega_n = 25$ rad/s  (stiff)", "#4CAF50"),
]:
    ax.plot(t, second_order_step(t, wn_val, zeta), label=label, color=color, linewidth=2.0)

ax.axhline(1.0, color="#9E9E9E", linewidth=1.2, linestyle="--")
ax.set_title(r"Critical damping ($\zeta=1$), varying stiffness", fontsize=12)
ax.set_xlabel("time (s)")
ax.set_ylabel("joint angle  q(t) / target")
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1.4)
ax.legend(loc="lower right", fontsize=9)

fig.suptitle("Closed-loop step response of a PD-controlled joint", fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig("public/pd_control_step_response.png", bbox_inches="tight")
print("saved public/pd_control_step_response.png")
