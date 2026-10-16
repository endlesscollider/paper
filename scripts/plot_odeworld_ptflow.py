#!/usr/bin/env python3
"""
一次性脚本：为 ODEWorld 深度解析系列生成两张示意图。

图 1（odeworld_velocity_profile.png）：物理时间上的瞬时速度幅值。数据取自论文
Fig. 3 描述的"先靠近把手、再拉出抽屉"片段——速度先升后降（到达把手时接近
零），随后进入拉动阶段再次上升，最后随目标达成衰减到零。用于说明 PT-Flow
学到的速度场不仅编码方向，也编码每个阶段的"正确瞬时速度"。

图 2（odeworld_savgol_smoothing.png）：Savitzky-Golay 导数滤波示意。真实位置
序列带有高频抖动，直接用有限差分求导会把抖动放大成巨大噪声；SG 滤波器在滑动
窗口内做局部多项式最小二乘拟合，再对拟合多项式求导，得到光滑且保形的速度目标。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ---------------- 图 1：速度幅值随归一化物理时间变化 ----------------
tau = np.linspace(0, 1, 600)
approach = 1.15 * np.exp(-((tau - 0.22) ** 2) / 0.0035)
pull = 0.85 * np.exp(-((tau - 0.68) ** 2) / 0.010)
v = approach + pull
v = v * (1 - 1 / (1 + np.exp(-40 * (0.92 - tau))))  # 接近终点时衰减到零

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
ax.plot(tau, v, color="#2196F3", linewidth=2.4, label="|v_theta(z_tau, tau)|")
ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.annotate("approaching handle:\nspeed drops near zero", xy=(0.40, 0.18),
            xytext=(0.44, 0.72), fontsize=10, color="#333",
            arrowprops=dict(arrowstyle="->", color="#999"))
ax.annotate("pulling phase:\nspeed rises again", xy=(0.68, 0.85),
            xytext=(0.70, 1.05), fontsize=10, color="#333",
            arrowprops=dict(arrowstyle="->", color="#999"))
ax.annotate("goal reached:\nvelocity decays to zero", xy=(0.93, 0.10),
            xytext=(0.66, 0.35), fontsize=10, color="#333",
            arrowprops=dict(arrowstyle="->", color="#999"))
ax.set_xlabel("Normalized physical time  tau = t / L", fontsize=12)
ax.set_ylabel("Latent velocity magnitude  |v_theta|", fontsize=12)
ax.set_title("PT-Flow: Time-Variant Velocity Field", fontsize=12)
ax.set_ylim(0, 1.35)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out1 = "public/odeworld_velocity_profile.png"
fig.savefig(out1)
print(f"saved to {out1}")
plt.close(fig)

# ---------------- 图 2：Savitzky-Golay 导数滤波 ----------------
rng = np.random.default_rng(0)
t = np.linspace(0, 1, 120)
clean = 0.9 / (1 + np.exp(-8 * (t - 0.5)))          # 期望的平滑位置曲线
noisy = clean + rng.normal(0, 0.012, size=t.size)     # 加高频抖动

dt = t[1] - t[0]
v_fd = np.gradient(noisy, dt)                        # 朴素有限差分
v_sg = savgol_filter(noisy, window_length=5, polyorder=2, deriv=1) / dt
v_true = np.gradient(clean, dt)

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
ax.plot(t, v_fd, color="#F44336", linewidth=1.4, alpha=0.8,
        label="Finite difference (st+1 - st)/dt (noise amplified)")
ax.plot(t, v_sg, color="#2196F3", linewidth=2.4,
        label="Savitzky-Golay derivative (smooth, shape-preserving)")
ax.plot(t, v_true, color="#4CAF50", linewidth=1.6, linestyle="--",
        label="True velocity")
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Target velocity  s_dot_t", fontsize=12)
ax.set_title("Savitzky-Golay Derivative Filter (window=5, order=2)", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out2 = "public/odeworld_savgol_smoothing.png"
fig.savefig(out2)
print(f"saved to {out2}")
