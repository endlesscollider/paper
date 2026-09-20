#!/usr/bin/env python3
"""
生成 Isaac Lab 关节 PD 增益（stiffness/damping）阶跃响应对比图。

背景：Isaac Lab 的隐式驱动器（ImplicitActuator）本质上是一个二阶弹簧-阻尼系统：
    I * theta'' + d * theta' + k * (theta - theta_target) = 0
其中 I 是关节的有效惯量（armature + 连杆惯量），k 是 stiffness，d 是 damping。
调参时最直观的诊断方式就是看"给一个阶跃目标角度，关节怎么响应"——
这张图对比欠阻尼（振铃）、临界阻尼、过阻尼（迟钝）三种典型情况，
以及"real robot"参考曲线，展示"用阶跃响应做系统辨识、反推 kp/kd"的直觉。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

I = 0.02  # 有效惯量 (kg*m^2)，包含 armature

def step_response(k, d, t):
    """二阶系统对单位阶跃目标角度的响应（欠/临界/过阻尼解析解）。"""
    wn = np.sqrt(k / I)          # 无阻尼自然频率
    zeta = d / (2 * np.sqrt(k * I))  # 阻尼比

    theta = np.zeros_like(t)
    if zeta < 1.0:
        # 欠阻尼：振铃
        wd = wn * np.sqrt(1 - zeta**2)
        theta = 1 - np.exp(-zeta * wn * t) * (
            np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t)
        )
    elif np.isclose(zeta, 1.0):
        # 临界阻尼
        theta = 1 - np.exp(-wn * t) * (1 + wn * t)
    else:
        # 过阻尼
        r1 = -wn * (zeta - np.sqrt(zeta**2 - 1))
        r2 = -wn * (zeta + np.sqrt(zeta**2 - 1))
        c1 = r2 / (r2 - r1)
        c2 = -r1 / (r2 - r1)
        theta = 1 - c1 * np.exp(r1 * t) - c2 * np.exp(r2 * t)
    return theta, zeta

t = np.linspace(0, 0.5, 1000)

configs = [
    ("k=2000, d=10（欠阻尼，振铃）", 2000, 10, "#F44336"),
    ("k=2000, d=57（临界阻尼）", 2000, 2 * np.sqrt(2000 * I), "#2196F3"),
    ("k=2000, d=150（过阻尼，迟钝）", 2000, 150, "#4CAF50"),
    ("k=400, d=25（偏软，接近真机手感）", 400, 25, "#FF9800"),
]

fig, ax = plt.subplots(figsize=(7.5, 5))

for label, k, d, color in configs:
    theta, zeta = step_response(k, d, t)
    ax.plot(t, theta, color=color, linewidth=2.2, label=f"{label}, ζ={zeta:.2f}")

ax.axhline(y=1.0, color="#333", linestyle="--", linewidth=1.2, alpha=0.6, label="目标角度")
ax.axhline(y=1.05, color="#999", linestyle=":", linewidth=1.0, alpha=0.5)
ax.axhline(y=0.95, color="#999", linestyle=":", linewidth=1.0, alpha=0.5)
ax.text(0.42, 1.07, "±5% 误差带", fontsize=8, color="#999")

ax.set_xlabel("时间 (s)", fontsize=12)
ax.set_ylabel("关节角度 / 目标角度（归一化）", fontsize=12)
ax.set_title("Isaac Lab 关节 PD 增益：阶跃响应 vs 阻尼比", fontsize=13, fontweight="bold")
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig("public/isaaclab_pd_step_response.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("✅ public/isaaclab_pd_step_response.png")
