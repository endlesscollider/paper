#!/usr/bin/env python3
"""具身高斯综述配图 1：视觉力机制的一维演示。

把 3DGS 的渲染简化成可加叠加的一维版本，用真实的数值例子说明：
  - 粒子位置偏了 -> 渲染图和真实图不匹配 -> 光度损失升高
  - 损失对位置的负梯度就是一个"力"，指向让图像重新对齐的方向
  - 沿这个力做梯度下降，粒子会自己走回正确位置

关键设计：三个高斯的间距取到 12 倍标准差，避免"整体平移恰好让一个高斯
顶替相邻高斯"造成的伪局部极小，保证损失曲线严格单峰。曲线单峰性由脚本
内的断言自查，不满足就直接报错。

输出：public/embodied_gs_visual_force.png
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

SIGMA = 0.032
SPACING = 0.38                              # = 11.9 sigma
P_TRUE = np.array([0.30, 0.30 + SPACING, 0.30 + 2 * SPACING])
COLORS = np.array([1.00, 0.72, 0.88])
GRID = np.linspace(-0.05, 1.45, 1600)
DELTA_MAX = 0.10
N_STEP = 50


def render(pos):
    """可加叠加渲染：把每个高斯投到一维像平面上累加。"""
    img = np.zeros_like(GRID)
    for p, c in zip(pos, COLORS):
        img += c * np.exp(-((GRID - p) ** 2) / (2.0 * SIGMA ** 2))
    return img


IMG_GT = render(P_TRUE)


def loss(delta):
    """所有粒子整体平移 delta 后，渲染图与真实图的均方误差。"""
    return float(np.mean((render(P_TRUE + delta) - IMG_GT) ** 2))


deltas = np.linspace(-DELTA_MAX, DELTA_MAX, 561)
L = np.array([loss(d) for d in deltas])
grad = np.gradient(L, deltas)               # dL/d(delta)


def img_deriv(pos):
    """渲染图对位置的导数，用于解析曲率。"""
    d = np.zeros_like(GRID)
    for p, c in zip(pos, COLORS):
        u = GRID - p
        d += c * (-u / SIGMA ** 2) * np.exp(-u ** 2 / (2.0 * SIGMA ** 2))
    return d


# L(d) ~ 0.5 * H * d^2，H = 2 * mean(s'(x)^2)
H = 2.0 * float(np.mean(img_deriv(P_TRUE) ** 2))
ETA = 0.35 / H

# ---- 单峰性自检 ----
i0 = int(np.argmin(np.abs(deltas)))
assert np.all(np.diff(L[:i0 + 1]) <= 1e-12), "损失曲线在负半轴非单调"
assert np.all(np.diff(L[i0:]) >= -1e-12), "损失曲线在正半轴非单调"


def descend(d0):
    """沿视觉力（负梯度）迭代，记录轨迹。"""
    traj, d = [d0], d0
    for _ in range(N_STEP):
        d = d + ETA * (-float(np.interp(d, deltas, grad)))
        traj.append(d)
    return np.array(traj)


t1, t2 = descend(0.140), descend(-0.130)

print(f"[诊断] 间距/sigma = {SPACING / SIGMA:.1f}")
print(f"[诊断] 解析曲率 H = {H:.1f}, 步长 eta = {ETA:.6f}")
print(f"[诊断] 单峰性自检通过（{len(deltas)} 个采样点）")
print(f"[诊断] 起点 +0.140 迭代 {N_STEP} 次后 -> {t1[-1]:+.6f}")
print(f"[诊断] 起点 -0.130 迭代 {N_STEP} 次后 -> {t2[-1]:+.6f}")
print(f"[诊断] L(0) = {loss(0.0):.3e}, L(0.14) = {loss(0.14):.3e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

# ---- (a) 渲染图随整体平移的变化 ----
ax = axes[0]
ax.plot(GRID, IMG_GT, color="#212121", lw=2.4, ls="--", label="ground truth image")
ax.plot(GRID, render(P_TRUE + 0.06), color="#FF9800", lw=1.9, label=r"rendered, $\delta=+0.06$")
ax.plot(GRID, render(P_TRUE + 0.14), color="#F44336", lw=1.9, label=r"rendered, $\delta=+0.14$")
ax.set_xlabel("image coordinate (1-D)")
ax.set_ylabel("intensity")
ax.set_title("(a) render drifts with the particles")
ax.legend(loc="upper right", fontsize=8)

# ---- (b) 光度损失曲线 + 梯度下降轨迹 ----
ax = axes[1]
ax.plot(deltas, L, color="#2196F3", lw=2.2, zorder=2)
ax.axvline(0.0, color="#607D8B", lw=1.2, ls="--", zorder=1)
for traj, col in ((t1, "#E91E63"), (t2, "#9C27B0")):
    ax.plot(traj, np.interp(traj, deltas, L), "o-", color=col, ms=4.0,
            lw=1.3, alpha=0.92, zorder=3)
ax.set_xlabel(r"particle offset $\delta$")
ax.set_ylabel(r"photometric loss $\mathcal{L}(\delta)$")
ax.set_title("(b) loss is minimised where the image matches")
ax.text(0.03, 0.90, r"markers = $40$ gradient-descent steps",
        transform=ax.transAxes, fontsize=8, color="#455A64")

# ---- (c) 视觉力 = 负梯度 ----
ax = axes[2]
force = -grad
ax.plot(deltas, force, color="#009688", lw=2.3)
ax.axhline(0.0, color="#607D8B", lw=1.2, ls="--")
ax.axvline(0.0, color="#607D8B", lw=1.2, ls="--")
for d0, col in ((0.140, "#E91E63"), (-0.130, "#9C27B0")):
    f0 = float(np.interp(d0, deltas, force))
    ax.plot([d0], [f0], "o", color=col, ms=8, zorder=4)
    ax.annotate("", xy=(d0 + 0.055 * np.sign(f0), f0), xytext=(d0, f0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.9))
ax.set_xlabel(r"particle offset $\delta$")
ax.set_ylabel(r"visual force $-\partial\mathcal{L}/\partial\delta$")
ax.set_title("(c) the force always points back to the truth")
ax.text(0.05, 0.06, "zero force only at $\\delta=0$", transform=ax.transAxes,
        fontsize=8, color="#455A64")

fig.tight_layout()
fig.savefig("public/embodied_gs_visual_force.png", dpi=150, bbox_inches="tight")
print("saved public/embodied_gs_visual_force.png")
