#!/usr/bin/env python3
"""具身高斯综述配图 2：为什么必须用椭球而不是椭球中心。

用一个二维椭圆（各向异性高斯）说明：
  - 从椭圆中心沿不同方向往外走同样距离，到椭圆表面的间隙差别很大
  - 把高斯简化成一个点（点云路线）或一个各向同性的球（平均半径），
    得到的间隙是常值，在大长轴方向上会严重高估安全性

输出：public/embodied_gs_ellipsoid_collision.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

A, B = 1.00, 0.40          # 椭圆半长轴、半短轴
R_PROBE = 1.30             # 探测点距中心的距离（大于 A，保证在椭圆外）
R_ROBOT = 0.10             # 机器人半径


def ellipse_point(phi, a=A, b=B):
    return np.array([a * np.cos(phi), b * np.sin(phi)])


def closest_point_on_ellipse(p):
    """数值求椭圆上距 p 最近的点。"""
    res = minimize_scalar(
        lambda phi: np.sum((p - ellipse_point(phi)) ** 2),
        bounds=(0.0, 2.0 * np.pi), method="bounded",
        options={"xatol": 1e-10},
    )
    return ellipse_point(res.x)


def exact_clearance(theta):
    """给定方向 theta，精确间隙 = 到椭圆表面距离 - 机器人半径。"""
    p = R_PROBE * np.array([np.cos(theta), np.sin(theta)])
    q = closest_point_on_ellipse(p)
    return float(np.linalg.norm(p - q) - R_ROBOT)


thetas = np.linspace(0.0, 2.0 * np.pi, 361)
exact = np.array([exact_clearance(t) for t in thetas])

r_bar = 0.5 * (A + B)                       # 各向同性近似：取平均半径
iso = R_PROBE - r_bar - R_ROBOT             # 常值，与方向无关
mean_only = R_PROBE - R_ROBOT               # 点云近似：高斯缩成一个点

i_major = int(np.argmin(np.abs(thetas - 0.0)))
i_minor = int(np.argmin(np.abs(thetas - np.pi / 2)))
print(f"[诊断] 沿长轴方向间隙 = {exact[i_major]:.3f}")
print(f"[诊断] 沿短轴方向间隙 = {exact[i_minor]:.3f}")
print(f"[诊断] 各向同性近似 = {iso:.3f}, 点云近似 = {mean_only:.3f}")
print(f"[诊断] 各向同性近似在长轴方向高估 = {iso - exact[i_major]:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

# ---- (a) 几何 ----
ax = axes[0]
phi = np.linspace(0, 2 * np.pi, 400)
ax.plot(A * np.cos(phi), B * np.sin(phi), color="#4CAF50", lw=2.4, zorder=3)
ax.fill(A * np.cos(phi), B * np.sin(phi), color="#4CAF50", alpha=0.12, zorder=1)
ax.plot(np.r_[r_bar * np.cos(phi), r_bar], np.r_[r_bar * np.sin(phi), 0.0],
        color="#607D8B", lw=1.4, ls="--", label=f"isotropic guess $\\bar r={r_bar:.2f}$")
ax.plot([0.0], [0.0], "x", color="#F44336", ms=10, mew=2.4, zorder=4)

for theta, col in zip(np.deg2rad([0, 55, 90, 140]),
                      ["#2196F3", "#FF9800", "#9C27B0", "#E91E63"]):
    p = R_PROBE * np.array([np.cos(theta), np.sin(theta)])
    q = closest_point_on_ellipse(p)
    ax.plot([p[0]], [p[1]], "o", color=col, ms=8, zorder=5)
    ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=col, lw=2.0, zorder=4)
    ax.plot([q[0]], [q[1]], "o", color=col, ms=5, zorder=5)
    n = (p - q) / np.linalg.norm(p - q)
    t = np.array([-n[1], n[0]])
    seg = np.stack([q - 0.16 * t, q + 0.16 * t])
    ax.plot(seg[:, 0], seg[:, 1], "--", color=col, lw=1.3, zorder=4)
    ax.text(p[0] * 1.06, p[1] * 1.06, f"{np.linalg.norm(p - q):.2f}",
            color=col, fontsize=9, ha="center")

ax.set_xlim(-1.75, 1.75)
ax.set_ylim(-1.75, 1.75)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("(a) same distance out, very different clearance")
ax.legend(loc="upper right", fontsize=8)

# ---- (b) 间隙随方向的变化 ----
ax = axes[1]
ax.plot(np.rad2deg(thetas), exact, color="#2196F3", lw=2.4, label="exact (ellipsoid)")
ax.axhline(iso, color="#607D8B", lw=1.8, ls="--", label="isotropic mean radius")
ax.axhline(mean_only, color="#F44336", lw=1.8, ls=":", label="points only (means)")
ax.axvline(0, color="#B0BEC5", lw=1.0)
ax.axvline(90, color="#B0BEC5", lw=1.0)
ax.set_xlim(0, 360)
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xlabel(r"approach direction $\theta$ (deg)")
ax.set_ylabel("clearance (distance to surface $-\\,r_{robot}$)")
ax.set_title("(b) both shortcuts are flat, the truth is not")
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig("public/embodied_gs_ellipsoid_collision.png", dpi=150, bbox_inches="tight")
print("saved public/embodied_gs_ellipsoid_collision.png")
