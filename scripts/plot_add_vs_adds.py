#!/usr/bin/env python3
"""
为「6DoF 位姿估计与 ADD-S 评测指标」前置知识生成配图。

用一个正方形刚体（模拟"沿某一轴近似对称的物体"，如方盒、瓶子）说明：
ADD 直接比较"同一编号点"旋转前后的距离，因此对旋转误差敏感；
ADD-S 只比较"最近点"的距离，因此对绕对称轴的旋转几乎不惩罚。
这正是对称物体位姿评测必须用 ADD-S 而不是 ADD 的原因。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 在边长 1 的正方形边界上均匀采样 800 个点（形状可视为沿各个轴近似对称）
n_side = 200
s = np.linspace(0, 1, n_side, endpoint=False)
square = np.concatenate([
    np.stack([s, np.zeros_like(s)], axis=1),
    np.stack([np.ones_like(s), s], axis=1),
    np.stack([1 - s, np.ones_like(s)], axis=1),
    np.stack([np.zeros_like(s), 1 - s], axis=1),
])  # (800, 2)


def rot(theta):
    c, sn = np.cos(theta), np.sin(theta)
    return np.array([[c, -sn], [sn, c]])


def add(p, q):
    return np.mean(np.linalg.norm(p - q, axis=1))


def adds(p, q):
    # 每个 p 点到 q 中最近点的距离，取平均
    d2 = ((p[:, None, :] - q[None, :, :]) ** 2).sum(-1)
    return np.mean(np.sqrt(d2.min(axis=1)))


angles = np.linspace(0, np.pi / 2, 60)
add_vals, adds_vals = [], []
for a in angles:
    q = square @ rot(a).T
    add_vals.append(add(square, q))
    adds_vals.append(adds(square, q))
add_vals = np.array(add_vals)
adds_vals = np.array(adds_vals)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(np.degrees(angles), add_vals, color="#F44336", lw=2.2, label="ADD (same point index)")
ax.plot(np.degrees(angles), adds_vals, color="#2196F3", lw=2.2, label="ADD-S (nearest point)")
ax.set_xlabel("Yaw rotation error (degrees)")
ax.set_ylabel("Mean point distance (normalized)")
ax.set_title("ADD vs ADD-S on a symmetric (square) object", fontsize=11.5)
ax.axhline(0.1, color="#607D8B", ls="--", lw=1.1, label="0.1d success threshold")
ax.set_ylim(0, 1.1)
ax.grid(linestyle="--", alpha=0.4)
ax.legend(fontsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("public/add_vs_adds_symmetry.png", dpi=150)
print("saved public/add_vs_adds_symmetry.png")
