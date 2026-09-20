#!/usr/bin/env python3
"""
生成形变梯度 F 作用在一个单位方格网格上的四种典型变形效果对比图：
- 恒等变换（不形变）
- 单轴拉伸
- 剪切
- 拉伸叠加旋转（说明 F 里同时混着"形状变化"和"朝向变化"两种信息）

用网格线在变形前后的形状,给"形变梯度到底在做什么"一个最直观的视觉解释。
"""
import numpy as np
import matplotlib.pyplot as plt

def make_grid(n=6):
    """在 [0,1]x[0,1] 上生成一个 n x n 的网格点阵，用于展示变形前后的网格线。"""
    lin = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(lin, lin)
    return xx, yy

def apply_F(F, xx, yy):
    """把 2x2 形变梯度矩阵 F 作用在网格每个点上（以左下角为原点）。"""
    pts = np.stack([xx.ravel(), yy.ravel()], axis=0)  # (2, N)
    deformed = F @ pts
    return deformed[0].reshape(xx.shape), deformed[1].reshape(xx.shape)

def draw_grid(ax, xx, yy, color='#2196F3', lw=1.2):
    n = xx.shape[0]
    for i in range(n):
        ax.plot(xx[i, :], yy[i, :], color=color, linewidth=lw)
        ax.plot(xx[:, i], yy[:, i], color=color, linewidth=lw)

n = 7
xx, yy = make_grid(n)

cases = [
    ("Identity (no deformation)", np.eye(2)),
    ("Uniaxial stretch\n$F=\\mathrm{diag}(1.8, 1.0)$", np.array([[1.8, 0.0], [0.0, 1.0]])),
    ("Shear\n$F_{12}=0.7$ (off-diagonal)", np.array([[1.0, 0.7], [0.0, 1.0]])),
    ("Stretch + Rotation\n$F=R(30^\\circ)\\,\\mathrm{diag}(1.6,0.9)$",
     (lambda th=np.radians(30): np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
      @ np.array([[1.6, 0.0], [0.0, 0.9]]))()),
]

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
axes = axes.flatten()

for ax, (title, F) in zip(axes, cases):
    dx, dy = apply_F(F, xx, yy)
    # 原始网格用淡灰色虚线做参照
    draw_grid(ax, xx, yy, color='#BDBDBD', lw=0.8)
    draw_grid(ax, dx, dy, color='#2196F3', lw=1.6)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.6, 2.2)
    ax.set_ylim(-0.6, 2.2)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('$x_1$', fontsize=9)
    ax.set_ylabel('$x_2$', fontsize=9)

fig.suptitle('Deformation Gradient F Acting on a Unit Grid', fontsize=13, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('public/deformation_gradient_grid_examples.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/deformation_gradient_grid_examples.png")
