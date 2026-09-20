#!/usr/bin/env python3
"""
生成"极分解 F = R S"的可视化：同一个形变梯度 F（拉伸+旋转混在一起），
拆解成先纯拉伸 S 再纯旋转 R 两步，用三个网格面板依次展示：
原始方格 -> 只做 S（纯变形，无旋转）-> 再做 R（旋转到最终朝向，形状不再变化）
直观说明"F 里的旋转成分不该被算进应变能量"这件事。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import polar

def make_grid(n=7):
    lin = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(lin, lin)
    return xx, yy

def apply(F, xx, yy):
    pts = np.stack([xx.ravel(), yy.ravel()], axis=0)
    out = F @ pts
    return out[0].reshape(xx.shape), out[1].reshape(xx.shape)

def draw_grid(ax, xx, yy, color='#2196F3', lw=1.6, ref=None):
    if ref is not None:
        rx, ry = ref
        n = rx.shape[0]
        for i in range(n):
            ax.plot(rx[i, :], ry[i, :], color='#CFCFCF', linewidth=0.8)
            ax.plot(rx[:, i], ry[:, i], color='#CFCFCF', linewidth=0.8)
    n = xx.shape[0]
    for i in range(n):
        ax.plot(xx[i, :], yy[i, :], color=color, linewidth=lw)
        ax.plot(xx[:, i], yy[:, i], color=color, linewidth=lw)

n = 7
xx, yy = make_grid(n)

theta = np.radians(35)
R_true = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
S_true = np.array([[1.7, 0.3], [0.3, 0.9]])  # 对称拉伸+剪切矩阵
F = R_true @ S_true

# scipy.linalg.polar 默认返回 F = U P（U 正交, P 正定对称），
# 对应我们要的 F = R S
R, S = polar(F)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

dx0, dy0 = xx, yy
draw_grid(axes[0], dx0, dy0, color='#607D8B')
axes[0].set_title('Original grid $X$\n(reference shape)', fontsize=11, fontweight='bold')

dxS, dyS = apply(S, xx, yy)
draw_grid(axes[1], dxS, dyS, color='#4CAF50', ref=(dx0, dy0))
axes[1].set_title('After stretch $S$ only\n(shape changes, no spin)', fontsize=11, fontweight='bold')

dxF, dyF = apply(F, xx, yy)
draw_grid(axes[2], dxF, dyF, color='#F44336', ref=(dxS, dyS))
axes[2].set_title('After rotation $R$ too\n$F=RS$ (final orientation)', fontsize=11, fontweight='bold')

for ax in axes:
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.8, 2.4)
    ax.set_ylim(-0.8, 2.4)
    ax.grid(True, alpha=0.15)
    ax.set_xlabel('$x_1$', fontsize=9)
    ax.set_ylabel('$x_2$', fontsize=9)

fig.suptitle('Polar Decomposition $F=RS$: Separating Rotation from Pure Stretch',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('public/polar_decomposition_rotation_stretch.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/polar_decomposition_rotation_stretch.png")
