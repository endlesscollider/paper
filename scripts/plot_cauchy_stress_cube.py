#!/usr/bin/env python3
"""
生成柯西应力张量的可视化：
- 子图1：一个立方体微元，在三个坐标面上受到的应力分量示意
  （正应力沿法向、剪应力沿切向），用箭头标出 sigma_xx, sigma_xy 等九个分量里
  最具代表性的几个，帮助建立"应力张量的每一列对应一个切面上的力"这个直觉。
- 子图2：牵引力公式 t = sigma @ n 的几何演示——同一个应力状态，
  法向 n 选择不同的切面方向，牵引力大小和方向如何变化（极坐标风格展示）。
"""
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

# ---------- 子图1：立方体微元上的应力分量示意（2D 简化为方块） ----------
ax1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor='#E3F2FD',
                              edgecolor='#1565C0', linewidth=2))

arrow_kwargs = dict(head_width=0.05, head_length=0.06, length_includes_head=True)

# sigma_xx: 右面法向正应力（拉伸，指向外）
ax1.arrow(1.0, 0.5, 0.28, 0, color='#F44336', **arrow_kwargs)
ax1.arrow(0.0, 0.5, -0.28, 0, color='#F44336', **arrow_kwargs)
ax1.text(1.32, 0.5, r'$\sigma_{xx}$', fontsize=12, color='#F44336', va='center')

# sigma_yy: 顶面法向正应力
ax1.arrow(0.5, 1.0, 0, 0.28, color='#4CAF50', **arrow_kwargs)
ax1.arrow(0.5, 0.0, 0, -0.28, color='#4CAF50', **arrow_kwargs)
ax1.text(0.5, 1.36, r'$\sigma_{yy}$', fontsize=12, color='#4CAF50', ha='center')

# sigma_xy: 顶面切向剪应力（沿x方向）
ax1.arrow(0.25, 1.0, 0.2, 0, color='#FF9800', **arrow_kwargs)
ax1.arrow(0.75, 1.0, -0.2, 0, color='#FF9800', **arrow_kwargs)
ax1.text(0.5, 1.08, r'$\sigma_{xy}$ (shear)', fontsize=10, color='#FF9800', ha='center')

# sigma_yx: 右面切向剪应力（沿y方向），对称配对
ax1.arrow(1.0, 0.25, 0, 0.2, color='#FF9800', **arrow_kwargs)
ax1.arrow(1.0, 0.75, 0, -0.2, color='#FF9800', **arrow_kwargs)
ax1.text(1.08, 0.5, r'$\sigma_{yx}$', fontsize=10, color='#FF9800', va='center', rotation=90)

ax1.set_xlim(-0.7, 1.7)
ax1.set_ylim(-0.7, 1.7)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Stress Components on a Cube Element\n(normal $\\sigma_{ii}$ vs shear $\\sigma_{ij}$)',
               fontsize=11.5, fontweight='bold')

# ---------- 子图2：牵引力 t = sigma @ n 随法向角度变化 ----------
sigma = np.array([[3.0, 1.2], [1.2, 1.0]])  # 一个固定的2D应力状态（对称）

angles = np.linspace(0, 2 * np.pi, 200)
normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
tractions = normals @ sigma.T  # t_i = sigma_ij n_j

t_mag = np.linalg.norm(tractions, axis=1)

ax2.plot(np.cos(angles), np.sin(angles), '--', color='#BDBDBD', linewidth=1.2, label='Unit normal $\\mathbf{n}$ (reference circle)')
ax2.plot(tractions[:, 0], tractions[:, 1], color='#2196F3', linewidth=2.2, label='Traction $\\mathbf{t}=\\sigma\\mathbf{n}$')

# 标出几个特定方向的法向和对应牵引力
sample_idx = [0, 25, 50, 75, 100, 125, 150, 175]
for idx in sample_idx:
    n = normals[idx]
    t = tractions[idx]
    ax2.annotate('', xy=(t[0], t[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.3, alpha=0.7))
    ax2.plot([0, n[0]], [0, n[1]], color='#9E9E9E', linewidth=0.8, alpha=0.6)

ax2.set_xlabel('$x$', fontsize=11)
ax2.set_ylabel('$y$', fontsize=11)
ax2.set_title('Traction $\\mathbf{t}=\\sigma\\mathbf{n}$ Depends on Cut Direction $\\mathbf{n}$',
               fontsize=11.5, fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8.5, loc='upper left')
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)

plt.tight_layout()
plt.savefig('public/cauchy_stress_traction_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/cauchy_stress_traction_visualization.png")
