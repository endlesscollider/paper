#!/usr/bin/env python3
"""
生成物质点法(MPM)的核心可视化：
- 子图1：粒子(拉格朗日)与背景网格(欧拉)的叠加示意——
  每个粒子携带质量/速度/形变梯度信息，背景网格只是临时的计算脚手架。
- 子图2：P2G -> Grid Update -> G2P 三步循环流程图，配合权重函数示意
  单个粒子如何把自己的质量/动量"撒"到周围16个(2D中B-spline是3x3或4x4)
  网格节点上。
"""
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

# ---------- 子图1：粒子+网格叠加示意 ----------
grid_n = 8
grid_lin = np.linspace(0, 7, grid_n)
for g in grid_lin:
    ax1.axhline(y=g, color='#BDBDBD', linewidth=0.6, zorder=1)
    ax1.axvline(x=g, color='#BDBDBD', linewidth=0.6, zorder=1)
ax1.plot(np.repeat(grid_lin, grid_n), np.tile(grid_lin, grid_n), 's', color='#90A4AE', markersize=4, zorder=2, label='Grid nodes (Eulerian, temporary)')

np.random.seed(7)
n_particles = 60
# 模拟一团粒子聚集在中间区域(比如一块被压缩的雪球)
cx, cy = 3.5, 3.5
particles = cx + np.random.randn(n_particles, 2) * 1.0
ax1.plot(particles[:, 0], particles[:, 1], 'o', color='#F44336', markersize=5, zorder=4, label='Material particles (Lagrangian, carry mass/velocity/F)')

# 标出一个粒子及其周围3x3网格节点(展示权重函数的作用范围)
p_highlight = np.array([3.7, 3.3])
ax1.plot(*p_highlight, 'o', color='#FFEB3B', markersize=10, markeredgecolor='#F57F17', markeredgewidth=2, zorder=5)
nearby = [(gx, gy) for gx in [3.0, 4.0, 5.0] for gy in [3.0, 4.0, 5.0]]
for gx, gy in nearby:
    ax1.plot([p_highlight[0], gx], [p_highlight[1], gy], '-', color='#FF9800', alpha=0.5, linewidth=1.2, zorder=3)

ax1.set_xlim(-0.5, 7.5)
ax1.set_ylim(-0.5, 7.5)
ax1.set_aspect('equal')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_title('MPM: Particles Carry State,\nGrid is a Temporary Scaffold', fontsize=11, fontweight='bold')
ax1.axis('off')

# ---------- 子图2：P2G -> Grid Update -> G2P 循环流程 ----------
ax2.axis('off')
steps = ['Particles\n(mass, velocity, F)', 'P2G:\nscatter to grid\n(weighted transfer)',
          'Grid update:\nsolve momentum eq.\non grid nodes', 'G2P:\ngather back to particles\n(velocity, gradient)']
colors_box = ['#F44336', '#FF9800', '#4CAF50', '#2196F3']
y_positions = [0.85, 0.6, 0.35, 0.1]

for i, (s, c, y) in enumerate(zip(steps, colors_box, y_positions)):
    ax2.add_patch(plt.Rectangle((0.05, y - 0.08), 0.9, 0.16, facecolor=c, alpha=0.15, edgecolor=c, linewidth=2))
    ax2.text(0.5, y, s, fontsize=10, ha='center', va='center', fontweight='bold', color=c)
    if i < len(steps) - 1:
        ax2.annotate('', xy=(0.5, y_positions[i+1] + 0.09), xytext=(0.5, y - 0.09),
                      arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))

ax2.annotate('', xy=(0.98, 0.85), xytext=(0.98, 0.1),
              arrowprops=dict(arrowstyle='->', color='#999', lw=1.5,
                               connectionstyle="arc3,rad=0.3"))
ax2.text(1.15, 0.475, 'next\ntimestep', fontsize=8.5, color='#777', rotation=90, va='center')

ax2.set_xlim(0, 1.3)
ax2.set_ylim(0, 1.0)
ax2.set_title('MPM Cycle: P2G $\\to$ Grid Solve $\\to$ G2P', fontsize=11.5, fontweight='bold')

plt.tight_layout()
plt.savefig('public/mpm_particle_grid_p2g_g2p_cycle.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/mpm_particle_grid_p2g_g2p_cycle.png")
