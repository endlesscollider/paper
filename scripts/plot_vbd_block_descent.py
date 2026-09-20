#!/usr/bin/env python3
"""
生成 Vertex Block Descent (VBD) 算法的可视化：
- 子图1：VBD 逐顶点更新示意——同一时刻，每个顶点独立地"看着"周围所有邻居
  当前（可能是上一轮迭代或本轮部分更新过）的位置，各自求解一个只有
  自己3个自由度的小型局部优化问题，与Projective Dynamics需要构造
  全局大型线性系统形成对比。
- 子图2：VBD和PD在同一个弹簧网络任务上的收敛速度对比（示意曲线），
  展示两者在并行度和收敛性质上的权衡差异。
"""
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---------- 子图1：VBD逐顶点局部更新示意 ----------
np.random.seed(3)
center = np.array([0.7, 0.7])
neighbors = np.array([
    [0.0, 0.3], [0.3, 1.3], [1.5, 1.2], [1.4, 0.1], [0.0, 1.0]
])

for nb in neighbors:
    ax1.plot([center[0], nb[0]], [center[1], nb[1]], '-', color='#BDBDBD', linewidth=1.3, zorder=1)
    ax1.plot(*nb, 'o', color='#2196F3', markersize=9, zorder=3)

ax1.plot(*center, 'o', color='#F44336', markersize=14, zorder=5, label='Vertex being updated (only 3 unknowns)')
ax1.plot(neighbors[:, 0], neighbors[:, 1], 'o', color='#2196F3', markersize=9, label='Neighbor vertices (fixed this sub-step)')

# 画一个箭头示意"局部最优移动方向"
new_center = center + np.array([-0.05, 0.08])
ax1.annotate('', xy=new_center, xytext=center,
              arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2.2))
ax1.text(new_center[0]-0.05, new_center[1]+0.08, 'locally optimal\nmove', fontsize=8.5, color='#2E7D32')

ax1.set_xlim(-0.3, 1.9)
ax1.set_ylim(-0.1, 1.7)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.legend(fontsize=8, loc='lower center')
ax1.set_title('VBD: Each Vertex Solves Its Own\n3-DOF Local Problem, Neighbors Fixed', fontsize=11, fontweight='bold')

# ---------- 子图2：VBD vs PD收敛/并行度对比示意 ----------
iters = np.arange(1, 31)
pd_energy = 8.0 * np.exp(-iters / 5.5) + 0.12
vbd_energy = 8.0 * np.exp(-iters / 4.0) + 0.10

ax2.plot(iters, pd_energy, '-o', color='#9C27B0', markersize=4, linewidth=1.8, label='Projective Dynamics\n(global linear solve per iter)')
ax2.plot(iters, vbd_energy, '-s', color='#FF9800', markersize=4, linewidth=1.8, label='VBD\n(local per-vertex solve per iter)')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Total system energy', fontsize=11)
ax2.set_title('VBD Converges Comparably Fast,\nWithout Any Global Linear Solve', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('public/vbd_local_update_and_convergence.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/vbd_local_update_and_convergence.png")
