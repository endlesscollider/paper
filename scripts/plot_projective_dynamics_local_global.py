#!/usr/bin/env python3
"""
生成 Projective Dynamics 局部-全局求解框架的可视化：
- 子图1：一个简单弹簧网络示意图，局部步骤(local step)把每根弹簧独立"投影"
  到最近的合法长度，全局步骤(global step)把所有投影结果揉合成一个
  兼顾惯性和所有约束的折中位置——用箭头示意这个两阶段过程。
- 子图2：迭代次数 vs 系统总能量的收敛曲线，展示局部-全局迭代
  如何逐步让弹簧网络趋于平衡状态。
"""
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---------- 子图1：局部投影 + 全局求解示意 ----------
# 简单的三角形弹簧网络，模拟一次局部-全局迭代
pts_current = np.array([[0.0, 0.0], [1.3, 0.3], [0.6, 1.4]])  # 当前（偏离rest length）的位置
rest_pairs = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]  # (i, j, rest_length)

ax1.set_title('Local Step: project each spring\nto its rest length independently', fontsize=10.5, fontweight='bold')
colors = ['#F44336', '#4CAF50', '#2196F3']
for (i, j, L0), c in zip(rest_pairs, colors):
    p_i, p_j = pts_current[i], pts_current[j]
    mid = (p_i + p_j) / 2
    direction = (p_j - p_i) / np.linalg.norm(p_j - p_i)
    proj_i = mid - direction * L0 / 2
    proj_j = mid + direction * L0 / 2
    ax1.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], '--', color=c, alpha=0.4, linewidth=1.5)
    ax1.plot([proj_i[0], proj_j[0]], [proj_i[1], proj_j[1]], '-', color=c, linewidth=2.5,
              label=f'Spring {i}-{j} projected to $L_0$={L0}')
    ax1.plot(*proj_i, 'o', color=c, markersize=5)
    ax1.plot(*proj_j, 'o', color=c, markersize=5)

ax1.plot(pts_current[:, 0], pts_current[:, 1], 'ks', markersize=8, label='Current vertex positions', zorder=5)
ax1.set_aspect('equal')
ax1.legend(fontsize=7.5, loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_xlim(-0.8, 2.0)
ax1.set_ylim(-0.6, 2.0)
ax1.text(0.1, -0.45, 'Each spring independently "wants"\na different position for shared vertices\n-> Global step reconciles them',
          fontsize=8, color='#555', style='italic')

# ---------- 子图2：局部-全局迭代收敛曲线（模拟示意） ----------
np.random.seed(1)
n_iters = 30
# 模拟一个典型的、快速衰减到平台的能量下降曲线
iters = np.arange(n_iters)
energy = 8.0 * np.exp(-iters / 4.0) + 0.15 + 0.02 * np.random.rand(n_iters)

ax2.plot(iters, energy, '-o', color='#9C27B0', markersize=4, linewidth=1.8)
ax2.axhline(y=0.15, color='#9E9E9E', linestyle='--', linewidth=1.2, label='Converged equilibrium energy')
ax2.set_xlabel('Local-global iteration', fontsize=11)
ax2.set_ylabel('Total system energy', fontsize=11)
ax2.set_title('Energy Decreases Monotonically\nEach Local-Global Iteration', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('public/projective_dynamics_local_global_step.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/projective_dynamics_local_global_step.png")
