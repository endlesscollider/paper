#!/usr/bin/env python3
"""
生成 PBD（基于位置的动力学）约束求解迭代次数对绳索形状精度影响的对比图：
- 子图1：同一根两端固定的绳索，用 PBD 距离约束求解，
  分别跑 1 次、3 次、10 次全局迭代后的形状对比——
  迭代次数越多，形状越接近真实的悬链线（不可伸长约束满足得越彻底）
- 子图2：绳索总长度误差（相对自然长度的偏差百分比）随迭代次数的收敛曲线，
  直观展示"迭代几次就足够收敛"这一实践中的关键工程决策
"""
import numpy as np
import matplotlib.pyplot as plt

n_points = 20
total_length = 1.0
rest_len = total_length / (n_points - 1)
gravity = np.array([0.0, -9.8])
dt = 0.02
n_steps = 150

left_anchor = np.array([0.0, 0.0])
right_anchor = np.array([total_length, 0.0])


def pbd_simulate(n_iters):
    """
    最小 PBD 实现：
    1) 用当前速度做一次无约束的位置预测（predict）
    2) 对每一对相邻质点的距离约束做 n_iters 次 Gauss-Seidel 风格投影
    3) 用预测位置和投影后位置的差反推速度
    """
    pos = np.array([left_anchor + (right_anchor - left_anchor) * i / (n_points - 1)
                     for i in range(n_points)])
    vel = np.zeros_like(pos)
    inv_mass = np.ones(n_points)
    inv_mass[0] = 0.0   # 固定点：质量视为无穷大，逆质量为 0
    inv_mass[-1] = 0.0

    for _ in range(n_steps):
        vel = vel + dt * gravity[None, :]
        vel[inv_mass == 0] = 0.0  # 固定点（逆质量为0）永远不受力影响
        pred = pos + dt * vel

        for _ in range(n_iters):
            for i in range(n_points - 1):
                w1, w2 = inv_mass[i], inv_mass[i + 1]
                if w1 + w2 == 0:
                    continue
                delta = pred[i + 1] - pred[i]
                dist = np.linalg.norm(delta) + 1e-9
                diff = (dist - rest_len) / dist
                correction = delta * diff / (w1 + w2)
                pred[i] += w1 * correction
                pred[i + 1] -= w2 * correction

        vel = (pred - pos) / dt
        pos = pred

    return pos


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

iters_to_plot = [1, 3, 10]
colors = ['#F44336', '#FF9800', '#2196F3']
for n_iters, color in zip(iters_to_plot, colors):
    shape = pbd_simulate(n_iters)
    ax1.plot(shape[:, 0], shape[:, 1], '-o', color=color, markersize=3,
              linewidth=1.8, label=f'{n_iters} iteration(s)')

ax1.plot([left_anchor[0], right_anchor[0]], [left_anchor[1], right_anchor[1]],
          's', color='#333', markersize=8, zorder=5)
ax1.set_xlabel('x (m)', fontsize=11)
ax1.set_ylabel('y (m)', fontsize=11)
ax1.set_title('PBD Rope Shape vs Constraint Iterations', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='lower center')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# 子图2：总长度误差 vs 迭代次数
iter_range = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
length_errors = []
for n_iters in iter_range:
    shape = pbd_simulate(n_iters)
    actual_len = np.sum(np.linalg.norm(np.diff(shape, axis=0), axis=1))
    err_pct = abs(actual_len - total_length) / total_length * 100
    length_errors.append(err_pct)

ax2.plot(iter_range, length_errors, '-o', color='#4CAF50', markersize=6, linewidth=2)
ax2.set_xlabel('Number of constraint solve iterations', fontsize=11)
ax2.set_ylabel('Total length error (%)', fontsize=11)
ax2.set_title('Constraint Satisfaction Converges with Iterations', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=5.0, color='#9E9E9E', linestyle='--', linewidth=1.2, label='5% error threshold')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('public/pbd_constraint_iterations_convergence.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/pbd_constraint_iterations_convergence.png")
