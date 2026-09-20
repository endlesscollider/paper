#!/usr/bin/env python3
"""
生成质点弹簧绳索仿真的两张图：
- 子图1：一条两端固定的质点弹簧链，在不同弹簧刚度 k 下，重力作用下的
  静止悬垂形状对比——刚度越大，形状越接近不可伸长绳索的悬链线（catenary）
- 子图2：达到目标刚度所需的最大稳定步长（近似 dt_max ~ 1/sqrt(k/m)）——
  直观展示"刚度越高、越接近真实绳索，显式积分允许的步长就越小"这一权衡，
  呼应正文里"为什么高刚度弹簧链难以用显式积分稳定仿真"的讨论
"""
import numpy as np
import matplotlib.pyplot as plt

# ============ 质点弹簧链参数 ============
n_points = 15          # 质点数（含两端固定点）
total_length = 1.0     # 绳索自然总长
m = 0.02                # 每个质点质量 (kg)
g = 9.8
segment_rest_len = total_length / (n_points - 1)

left_anchor = np.array([0.0, 0.0])
right_anchor = np.array([total_length, 0.0])

def simulate_chain(k, damping=0.98, n_steps=6000, dt=0.0005):
    """
    半隐式 Euler 模拟两端固定的质点弹簧链在重力下的松弛过程，
    返回稳态时各质点位置。damping 是简单的速度衰减,帮助更快收敛到静止状态
    （不影响最终静态形状，只加快数值收敛）。
    """
    # 初始位置：沿直线均匀分布
    pos = np.array([left_anchor + (right_anchor - left_anchor) * i / (n_points - 1)
                     for i in range(n_points)])
    vel = np.zeros_like(pos)

    for _ in range(n_steps):
        forces = np.zeros_like(pos)
        forces[:, 1] -= m * g  # 重力

        # 相邻质点之间的弹簧力（胡克定律）
        for i in range(n_points - 1):
            delta = pos[i + 1] - pos[i]
            dist = np.linalg.norm(delta)
            direction = delta / (dist + 1e-9)
            stretch = dist - segment_rest_len
            f = k * stretch * direction
            forces[i] += f
            forces[i + 1] -= f

        acc = forces / m
        vel = (vel + dt * acc) * damping  # 半隐式 Euler + 阻尼加速收敛
        pos = pos + dt * vel

        # 两端固定
        pos[0] = left_anchor
        pos[-1] = right_anchor
        vel[0] = 0
        vel[-1] = 0

    return pos

# ============ 子图1：不同刚度下的悬垂形状 ============
stiffnesses = [20, 100, 800, 8000]
colors = ['#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for k_val, color in zip(stiffnesses, colors):
    shape = simulate_chain(k_val, dt=0.0003 if k_val < 2000 else 0.00005, n_steps=8000)
    ax1.plot(shape[:, 0], shape[:, 1], '-o', color=color, markersize=3,
              linewidth=1.8, label=f'$k$={k_val} N/m')

ax1.plot([left_anchor[0], right_anchor[0]], [left_anchor[1], right_anchor[1]],
          's', color='#333', markersize=8, zorder=5)
ax1.set_xlabel('x (m)', fontsize=11)
ax1.set_ylabel('y (m)', fontsize=11)
ax1.set_title('Mass-Spring Chain Sag vs Stiffness $k$', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='lower center')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')
ax1.text(0.5, -0.32, 'Higher $k$ -> less sag,\ncloser to inextensible rope',
          fontsize=9, color='#555', ha='center', style='italic')

# ============ 子图2：稳定步长上限 vs 刚度 ============
k_range = np.logspace(1, 4.5, 200)
dt_max = 2.0 / np.sqrt(k_range / m)  # 近似：单个弹簧-质点系统显式方法稳定条件 dt < 2/omega

ax2.loglog(k_range, dt_max, '-', color='#F44336', linewidth=2.5)
ax2.set_xlabel('Spring stiffness $k$ (N/m)', fontsize=11)
ax2.set_ylabel('Max stable explicit step size $\\Delta t_{max}$ (s)', fontsize=11)
ax2.set_title('Stiffness Forces Smaller Time Steps', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')

for k_val, color in zip(stiffnesses, colors):
    dt_here = 2.0 / np.sqrt(k_val / m)
    ax2.plot(k_val, dt_here, 'o', color=color, markersize=8, zorder=5)

plt.tight_layout()
plt.savefig('public/mass_spring_rope_stiffness_tradeoff.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/mass_spring_rope_stiffness_tradeoff.png")
