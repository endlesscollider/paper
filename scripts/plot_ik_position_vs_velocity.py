#!/usr/bin/env python3
"""
生成位置级 IK vs 速度级 IK 的运动轨迹对比图：
- 位置级 IK 可能产生关节跳变
- 速度级 IK 天然平滑
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

# 时间轴
t = np.linspace(0, 2, 200)
dt = t[1] - t[0]

# 模拟一个关节角的轨迹
# 位置级 IK：每帧独立求解，可能在多解之间跳变
q_position = np.zeros_like(t)
q_position[:50] = 0.5 + 0.01 * t[:50]
# 在 t=0.5 时，位置级 IK 切换到另一个解（跳变）
q_position[50:100] = -0.8 + 0.005 * t[50:100]
# 再跳回来
q_position[100:150] = 0.52 + 0.008 * t[100:150]
q_position[150:] = 0.55 + 0.002 * t[150:]
# 添加小噪声模拟数值抖动
q_position += np.random.normal(0, 0.01, len(t))

# 速度级 IK（Pink）：通过速度积分，天然连续平滑
q_velocity = np.zeros_like(t)
q_velocity[0] = 0.5
target_traj = 0.5 + 0.1 * np.sin(2 * np.pi * t / 2)  # 平滑目标
for i in range(1, len(t)):
    error = target_traj[i] - q_velocity[i-1]
    # 速度限制 + 平滑
    dq = np.clip(error * 5.0, -1.0, 1.0)  # gain=5, vel_limit=1.0
    q_velocity[i] = q_velocity[i-1] + dq * dt

# 子图1：关节角轨迹
ax1.plot(t, q_position, '-', color='#F44336', linewidth=2, alpha=0.9, 
         label='Position-level IK (independent per frame)')
ax1.plot(t, q_velocity, '-', color='#2196F3', linewidth=2.5, 
         label='Velocity-level IK (Pink, integrated)')

# 标注跳变
ax1.annotate('Solution jump!\n(different IK branch)', 
             xy=(0.5, q_position[50]), xytext=(0.7, -0.4),
             fontsize=9, color='#C62828',
             arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_ylabel('Joint angle $q_3$ (rad)', fontsize=11)
ax1.set_title('Joint Trajectory: Position-level vs Velocity-level IK', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 2)

# 子图2：关节速度（一阶差分）
dq_position = np.diff(q_position) / dt
dq_velocity = np.diff(q_velocity) / dt

ax2.plot(t[1:], dq_position, '-', color='#F44336', linewidth=1.5, alpha=0.8, 
         label='Position-level IK (velocity via finite diff)')
ax2.plot(t[1:], dq_velocity, '-', color='#2196F3', linewidth=2, 
         label='Velocity-level IK (Pink, native output)')

# 标注速度尖峰
max_idx = np.argmax(np.abs(dq_position))
ax2.annotate('Velocity spike\n(dangerous for motors!)', 
             xy=(t[max_idx+1], dq_position[max_idx]), 
             xytext=(t[max_idx+1]+0.3, dq_position[max_idx]*0.7),
             fontsize=9, color='#C62828',
             arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

ax2.axhline(y=1.5, color='#4CAF50', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.axhline(y=-1.5, color='#4CAF50', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.text(1.8, 1.6, 'Velocity limit', fontsize=9, color='#4CAF50')

ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Joint velocity $\\dot{q}_3$ (rad/s)', fontsize=11)
ax2.set_title('Joint Velocity: Smooth (Pink) vs Discontinuous (Position IK)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2)
ax2.set_ylim(-15, 15)

plt.tight_layout()
plt.savefig('public/ik_position_vs_velocity_level.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/ik_position_vs_velocity_level.png")
