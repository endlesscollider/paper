#!/usr/bin/env python3
"""
生成 7-DOF 冗余臂自运动（self-motion）示意图：
固定末端位置，展示多种不同的关节构型（用 2D 简化示意）。
"""
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.set_aspect('equal')

# 固定末端位置
target_x, target_y = 1.5, 0.8

# 模拟多个不同构型到达同一末端点
# 用 3-link 平面臂近似展示自运动的概念
L1, L2, L3 = 0.8, 0.7, 0.5

# 预计算多组解（通过不同的中间关节角实现同一末端）
configs = []
for elbow_param in np.linspace(-0.8, 0.8, 7):
    # 不同的肘部位置参数
    # 肘部在目标点和基座之间的"上方偏移"
    mid_x = 0.6 + elbow_param * 0.3
    mid_y = 0.4 + abs(elbow_param) * 0.8
    
    # 通过中间点反算关节角（简化计算）
    q1 = np.arctan2(mid_y, mid_x)
    d1 = np.sqrt(mid_x**2 + mid_y**2)
    if d1 > L1:
        d1 = L1 * 0.95
        mid_x = d1 * np.cos(q1)
        mid_y = d1 * np.sin(q1)
    
    # 第二段：从肘部到某个过渡点
    wrist_x = target_x - L3 * 0.7  # 简化
    wrist_y = target_y + L3 * 0.3 * elbow_param
    
    alpha_val = min(0.4 + 0.1 * (3.5 - abs(elbow_param * 4)), 0.9)
    configs.append({
        'points': [(0, 0), (mid_x, mid_y), (wrist_x, wrist_y), (target_x, target_y)],
        'alpha': alpha_val
    })

# 画所有构型
cmap = plt.cm.coolwarm
for i, cfg in enumerate(configs):
    color = cmap(i / (len(configs) - 1))
    pts = cfg['points']
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, '-o', color=color, linewidth=2.5, markersize=7,
            alpha=min(cfg['alpha'], 1.0), markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=color, zorder=4)

# 目标点（固定）
ax.plot(target_x, target_y, '*', color='#FF9800', markersize=22, 
        markeredgecolor='#E65100', markeredgewidth=1.5, zorder=10)
ax.text(target_x + 0.08, target_y + 0.08, 'Fixed\nEnd-Effector', 
        fontsize=11, color='#E65100', fontweight='bold')

# 基座
ax.plot(0, 0, 's', color='#333', markersize=14, zorder=10)
ax.text(-0.15, -0.15, 'Base', fontsize=10, color='#333')

# 画肘部轨迹（自运动流形的投影）
elbow_xs = [cfg['points'][1][0] for cfg in configs]
elbow_ys = [cfg['points'][1][1] for cfg in configs]
ax.plot(elbow_xs, elbow_ys, '--', color='#9C27B0', linewidth=1.5, alpha=0.6, zorder=3)
ax.text(np.mean(elbow_xs) - 0.4, np.mean(elbow_ys) + 0.1, 
        'Self-motion\nmanifold\n(elbow trajectory)', 
        fontsize=10, color='#9C27B0', style='italic')

ax.set_xlim(-0.4, 2.2)
ax.set_ylim(-0.4, 1.8)
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Redundant Arm Self-Motion: Same End-Effector, Different Configs', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2)

# 添加说明
ax.text(0.02, 0.02, 
        'All configurations reach the same\nend-effector position — this is\nthe redundancy of 7-DOF arms.',
        transform=ax.transAxes, fontsize=10, color='#555',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.8))

plt.tight_layout()
plt.savefig('public/ik_redundancy_self_motion.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/ik_redundancy_self_motion.png")
