#!/usr/bin/env python3
"""
生成 2-DOF 机械臂可达工作空间可视化——展示可达空间和灵巧工作空间的区别。
"""
import numpy as np
import matplotlib.pyplot as plt

L1 = 1.0
L2 = 0.7

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# 采样所有可能的关节角组合，画出末端可达位置
q1_range = np.linspace(-np.pi, np.pi, 360)
q2_range = np.linspace(-np.pi, np.pi, 360)

# 全可达工作空间
all_x = []
all_y = []
for q1 in q1_range:
    for q2 in q2_range[::3]:  # 稀疏采样
        x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
        y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
        all_x.append(x)
        all_y.append(y)

# 画外边界和内边界
theta = np.linspace(0, 2*np.pi, 500)
outer_r = L1 + L2  # 1.7
inner_r = abs(L1 - L2)  # 0.3

# 可达工作空间（环形区域填充）
ax.fill_between(outer_r * np.cos(theta), 
                outer_r * np.sin(theta),
                alpha=0.15, color='#2196F3', label='Reachable Workspace')
# 扣掉内部不可达区域
circle_inner = plt.Circle((0, 0), inner_r, color='white', zorder=3)
ax.add_patch(circle_inner)

# 画边界线
ax.plot(outer_r * np.cos(theta), outer_r * np.sin(theta), 
        '-', color='#2196F3', linewidth=2, alpha=0.8)
ax.plot(inner_r * np.cos(theta), inner_r * np.sin(theta), 
        '-', color='#F44336', linewidth=2, alpha=0.8)

# 标注
ax.annotate('Outer boundary\n$r = L_1 + L_2 = 1.7$', 
            xy=(1.2, 1.2), xytext=(0.5, 1.9),
            fontsize=10, color='#1565C0',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

ax.annotate('Inner boundary\n$r = |L_1 - L_2| = 0.3$', 
            xy=(0.21, 0.21), xytext=(0.7, 0.6),
            fontsize=10, color='#C62828',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

# 画几个示意位姿
# 可达点
ax.plot(1.0, 0.8, '*', color='#4CAF50', markersize=15, zorder=5)
ax.text(1.05, 0.85, 'Reachable\n(multiple solutions)', fontsize=9, color='#2E7D32')

# 不可达点
ax.plot(1.8, 0.5, 'x', color='#F44336', markersize=12, markeredgewidth=3, zorder=5)
ax.text(1.55, 0.2, 'Unreachable\n(no solution)', fontsize=9, color='#C62828')

# 边界点（奇异）
ax.plot(1.7, 0.0, 'D', color='#FF9800', markersize=10, zorder=5)
ax.text(1.4, -0.25, 'Boundary\n(singular config)', fontsize=9, color='#E65100')

# 基座
ax.plot(0, 0, 's', color='#333', markersize=12, zorder=5)
ax.text(0.05, -0.15, 'Base', fontsize=10, color='#333')

ax.set_xlim(-2.1, 2.3)
ax.set_ylim(-2.1, 2.3)
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('2-DOF Arm Workspace ($L_1=1.0$, $L_2=0.7$)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.2)
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('public/ik_2dof_workspace.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/ik_2dof_workspace.png")
