#!/usr/bin/env python3
"""
生成 2-DOF 平面机械臂的 IK 多解可视化图：展示"肘上"和"肘下"两种构型。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 连杆参数
L1 = 1.0
L2 = 1.0

# 目标末端位置
target_x, target_y = 1.2, 0.8

# 计算 IK 解析解
D = (target_x**2 + target_y**2 - L1**2 - L2**2) / (2 * L1 * L2)
# 两个解
q2_up = np.arctan2(np.sqrt(1 - D**2), D)    # 肘上
q2_down = np.arctan2(-np.sqrt(1 - D**2), D)  # 肘下

def solve_q1(q2):
    k1 = L1 + L2 * np.cos(q2)
    k2 = L2 * np.sin(q2)
    return np.arctan2(target_y, target_x) - np.arctan2(k2, k1)

q1_up = solve_q1(q2_up)
q1_down = solve_q1(q2_down)

def fk(q1, q2):
    """正运动学：返回基座、肘部、末端三个点"""
    x0, y0 = 0, 0
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    return [(x0, y0), (x1, y1), (x2, y2)]

# 计算两种构型的关节位置
pts_up = fk(q1_up, q2_up)
pts_down = fk(q1_down, q2_down)

# 画图
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, pts, title, color, q1, q2 in [
    (axes[0], pts_up, 'Solution 1: Elbow-Up', '#2196F3', q1_up, q2_up),
    (axes[1], pts_down, 'Solution 2: Elbow-Down', '#F44336', q1_down, q2_down),
]:
    ax.set_xlim(-0.5, 2.2)
    ax.set_ylim(-1.2, 1.8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 画工作空间边界（圆环）
    theta = np.linspace(0, 2*np.pi, 200)
    # 外边界
    ax.plot((L1+L2)*np.cos(theta), (L1+L2)*np.sin(theta), 
            '--', color='#9E9E9E', alpha=0.4, linewidth=1)
    # 内边界
    ax.plot(abs(L1-L2)*np.cos(theta), abs(L1-L2)*np.sin(theta), 
            '--', color='#9E9E9E', alpha=0.4, linewidth=1)
    
    # 画连杆
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, '-o', color=color, linewidth=3.5, markersize=10, 
            markerfacecolor='white', markeredgewidth=2.5, markeredgecolor=color, zorder=5)
    
    # 基座三角形
    triangle = plt.Polygon([[-0.08, -0.06], [0.08, -0.06], [0, 0.04]], 
                           color='#333', zorder=6)
    ax.add_patch(triangle)
    
    # 目标点
    ax.plot(target_x, target_y, '*', color='#FF9800', markersize=18, 
            markeredgecolor='#E65100', markeredgewidth=1, zorder=7)
    ax.annotate(f'Target ({target_x}, {target_y})', 
                xy=(target_x, target_y), xytext=(target_x+0.1, target_y+0.15),
                fontsize=10, color='#E65100', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))
    
    # 标注关节角
    ax.text(0.15, -0.2, f'$q_1$ = {np.degrees(q1):.1f}°', fontsize=10, color='#333')
    ax.text(pts[1][0]+0.1, pts[1][1]-0.15, f'$q_2$ = {np.degrees(q2):.1f}°', 
            fontsize=10, color='#333')
    
    # 标注连杆长度
    mid1_x = (pts[0][0] + pts[1][0]) / 2
    mid1_y = (pts[0][1] + pts[1][1]) / 2
    ax.text(mid1_x - 0.15, mid1_y + 0.1, '$L_1=1$', fontsize=9, color=color, alpha=0.8)
    
    mid2_x = (pts[1][0] + pts[2][0]) / 2
    mid2_y = (pts[1][1] + pts[2][1]) / 2
    ax.text(mid2_x - 0.15, mid2_y + 0.1, '$L_2=1$', fontsize=9, color=color, alpha=0.8)

plt.tight_layout()
plt.savefig('public/ik_2dof_two_solutions.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/ik_2dof_two_solutions.png")
