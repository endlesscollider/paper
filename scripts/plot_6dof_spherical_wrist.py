#!/usr/bin/env python3
"""
生成 6-DOF 球形手腕臂的结构示意图：
- 展示前 3 关节（位置关节）和后 3 关节（手腕关节，轴线交于一点）
- 标注手腕中心位置、末端执行器位置、d6 偏移
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect('equal')
ax.set_xlim(-0.3, 3.5)
ax.set_ylim(-0.5, 3.2)

# === 关节位置（侧视图简化示意） ===
# 基座
base = np.array([0.3, 0.0])
# 关节 1（底座旋转，竖直轴）
j1 = np.array([0.3, 0.3])
# 关节 2（肩部）
j2 = np.array([0.3, 1.0])
# 关节 3（肘部）
j3 = np.array([1.3, 2.0])
# 手腕中心（后 3 轴交汇点）
wrist_center = np.array([2.3, 2.2])
# 末端执行器
ee = np.array([2.9, 2.0])

# === 画连杆 ===
# 前 3 关节的连杆（粗线，蓝色系）
links_position = [base, j1, j2, j3, wrist_center]
for i in range(len(links_position) - 1):
    ax.plot([links_position[i][0], links_position[i+1][0]],
            [links_position[i][1], links_position[i+1][1]],
            '-', color='#1565C0', linewidth=4, solid_capstyle='round', zorder=3)

# 手腕到末端的连杆（绿色系）
ax.plot([wrist_center[0], ee[0]], [wrist_center[1], ee[1]],
        '-', color='#2E7D32', linewidth=4, solid_capstyle='round', zorder=3)

# === 画关节 ===
# 前 3 关节（蓝色圆圈）
for i, (pos, label) in enumerate([(j1, '$q_1$'), (j2, '$q_2$'), (j3, '$q_3$')]):
    ax.plot(pos[0], pos[1], 'o', markersize=14, color='#1976D2', 
            markeredgecolor='white', markeredgewidth=2, zorder=5)
    offset = [(-0.25, 0), (-0.25, 0), (0, 0.15)]
    ax.text(pos[0] + offset[i][0], pos[1] + offset[i][1], label, 
            fontsize=12, color='#0D47A1', fontweight='bold', ha='center')

# 后 3 关节（红/橙色，汇聚于手腕中心）
# 画三条短轴线表示"三轴交于一点"
angles = [30, 90, 150]  # 三条轴的角度（示意）
axis_len = 0.35
for angle_deg in angles:
    angle_rad = np.radians(angle_deg)
    dx = axis_len * np.cos(angle_rad)
    dy = axis_len * np.sin(angle_rad)
    ax.plot([wrist_center[0] - dx, wrist_center[0] + dx],
            [wrist_center[1] - dy, wrist_center[1] + dy],
            '-', color='#E65100', linewidth=2, alpha=0.8, zorder=4)

# 手腕中心大圆
ax.plot(wrist_center[0], wrist_center[1], 'o', markersize=18, 
        color='#FF6D00', markeredgecolor='white', markeredgewidth=2.5, zorder=6)
ax.text(wrist_center[0], wrist_center[1] + 0.3, 'Wrist Center\n$\\mathbf{p}_w$', 
        fontsize=11, color='#E65100', fontweight='bold', ha='center')

# 后 3 关节标签
ax.text(wrist_center[0] + 0.45, wrist_center[1] + 0.15, '$q_4, q_5, q_6$\n(axes intersect\nat one point)', 
        fontsize=9, color='#BF360C', style='italic', ha='left')

# 末端执行器
ax.plot(ee[0], ee[1], 's', markersize=12, color='#2E7D32', 
        markeredgecolor='white', markeredgewidth=2, zorder=5)
ax.text(ee[0] + 0.1, ee[1] - 0.15, 'End-Effector\n$\\mathbf{p}_{\\mathrm{target}}$', 
        fontsize=10, color='#1B5E20', fontweight='bold')

# 基座
rect = mpatches.FancyBboxPatch((base[0] - 0.2, base[1] - 0.15), 0.4, 0.15,
                                boxstyle="round,pad=0.02", 
                                facecolor='#455A64', edgecolor='#263238', linewidth=2)
ax.add_patch(rect)
ax.text(base[0], base[1] - 0.3, 'Base', fontsize=10, color='#37474F', ha='center')

# === 标注 d6 偏移 ===
mid_wrist_ee = (wrist_center + ee) / 2
ax.annotate('', xy=(ee[0], ee[1] + 0.15), xytext=(wrist_center[0], wrist_center[1] + 0.15),
            arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=1.5))
ax.text(mid_wrist_ee[0], mid_wrist_ee[1] + 0.3, '$d_6$', 
        fontsize=12, color='#2E7D32', fontweight='bold', ha='center')

# === 分组标注框 ===
# 前 3 关节组
from matplotlib.patches import FancyBboxPatch
box1 = FancyBboxPatch((-0.1, -0.05), 1.7, 2.5, boxstyle="round,pad=0.1",
                      facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5, 
                      alpha=0.3, linestyle='--', zorder=1)
ax.add_patch(box1)
ax.text(0.7, 2.55, 'Position joints\n(determine $\\mathbf{p}_w$)', 
        fontsize=10, color='#1565C0', ha='center', fontweight='bold')

# 后 3 关节组
box2 = FancyBboxPatch((1.85, 1.6), 1.4, 1.0, boxstyle="round,pad=0.1",
                      facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5,
                      alpha=0.3, linestyle='--', zorder=1)
ax.add_patch(box2)
ax.text(2.55, 1.55, 'Orientation joints\n(determine $R$)', 
        fontsize=10, color='#BF360C', ha='center', fontweight='bold')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('z', fontsize=11)
ax.set_title('6-DOF Arm with Spherical Wrist: Structure Overview', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.15)

# 去掉坐标轴刻度（示意图不需要具体数值）
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig('public/ik_6dof_spherical_wrist_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/ik_6dof_spherical_wrist_structure.png")
