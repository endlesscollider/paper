#!/usr/bin/env python3
"""
生成"从末端位姿反推手腕中心"的几何示意图：
展示 p_target、d6 偏移方向、p_wrist 之间的关系。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.set_aspect('equal')
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 3.5)

# 手腕中心
pw = np.array([1.5, 2.0])
# 末端执行器
ee = np.array([3.0, 1.2])
# d6 向量（从手腕中心到末端）
d6_vec = ee - pw
d6_len = np.linalg.norm(d6_vec)
d6_dir = d6_vec / d6_len

# 目标 z 轴方向（末端朝向） = d6_dir
z_axis_len = 1.0

# === 画连杆（手腕→末端） ===
ax.plot([pw[0], ee[0]], [pw[1], ee[1]], '-', color='#2E7D32', linewidth=4, 
        solid_capstyle='round', zorder=3)

# === 手腕中心 ===
ax.plot(pw[0], pw[1], 'o', markersize=16, color='#FF6D00', 
        markeredgecolor='white', markeredgewidth=2.5, zorder=5)
ax.text(pw[0] - 0.15, pw[1] + 0.3, '$\\mathbf{p}_w$\n(wrist center)', 
        fontsize=11, color='#E65100', fontweight='bold', ha='center')

# === 末端执行器 ===
ax.plot(ee[0], ee[1], 's', markersize=14, color='#1B5E20', 
        markeredgecolor='white', markeredgewidth=2, zorder=5)
ax.text(ee[0] + 0.15, ee[1] - 0.25, '$\\mathbf{p}_{\\mathrm{target}}$', 
        fontsize=12, color='#1B5E20', fontweight='bold')

# === 画末端坐标系的 z 轴（延伸方向） ===
z_end = ee + z_axis_len * d6_dir
ax.annotate('', xy=(z_end[0], z_end[1]), xytext=(ee[0], ee[1]),
            arrowprops=dict(arrowstyle='->', color='#F44336', lw=2.5))
ax.text(z_end[0] + 0.1, z_end[1] + 0.1, '$\\hat{\\mathbf{z}}_{\\mathrm{target}}$', 
        fontsize=11, color='#C62828', fontweight='bold')

# === d6 标注 ===
# 画 d6 的距离标注（偏移线）
perp_dir = np.array([-d6_dir[1], d6_dir[0]])  # 垂直于 d6 方向
offset = 0.2
pw_off = pw + offset * perp_dir
ee_off = ee + offset * perp_dir
ax.plot([pw_off[0], ee_off[0]], [pw_off[1], ee_off[1]], '-', color='#4CAF50', linewidth=1.5)
ax.plot([pw[0], pw_off[0]], [pw[1], pw_off[1]], '-', color='#4CAF50', linewidth=1, alpha=0.5)
ax.plot([ee[0], ee_off[0]], [ee[1], ee_off[1]], '-', color='#4CAF50', linewidth=1, alpha=0.5)
mid_off = (pw_off + ee_off) / 2
ax.text(mid_off[0] + 0.15, mid_off[1] + 0.15, '$d_6$', 
        fontsize=13, color='#2E7D32', fontweight='bold')

# === 反推公式标注 ===
# 从 ee 画一条虚线箭头回到 pw，表示"反推"
ax.annotate('', xy=(pw[0] + 0.15, pw[1] - 0.1), xytext=(ee[0] - 0.15, ee[1] + 0.1),
            arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=2, linestyle='--'))
ax.text(2.0, 1.9, '$\\mathbf{p}_w = \\mathbf{p}_{\\mathrm{target}} - d_6 \\cdot R\\hat{z}$', 
        fontsize=11, color='#6A1B9A', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', alpha=0.9))

# === 画前面的机器人连杆（简化，表示前 3 关节到手腕） ===
j_base = np.array([0.0, 0.0])
j_shoulder = np.array([0.0, 0.8])
j_elbow = np.array([0.8, 1.6])
links_prev = [j_base, j_shoulder, j_elbow, pw]
for i in range(len(links_prev) - 1):
    ax.plot([links_prev[i][0], links_prev[i+1][0]],
            [links_prev[i][1], links_prev[i+1][1]],
            '-', color='#1565C0', linewidth=3, solid_capstyle='round', zorder=2, alpha=0.6)
for pos in [j_shoulder, j_elbow]:
    ax.plot(pos[0], pos[1], 'o', markersize=10, color='#1976D2', 
            markeredgecolor='white', markeredgewidth=1.5, zorder=4, alpha=0.7)

# 基座
rect = mpatches.FancyBboxPatch((j_base[0] - 0.15, j_base[1] - 0.12), 0.3, 0.12,
                                boxstyle="round,pad=0.02", 
                                facecolor='#455A64', edgecolor='#263238', linewidth=1.5)
ax.add_patch(rect)

# === 说明文字 ===
ax.text(0.02, 0.98, 
        'Key insight:\n'
        '$\\mathbf{p}_w$ only depends on $q_1, q_2, q_3$\n'
        '(front 3 joints control position)',
        transform=ax.transAxes, fontsize=9, color='#1565C0',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.9))

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Wrist Center Decoupling: Back-computing $\\mathbf{p}_w$ from Target', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.1)

plt.tight_layout()
plt.savefig('public/ik_wrist_center_decoupling.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/ik_wrist_center_decoupling.png")
