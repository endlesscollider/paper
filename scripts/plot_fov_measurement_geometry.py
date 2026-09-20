#!/usr/bin/env python3
"""
"用已知宽度的标定板反推相机视场角 FOV" 的几何推导示意图（中文标注）。

画法：
  - 相机在左侧一个点 C（用一个简化的相机图标表示）。
  - 目标物（比如一块已知宽度 W 的标定板/直尺）放在正前方，
    距相机的垂直距离是 d，且刚好左右两端都贴着画面边缘。
  - 从相机中心分别画两条线连到目标物的上下（这里画成上下，
    代表画面的两条边缘），这两条线的夹角就是相机的满幅 FOV。
  - 图中标出半角 theta = FOV/2，以及直角三角形关系
    tan(theta) = (W/2) / d，从而推出 FOV = 2*atan(W / (2d))。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

d = 3.0     # 相机到目标物的距离
W = 2.4     # 目标物宽度（这里画成竖直方向，代表画面的上下边缘）
half_W = W / 2
theta = np.arctan2(half_W, d)
fov_deg = np.rad2deg(2 * theta)

C = np.array([0.0, 0.0])          # 相机位置
target_top = np.array([d, half_W])
target_bot = np.array([d, -half_W])
target_center = np.array([d, 0.0])

fig, ax = plt.subplots(figsize=(7, 5.5))

# 光轴（相机正前方方向）
ax.plot([C[0], target_center[0]], [C[1], target_center[1]], ':', color='#999', linewidth=1.2, zorder=2)

# 相机到目标物上/下边缘的两条视线
ax.plot([C[0], target_top[0]], [C[1], target_top[1]], '-', color='#2196F3', linewidth=2, zorder=3)
ax.plot([C[0], target_bot[0]], [C[1], target_bot[1]], '-', color='#2196F3', linewidth=2, zorder=3)

# 目标物（竖直线段，代表标定板/直尺，恰好顶满画面上下边缘）
ax.plot([target_top[0], target_bot[0]], [target_top[1], target_bot[1]], '-', color='#37474F', linewidth=3.5, zorder=4)
ax.plot(*target_top, 'o', markersize=6, color='#37474F', zorder=5)
ax.plot(*target_bot, 'o', markersize=6, color='#37474F', zorder=5)
ax.text(target_top[0] + 0.12, target_top[1], '目标物上边缘\n（刚好顶到画面边缘）', fontsize=9, color='#263238', va='center')
ax.text(target_bot[0] + 0.12, target_bot[1], '目标物下边缘', fontsize=9, color='#263238', va='center')

# 相机图标（简化三角形+镜头小圆）
ax.plot(*C, 'o', markersize=11, color='#212121', zorder=6)
ax.text(C[0], C[1] - 0.35, '相机', fontsize=10.5, color='#212121', ha='center')

# d 标注（垂直距离）
ax.plot([C[0], d], [-half_W - 0.3, -half_W - 0.3], '-', color='#616161', linewidth=1, zorder=2)
ax.plot([0, 0], [-half_W - 0.35, -half_W - 0.25], '-', color='#616161', linewidth=1)
ax.plot([d, d], [-half_W - 0.35, -half_W - 0.25], '-', color='#616161', linewidth=1)
ax.text(d / 2, -half_W - 0.55, '距离 $d$', fontsize=11, color='#455A64', ha='center')

# W/2 标注（半宽）
ax.plot([d + 0.35, d + 0.35], [0, half_W], '-', color='#616161', linewidth=1)
ax.text(d + 0.45, half_W / 2, '$W/2$', fontsize=11, color='#455A64', va='center')

# theta 标注（半角）
arc_r = 0.9
arc_angles = np.linspace(0, theta, 50)
ax.plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles), color='#FF9800', linewidth=1.8, zorder=5)
ax.text(arc_r * np.cos(theta / 2) + 0.1, arc_r * np.sin(theta / 2) + 0.1,
        r'$\theta$', fontsize=13, color='#E65100', fontweight='bold')

# FOV 全角标注（上下两条视线之间）
arc_angles_full = np.linspace(-theta, theta, 50)
ax.plot(0.55 * np.cos(arc_angles_full), 0.55 * np.sin(arc_angles_full), color='#9C27B0', linewidth=1.5, zorder=5)
ax.text(0.7, 0, 'FOV', fontsize=10, color='#6A1B9A', va='center', fontweight='bold')

# 推导文字框
ax.text(0.03, 0.97,
        "直角三角形：对边=W/2，邻边=d\n"
        r"$\tan\theta = (W/2)/d$" + "\n"
        r"$\theta = \arctan\!\left(\dfrac{W}{2d}\right)$" + "\n"
        r"$\mathrm{FOV} = 2\theta = 2\arctan\!\left(\dfrac{W}{2d}\right)$",
        transform=ax.transAxes, fontsize=10.5, color='#1565C0', va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.95, edgecolor='#90CAF9'))

ax.text(0.97, 0.03, f'示例: W={W}m, d={d}m\n→ FOV≈{fov_deg:.1f}°',
        transform=ax.transAxes, fontsize=9.5, color='#2E7D32', va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', alpha=0.95, edgecolor='#A5D6A7'))

ax.set_xlim(-0.6, d + 1.3)
ax.set_ylim(-half_W - 1.0, half_W + 0.6)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title('实测视场角 FOV：用已知宽度的目标物反推', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('public/fov_measurement_geometry.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/fov_measurement_geometry.png")
