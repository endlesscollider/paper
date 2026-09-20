#!/usr/bin/env python3
"""
针孔相机成像几何推导示意图（中文标注）。

目的：直接回答"r = f*tan(theta) 这个公式是怎么来的"。
画法：
  - 光轴水平，针孔（镜头中心）在原点 O。
  - 场景中的一个点 P，位于光轴上方，与光轴的夹角是 theta（入射角）。
  - 像平面是与光轴垂直、位于针孔前方距离 f 处的一条竖线
    （这里用"虚拟像平面"画法：把像画在镜头前方而不是真实成像的
    镜头后方倒像位置，这是计算机视觉教材里的标准简化画法，
    避免"倒像"混淆读者）。
  - P 点发出的光线沿直线穿过针孔 O，延长后打在像平面上的点就是
    P 的像 P'。
  - 关键的直角三角形：O 到像平面的垂足 O'，O' 到 P' 的距离就是 r，
    O 到 O' 的距离就是 f，角 P'OO' 就是 theta。
  - 这个直角三角形里，tan(theta) = 对边/邻边 = r/f，
    所以 r = f * tan(theta) —— 直接从图上就能读出这个关系，
    不需要死记硬背三角函数定义。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

f = 1.6                  # 针孔到像平面的距离
theta_deg = 35
theta = np.deg2rad(theta_deg)
r = f * np.tan(theta)    # 像平面上的落点高度

# 场景点 P：在光轴上方、距针孔较远处，保证视觉上"P 在像平面之外"
P_dist = 4.2
P = np.array([-P_dist * np.cos(theta), P_dist * np.sin(theta)])
O = np.array([0.0, 0.0])            # 针孔（镜头中心）
O_prime = np.array([f, 0.0])        # 像平面与光轴的交点
P_prime = np.array([f, r])          # P 在像平面上的像

fig, ax = plt.subplots(figsize=(7.5, 5.5))

# ---- 光轴 ----
ax.annotate('', xy=(f + 1.0, 0), xytext=(-P_dist - 0.3, 0),
            arrowprops=dict(arrowstyle='->', color='#999', lw=1.2))
ax.text(f + 1.0, -0.18, '光轴', fontsize=10, color='#777', ha='right', va='top')

# ---- 像平面（竖线）----
plane_half = r + 0.8
ax.plot([f, f], [-0.3, plane_half], color='#37474F', linewidth=3, zorder=4)
ax.text(f, plane_half + 0.1, '像平面\n(成像传感器)', fontsize=10, color='#263238', ha='center', va='bottom')

# ---- 场景点 P 到针孔 O 的光线，延长到像平面 ----
# 实际光线段：P -> O
ax.plot([P[0], O[0]], [P[1], O[1]], '-', color='#2196F3', linewidth=2, zorder=3)
# 延长线：O -> P'（表示光线穿过针孔后继续直线传播，打在像平面上）
ax.plot([O[0], P_prime[0]], [O[1], P_prime[1]], '-', color='#2196F3', linewidth=2, zorder=3)

# ---- 针孔 O ----
ax.plot(*O, 'o', markersize=11, color='#212121', zorder=6)
ax.text(O[0], O[1] - 0.35, '针孔 $O$\n(镜头中心)', fontsize=10, color='#212121', ha='center', va='top')

# ---- 场景点 P ----
ax.plot(*P, '*', markersize=16, color='#4CAF50', zorder=6)
ax.text(P[0] - 0.15, P[1] + 0.25, '场景中的点 $P$', fontsize=10.5, color='#2E7D32', ha='right')

# ---- 像点 P' ----
ax.plot(*P_prime, 'o', markersize=9, color='#F44336', zorder=6)
ax.text(P_prime[0] + 0.12, P_prime[1], "$P$ 的像 $P'$", fontsize=10.5, color='#C62828', va='center')

# ---- 直角三角形：O, O', P' ----
ax.plot([O[0], O_prime[0]], [O[1], O_prime[1]], '--', color='#616161', linewidth=1.3, zorder=2)  # 邻边 f
ax.plot([O_prime[0], P_prime[0]], [O_prime[1], P_prime[1]], '--', color='#616161', linewidth=1.3, zorder=2)  # 对边 r
ax.plot(*O_prime, 's', markersize=6, color='#616161', zorder=5)

# 直角标记（在 O' 处画一个小方块示意 90°）
sq = 0.12
ax.plot([O_prime[0]-sq, O_prime[0]-sq, O_prime[0]], [O_prime[1], O_prime[1]+sq, O_prime[1]+sq],
        color='#616161', linewidth=1.1, zorder=5)

# f 标注（邻边）
ax.text(f / 2, -0.22, '$f$ (焦距)', fontsize=11, color='#455A64', ha='center', va='top')
# r 标注（对边）
ax.text(f + 0.15, r / 2, "$r$", fontsize=13, color='#455A64', va='center', fontweight='bold')

# theta 标注（O 处的夹角，用一个圆弧表示）
arc_radius = 0.9
arc_theta = np.linspace(0, theta, 60)
arc_x = O[0] + arc_radius * np.cos(arc_theta)
arc_y = O[1] + arc_radius * np.sin(arc_theta)
ax.plot(arc_x, arc_y, color='#FF9800', linewidth=1.8, zorder=5)
mid_angle = theta / 2
ax.text(O[0] + (arc_radius + 0.25) * np.cos(mid_angle),
        O[1] + (arc_radius + 0.25) * np.sin(mid_angle),
        r'$\theta$', fontsize=13, color='#E65100', fontweight='bold')

# ---- 公式推导文字框（中文说明和数学符号分开写，避免 mathtext 内嵌中文报错）----
ax.text(0.03, 0.97,
        "直角三角形 O-O'-P' 中：\n"
        "邻边 = f，对边 = r\n"
        r"$\tan\theta = r / f$" + "\n"
        r"$\Rightarrow\ r = f\tan\theta$",
        transform=ax.transAxes, fontsize=11, color='#1565C0', va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.95, edgecolor='#90CAF9'))

ax.set_xlim(-P_dist - 0.6, f + 1.3)
ax.set_ylim(-0.9, plane_half + 0.6)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title('针孔相机成像几何：r = f·tanθ 是怎么来的', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('public/pinhole_geometry_derivation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/pinhole_geometry_derivation.png")
