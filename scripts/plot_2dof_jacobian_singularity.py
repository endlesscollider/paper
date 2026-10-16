#!/usr/bin/env python3
"""
二连杆平面臂（L1 = L2 = 1）的雅可比奇异性可视化。

生成两张图：
1. public/robot_2dof_jacobian_singularity.png
   左面板 q2 = 90°（弯曲）：雅可比两列方向不同，速度椭球有面积。
   右面板 q2 = 0°（完全伸直）：两列共线（都垂直于手臂），
   速度椭球退化成一条线段，沿臂方向的瞬时速度恒为零。
2. public/robot_2dof_manipulability_curve.png
   可操作度 w = L1*L2*|sin q2| 随肘角的变化，在 q2 = 0° 与 ±180° 归零。
"""
import numpy as np
import matplotlib.pyplot as plt

L1 = L2 = 1.0
BLUE, GREEN, ORANGE, GRAY, RED = '#2196F3', '#4CAF50', '#FF9800', '#607D8B', '#F44336'


def fk(q1, q2):
    """正运动学：返回基座、肘、末端三个点。"""
    base = np.zeros(2)
    elbow = np.array([L1 * np.cos(q1), L1 * np.sin(q1)])
    ee = elbow + np.array([L2 * np.cos(q1 + q2), L2 * np.sin(q1 + q2)])
    return base, elbow, ee


def jacobian(q1, q2):
    """平面二连杆的 2x2 位置雅可比。"""
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def draw_config(ax, q1, q2, title, col2_dx=0.0):
    """画一个构型：连杆 + 雅可比两列 + 速度椭球。返回末端位置。"""
    base, elbow, ee = fk(q1, q2)
    J = jacobian(q1, q2)

    # 速度椭球：单位关节速度球在末端速度空间的像
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    img = J @ np.vstack([np.cos(t), np.sin(t)])
    ax.fill(ee[0] + img[0], ee[1] + img[1], color=ORANGE, alpha=0.10, zorder=1)
    ax.plot(ee[0] + img[0], ee[1] + img[1], '--', color=ORANGE, lw=2.0, zorder=2)

    # 连杆与关节
    ax.plot([base[0], elbow[0], ee[0]], [base[1], elbow[1], ee[1]],
            '-o', color=GRAY, lw=5, markersize=11, markerfacecolor='white',
            markeredgewidth=2.5, markeredgecolor=GRAY, zorder=5)
    ax.plot(ee[0], ee[1], '*', color=GRAY, markersize=15, zorder=7)

    # 雅可比两列：单关节单位速度产生的末端速度
    tip1 = ee + J[:, 0]
    start2 = ee + np.array([col2_dx, 0.0])
    tip2 = start2 + J[:, 1]
    ax.annotate('', xy=tip1, xytext=ee,
                arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.6,
                                mutation_scale=18), zorder=6)
    ax.annotate('', xy=tip2, xytext=start2,
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=2.6,
                                mutation_scale=18), zorder=6)
    # 标签按箭头方向自适应摆放：接近竖直的箭头把标签放到右侧，
    # 否则放到箭尖上方，避免压住箭头本身。
    for tip, col, name, color in ((tip1, J[:, 0], r'$J_{:,1}$', BLUE),
                                  (tip2, J[:, 1], r'$J_{:,2}$', GREEN)):
        if abs(col[0]) < 0.30 * abs(col[1]):
            ax.text(tip[0] + 0.16, tip[1], name, color=color,
                    fontsize=12, ha='left', va='center', fontweight='bold', zorder=8)
        else:
            ax.text(tip[0], tip[1] + 0.18, name, color=color,
                    fontsize=12, ha='center', va='bottom', fontweight='bold', zorder=8)

    ax.text(0.03, 0.05, r'$\det J = %.2f$' % np.linalg.det(J),
            transform=ax.transAxes, fontsize=12, color='#333', va='bottom')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-0.7, 2.9)
    ax.set_ylim(-2.6, 2.6)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    return ee


# ── 图 1：弯曲 vs 完全伸直 ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))

draw_config(axes[0], 0.0, np.pi / 2, '(a) q2 = 90 deg: bent')
ee_b = draw_config(axes[1], 0.0, 0.0, '(b) q2 = 0 deg: fully extended', col2_dx=0.07)

# 右面板：沿臂方向不可达
axes[1].annotate('', xy=(ee_b[0] + 0.75, ee_b[1]), xytext=(ee_b[0] - 0.75, ee_b[1]),
                 arrowprops=dict(arrowstyle='<|-|>', color=RED, lw=2.0,
                                 linestyle='--', mutation_scale=14), zorder=4)
axes[1].text(0.30, 0.26, 'columns collinear:\nno motion along the arm',
             transform=axes[1].transAxes, ha='center', va='center', color=RED,
             fontsize=10.5, fontweight='bold')

plt.tight_layout()
plt.savefig('public/robot_2dof_jacobian_singularity.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print('OK public/robot_2dof_jacobian_singularity.png')

# ── 图 2：可操作度随肘角变化 ───────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4))
q2d = np.linspace(-180.0, 180.0, 1201)
w = np.abs(np.sin(np.deg2rad(q2d)))
ax2.plot(q2d, w, color=BLUE, lw=2.2, label=r'$w = L_1 L_2 |\sin q_2|$')
ax2.axhline(0.0, color=GRAY, lw=1.0, alpha=0.6)
for x in (-180.0, 0.0, 180.0):
    ax2.plot(x, 0.0, 'o', color=RED, markersize=7, zorder=5)
ax2.annotate('singular', xy=(0.0, 0.0), xytext=(10.0, 0.20), color=RED,
             fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
ax2.annotate('singular', xy=(-180.0, 0.0), xytext=(-172.0, 0.20), color=RED,
             fontsize=11, fontweight='bold')
ax2.annotate('singular', xy=(180.0, 0.0), xytext=(150.0, 0.30), color=RED,
             fontsize=11, fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
ax2.set_xlabel('Elbow angle q2 (deg)')
ax2.set_ylabel('Manipulability  w')
ax2.set_title('Two-link planar arm: manipulability vs elbow angle',
              fontsize=12, fontweight='bold')
ax2.set_xlim(-190.0, 190.0)
ax2.set_ylim(-0.06, 1.18)
ax2.grid(True, alpha=0.25)
ax2.legend(loc='lower center', fontsize=10)
fig2.tight_layout()
fig2.savefig('public/robot_2dof_manipulability_curve.png', dpi=150,
             bbox_inches='tight', facecolor='white')
plt.close(fig2)
print('OK public/robot_2dof_manipulability_curve.png')
