#!/usr/bin/env python3
"""
"等距投影 r = f·θ 是怎么来的" 几何示意图（中文标注）。

核心直觉：针孔模型是把光线"拍扁"投影到一块平的像平面上（用直线传播 +
正切函数）；鱼眼的等距投影，可以理解成先假想一个以镜头中心为圆心、
半径为 f 的圆弧（代表镜头前组把光线"弯"到的一个虚拟参考球面），
入射角 theta 对应的光线在这个圆弧上走过的弧长，按弧长公式
s = f * theta（theta 用弧度）——然后镜头把这段弧长"原样展开、不拉伸
不压缩"地铺到平的传感器上，就是像面上的落点距离 r。所以 r = f*theta
的物理含义是：**像面距离 = 参考圆弧上的弧长，弧长展开时保持原长不变**。

画法：
  左图：以镜头中心 O 为圆心画一个半径 f 的四分之一圆弧，标出几个不同
        入射角 theta 对应的半径线，并把每个角度对应的圆弧段用不同颜色
        高亮，标注其弧长 = f*theta。
  右图："展开"示意——把左图里每一段圆弧，原样拉直排列在一条直线上
        （代表传感器平面），可以看到角度间隔相等时，展开后的距离间隔
        也相等——这就是"等距"（equidistant）这个名字的来源，也是
        r = f*theta 这个线性关系的几何本质。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

f = 1.6
angles_deg = [20, 40, 60, 80]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ============ 左图：镜头中心 + 参考圆弧 + 弧长 ============
O = np.array([0.0, 0.0])

# 完整的参考圆弧（0°~90°），浅色底
full_arc_t = np.linspace(0, np.pi / 2, 200)
ax1.plot(f * np.cos(full_arc_t), f * np.sin(full_arc_t), '-', color='#CFD8DC', linewidth=6, alpha=0.6, zorder=1)

# 光轴
ax1.annotate('', xy=(f + 0.3, 0), xytext=(-0.2, 0), arrowprops=dict(arrowstyle='->', color='#999', lw=1))
ax1.text(f + 0.3, -0.12, '光轴', fontsize=9.5, color='#777', ha='right', va='top')

# 镜头中心
ax1.plot(*O, 'o', markersize=10, color='#212121', zorder=6)
ax1.text(O[0], O[1] - 0.22, '镜头中心 O\n(参考圆弧的圆心)', fontsize=9, color='#212121', ha='center', va='top')

prev_theta = 0.0
for theta_deg, color in zip(angles_deg, colors):
    theta = np.deg2rad(theta_deg)
    # 高亮从 prev_theta 到 theta 的这一段弧（不同角度区间不同颜色，直观显示"每段弧对应一个角度增量"）
    seg_t = np.linspace(prev_theta, theta, 60)
    ax1.plot(f * np.cos(seg_t), f * np.sin(seg_t), '-', color=color, linewidth=4, zorder=3)
    # 半径线（从O到圆弧上该角度的点）
    px, py = f * np.cos(theta), f * np.sin(theta)
    ax1.plot([O[0], px], [O[1], py], '--', color=color, linewidth=1.3, alpha=0.8, zorder=2)
    ax1.plot(px, py, 'o', markersize=6, color=color, zorder=6)
    arc_len = f * theta
    ax1.text(px + 0.08, py + 0.05, f'{theta_deg}°\n弧长={arc_len:.2f}', fontsize=8.3, color=color, va='bottom')
    prev_theta = theta

ax1.text(f * 1.02, 0.15, '参考圆弧\n(半径 f)', fontsize=9, color='#78909C', rotation=60)
ax1.set_xlim(-0.5, f + 1.4)
ax1.set_ylim(-0.5, f + 0.6)
ax1.set_aspect('equal')
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.set_title('第一步：入射角 θ 对应\n参考圆弧上的一段弧长 s=f·θ', fontsize=12, fontweight='bold')

# ============ 右图：把弧长原样展开到平的传感器上 ============
ax2.axhline(y=0, color='#37474F', linewidth=3, zorder=2)
ax2.text(-0.3, 0.15, '传感器\n(展开后是平的)', fontsize=9.5, color='#263238', ha='left')

prev_r = 0.0
prev_theta = 0.0
for theta_deg, color in zip(angles_deg, colors):
    theta = np.deg2rad(theta_deg)
    r = f * theta
    # 展开后的这一段（长度 = 弧长，和左图对应颜色的弧长完全相等）
    ax2.plot([prev_r, r], [0, 0], '-', color=color, linewidth=5, zorder=3, solid_capstyle='butt')
    ax2.plot(r, 0, 'o', markersize=7, color=color, zorder=6)
    ax2.plot([r, r], [-0.08, 0.08], '-', color=color, linewidth=1.3, zorder=4)
    ax2.text(r, 0.18 + 0.16 * (['20','40','60','80'].index(str(theta_deg)) % 2),
              f'r={r:.2f}\n({theta_deg}°)', fontsize=8.6, color=color, ha='center')
    prev_r = r
    prev_theta = theta

ax2.plot(0, 0, 'o', markersize=9, color='#212121', zorder=6)
ax2.text(0, -0.28, '光轴与传感器\n交点 (r=0)', fontsize=8.8, color='#212121', ha='center', va='top')

ax2.set_xlim(-0.5, f * np.deg2rad(80) + 0.6)
ax2.set_ylim(-0.6, 0.9)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.set_title('第二步：把每一段弧长原样"拉直"\n铺到平的传感器上，长度不变', fontsize=12, fontweight='bold')

ax2.text(0.5, -0.5,
         '关键观察：左图 20°→40°→60°→80° 每次角度增加 20°，\n'
         '对应的彩色弧段长度都一样长——展开后（右图）间距也完全相等。\n'
         '这就是"等距"(equidistant)的含义：角度每变化相同的量，\n'
         '像面距离 r 也变化相同的量，两者是线性关系 r=f·θ。',
         transform=ax2.transData, fontsize=9.3, color='#455A64', ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1', alpha=0.92))

plt.tight_layout()
plt.savefig('public/equidistant_arc_unroll.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/equidistant_arc_unroll.png")
