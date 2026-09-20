#!/usr/bin/env python3
"""
针孔相机 vs 鱼眼相机：入射角 -> 像面落点 对比示意图（中文标注版）。

画法：
  - 镜头中心在原点，光轴沿水平方向向右。
  - 像面是镜头中心右侧距离 f 处的一条竖线，浅灰色阴影范围代表
    传感器实际能接住光线的物理区域。
  - 对每个入射角 theta，画一条从镜头中心出发、扎到像面上的直线，
    落点距光轴的距离就是 r。两个子图的唯一区别是"扎到像面哪个
    高度"：
      左图（针孔）：r = f*tan(theta)，角度稍大落点就飙出传感器
      右图（鱼眼）：r = f*theta，落点随角度线性增长，同样的角度
                    范围全部还能落在传感器内
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

angles_deg = [20, 40, 60, 75]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
f = 1.0
sensor_half_h = 1.5
plot_ymax = 4.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.8))

for ax, mode in [(ax1, 'pinhole'), (ax2, 'fisheye')]:
    ax.annotate('', xy=(2.6, 0), xytext=(-0.4, 0), arrowprops=dict(arrowstyle='->', color='#999', lw=1))
    ax.text(2.6, -0.15, '光轴', fontsize=10, color='#777', ha='right', va='top')

    # 传感器可用范围
    ax.add_patch(plt.Rectangle((f - 0.05, -sensor_half_h), 0.1, 2 * sensor_half_h,
                                 facecolor='#90A4AE', alpha=0.35, zorder=2))
    ax.plot([f, f], [-sensor_half_h, sensor_half_h], color='#37474F', linewidth=3, zorder=4)
    ax.text(f, sensor_half_h + 0.12, '传感器', fontsize=10.5, color='#263238', ha='center')

    ax.plot(0, 0, 'o', markersize=11, color='#212121', zorder=6)
    ax.text(0, -0.28, '镜头中心', fontsize=10.5, color='#212121', ha='center')

    for theta_deg, color in zip(angles_deg, colors):
        theta = np.deg2rad(theta_deg)
        r = f * np.tan(theta) if mode == 'pinhole' else f * theta

        if r <= plot_ymax:
            ax.plot([0, f], [0, r], '-', color=color, linewidth=2, zorder=3)
            ax.plot(f, r, 'o', markersize=7, color=color, zorder=7)
            on_sensor = r <= sensor_half_h
            tag = f'{theta_deg}°  →  r={r:.2f}' + ('' if on_sensor else '（超出传感器！）')
            ax.text(f + 0.12, r, tag, fontsize=9.2, color=color, va='center',
                    fontweight='bold' if not on_sensor else 'normal')
        else:
            end_x = min(2.5, f + 0.5)
            end_y = min(plot_ymax - 0.1, end_x * np.tan(theta) if mode == 'pinhole' else end_x * theta)
            ax.annotate('', xy=(end_x, plot_ymax - 0.15), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2, linestyle='-'))
            ax.text(end_x + 0.05, plot_ymax - 0.15, f'{theta_deg}°  →  r={r:.1f}\n（直接飞出画面！）',
                    fontsize=9.2, color=color, va='top', fontweight='bold')

    ax.set_xlim(-0.7, 3.1)
    ax.set_ylim(-0.6, plot_ymax + 0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# 针孔图：画直角三角形标注 theta / f / r 的关系（用 40° 这条为例）
theta_demo = np.deg2rad(40)
r_demo = f * np.tan(theta_demo)
ax1.plot([0, f], [0, 0], ':', color='#555', linewidth=1.3, zorder=3)
ax1.plot([f, f], [0, r_demo], ':', color='#555', linewidth=1.3, zorder=3)
ax1.text(f / 2, -0.22, '$f$', fontsize=11, color='#555', ha='center')
ax1.text(f + 0.14, r_demo / 2, '$r$', fontsize=11, color='#555', va='center')

ax1.set_title('针孔模型：r = f·tanθ', fontsize=13, fontweight='bold', color='#B71C1C')
ax1.text(1.15, plot_ymax - 0.55,
         'θ 越大，r 爆炸式增长\n60°/75° 已经超出\n有限尺寸的传感器',
         fontsize=9.5, color='#B71C1C',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.9))

ax2.set_title('鱼眼模型：r = f·θ', fontsize=13, fontweight='bold', color='#1565C0')
ax2.text(1.15, plot_ymax - 0.55,
         'r 只随 θ 线性增长\n即使 75° 也依然\n落在传感器范围内',
         fontsize=9.5, color='#1565C0',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.9))

fig.suptitle('同一个入射角 θ，两种公式把它换算成像面上不同的落点距离 r',
             fontsize=12, y=1.0)

plt.tight_layout()
plt.savefig('public/pinhole_vs_fisheye_optics.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/pinhole_vs_fisheye_optics.png")
