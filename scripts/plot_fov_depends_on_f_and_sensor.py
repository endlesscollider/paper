#!/usr/bin/env python3
"""
"FOV 到底由什么决定" 示意图（中文标注）。

核心结论：FOV 不是镜头单独决定的一个数字，而是"镜头的投影曲线 r(θ)"
和"传感器物理尺寸（决定 r 的截止范围 r_max）"两者共同决定的。

左图：固定镜头（固定焦距 f，固定 r=f·θ 曲线），换用三种不同尺寸的
      传感器（r_max 不同）。传感器越大，能截到曲线上更大的 θ，FOV 越大。
右图：固定传感器尺寸（固定 r_max），换用三种不同焦距的镜头（f 越小，
      曲线越平，同样的 r_max 能截到更大的 θ）。焦距越短，FOV 越大。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

theta_max_plot = np.deg2rad(100)
theta = np.linspace(0, theta_max_plot, 400)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# ============ 左图：固定焦距 f，变化传感器半宽 r_max ============
f_fixed = 1.0
r_curve = f_fixed * theta  # 等距投影 r = f*theta

ax1.plot(np.rad2deg(theta), r_curve, '-', color='#2196F3', linewidth=2.3, label=f'镜头曲线 r=f·θ (f={f_fixed})')

sensor_sizes = [1.0, 1.6, 2.2]
sensor_colors = ['#F44336', '#FF9800', '#4CAF50']
sensor_labels = ['小传感器', '中传感器', '大传感器']

for r_max, color, label in zip(sensor_sizes, sensor_colors, sensor_labels):
    theta_max = r_max / f_fixed
    fov = 2 * np.rad2deg(theta_max)
    ax1.axhline(y=r_max, color=color, linestyle='--', linewidth=1.3, alpha=0.7)
    ax1.plot(np.rad2deg(theta_max), r_max, 'o', markersize=8, color=color, zorder=6)
    ax1.annotate('', xy=(np.rad2deg(theta_max), 0), xytext=(np.rad2deg(theta_max), r_max),
                 arrowprops=dict(arrowstyle='-', color=color, lw=1, linestyle=':'))
    ax1.text(np.rad2deg(theta_max) + 1.5, r_max - 0.08,
              f'{label} r_max={r_max}\nθ_max={np.rad2deg(theta_max):.0f}° → FOV={fov:.0f}°',
              fontsize=8.6, color=color, va='top')

ax1.set_xlabel('入射角 θ（度）', fontsize=11)
ax1.set_ylabel('像面距离 r', fontsize=11)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 2.6)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_title('固定镜头(f不变)，换用不同尺寸的传感器\n→ 传感器越大，能截到的θ越大，FOV越大', fontsize=11.5, fontweight='bold')

# ============ 右图：固定传感器 r_max，变化焦距 f ============
r_max_fixed = 1.6
f_values = [0.7, 1.0, 1.5]
f_colors = ['#4CAF50', '#2196F3', '#F44336']

for f_val, color in zip(f_values, f_colors):
    r_curve_f = f_val * theta
    ax2.plot(np.rad2deg(theta), r_curve_f, '-', color=color, linewidth=2.3, label=f'f={f_val}')
    theta_max = r_max_fixed / f_val
    if np.rad2deg(theta_max) <= 100:
        fov = 2 * np.rad2deg(theta_max)
        ax2.plot(np.rad2deg(theta_max), r_max_fixed, 'o', markersize=8, color=color, zorder=6)
        ax2.text(np.rad2deg(theta_max) + 1.5, r_max_fixed + (0.15 if f_val != 1.0 else -0.32),
                  f'θ_max={np.rad2deg(theta_max):.0f}°\nFOV={fov:.0f}°',
                  fontsize=8.6, color=color, va='center')

ax2.axhline(y=r_max_fixed, color='#616161', linestyle='--', linewidth=1.5, alpha=0.8, label=f'传感器半宽 r_max={r_max_fixed}(固定)')

ax2.set_xlabel('入射角 θ（度）', fontsize=11)
ax2.set_ylabel('像面距离 r', fontsize=11)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 2.6)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='upper left')
ax2.set_title('固定传感器尺寸不变，换用不同焦距的镜头\n→ 焦距越短，曲线越平，FOV越大', fontsize=11.5, fontweight='bold')

fig.suptitle('FOV = 2 × (曲线在 r=r_max 处对应的 θ) —— 由镜头焦距 f 和传感器尺寸共同决定',
             fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig('public/fov_depends_on_f_and_sensor.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/fov_depends_on_f_and_sensor.png")
