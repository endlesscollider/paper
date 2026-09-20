#!/usr/bin/env python3
"""
生成MPM中常用的二次B-spline插值权重函数曲线图，
展示粒子对周围网格节点的影响权重如何随距离平滑衰减到0。
"""
import numpy as np
import matplotlib.pyplot as plt

def quadratic_bspline(x):
    """
    分段二次B-spline基函数（以网格间距为单位的距离 x）。
    |x| < 0.5: 3/4 - x^2
    0.5 <= |x| < 1.5: 0.5*(1.5-|x|)^2
    |x| >= 1.5: 0
    """
    ax = np.abs(x)
    w = np.zeros_like(x)
    mask1 = ax < 0.5
    mask2 = (ax >= 0.5) & (ax < 1.5)
    w[mask1] = 0.75 - ax[mask1]**2
    w[mask2] = 0.5 * (1.5 - ax[mask2])**2
    return w

x = np.linspace(-2, 2, 500)
w = quadratic_bspline(x)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(x, w, color='#2196F3', linewidth=2.4)
ax.fill_between(x, w, alpha=0.12, color='#2196F3')

for xi in [-1, 0, 1]:
    ax.axvline(x=xi, color='#BDBDBD', linestyle=':', linewidth=1.0)
ax.plot([-1, 0, 1], quadratic_bspline(np.array([-1.0, 0.0, 1.0])), 'o', color='#F44336', markersize=6, zorder=5)
ax.text(0, quadratic_bspline(np.array([0.0]))[0] + 0.04, 'grid node\n(distance=0)', fontsize=8, ha='center', color='#B71C1C')

ax.set_xlabel('Distance from particle to grid node (in grid-cell units)', fontsize=10.5)
ax.set_ylabel('Interpolation weight $w$', fontsize=11)
ax.set_title('Quadratic B-Spline Weight Function\n(used for P2G / G2P transfer)', fontsize=11.5, fontweight='bold')
ax.set_xlim(-2, 2)
ax.set_ylim(0, 0.85)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('public/mpm_quadratic_bspline_weight.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/mpm_quadratic_bspline_weight.png")
