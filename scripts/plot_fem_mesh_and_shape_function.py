#!/usr/bin/env python3
"""
生成有限元方法(FEM)基础的两张图：
- 子图1：连续体 -> 三角形网格离散化示意（一个圆形区域被三角化）
- 子图2：单个三角形单元上,某个顶点的线性形函数 N_i 的3D曲面
  （形函数在自己所在顶点=1，在对边=0，中间线性过渡）
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

fig = plt.figure(figsize=(12, 5.2))

# ---------- 子图1：连续体离散化为三角网格 ----------
ax1 = fig.add_subplot(1, 2, 1)

# 用一个简单的规则三角网格填充一个近似圆形区域
theta_c = np.linspace(0, 2*np.pi, 100)
circle_x, circle_y = np.cos(theta_c), np.sin(theta_c)
ax1.plot(circle_x, circle_y, color='#9E9E9E', linewidth=1.5, linestyle='--', label='Continuous body boundary')

# 生成三角网格点（简单笛卡尔网格裁剪到圆内）
grid_pts = []
n_grid = 9
lin = np.linspace(-1.1, 1.1, n_grid)
for gx in lin:
    for gy in lin:
        if gx**2 + gy**2 <= 1.05**2:
            grid_pts.append((gx, gy))
grid_pts = np.array(grid_pts)

from scipy.spatial import Delaunay
tri = Delaunay(grid_pts)
ax1.triplot(grid_pts[:, 0], grid_pts[:, 1], tri.simplices, color='#2196F3', linewidth=0.9)
ax1.plot(grid_pts[:, 0], grid_pts[:, 1], 'o', color='#1565C0', markersize=2.5)

ax1.set_aspect('equal')
ax1.set_title('Continuous Body $\\rightarrow$ Discretized Tet/Tri Mesh', fontsize=11.5, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper right')
ax1.set_xlim(-1.3, 1.3)
ax1.set_ylim(-1.3, 1.3)
ax1.axis('off')

# ---------- 子图2：单个三角形上的线性形函数 N_i ----------
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# 三角形顶点
v0, v1, v2 = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])

def barycentric(p, v0, v1, v2):
    T = np.array([v0 - v2, v1 - v2]).T
    lam01 = np.linalg.solve(T, p - v2)
    lam2 = 1 - lam01[0] - lam01[1]
    return lam01[0], lam01[1], lam2

# 网格化三角形内部区域，画出 N0 (对应v0的形函数，就是重心坐标lambda0)
res = 40
xs = np.linspace(0, 1, res)
ys = np.linspace(0, 1, res)
X, Y = np.meshgrid(xs, ys)
N0 = np.full_like(X, np.nan)
for i in range(res):
    for j in range(res):
        p = np.array([X[i, j], Y[i, j]])
        if p[0] + p[1] <= 1.0 + 1e-9:
            l0, l1, l2 = barycentric(p, v0, v1, v2)
            N0[i, j] = l0

surf = ax2.plot_surface(X, Y, N0, cmap='viridis', edgecolor='none', alpha=0.9)
# 标出三角形三个顶点在z=对应值处
ax2.scatter([v0[0]], [v0[1]], [1], color='red', s=40)
ax2.text(v0[0], v0[1], 1.05, '$N_0=1$ at $v_0$', fontsize=8, color='red')
ax2.scatter([v1[0], v2[0]], [v1[1], v2[1]], [0, 0], color='blue', s=30)
ax2.text(v1[0], v1[1], 0.1, '$N_0=0$', fontsize=8, color='blue')

ax2.set_xlabel('$x$', fontsize=9)
ax2.set_ylabel('$y$', fontsize=9)
ax2.set_zlabel('$N_0(x,y)$', fontsize=9)
ax2.set_title('Linear Shape Function $N_0$\n(1 at its own vertex, 0 at others)', fontsize=11, fontweight='bold')
ax2.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.savefig('public/fem_mesh_discretization_and_shape_function.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/fem_mesh_discretization_and_shape_function.png")
