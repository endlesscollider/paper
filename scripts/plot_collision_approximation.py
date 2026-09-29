#!/usr/bin/env python3
"""为「碰撞近似类型」文章画三张图：

1. collision_approximation_lshape.png
   同一个 L 形凹零件，在 boundingCube / boundingSphere / convexHull /
   convexDecomposition / triangleMesh(SDF) 五种近似下的有效碰撞区域对比。
   填充色表示近似区域，灰虚线表示真实几何轮廓；红色区域是"幽灵接触"来源。

2. sdf_voxel_resolution.png
   SDF 网格分辨率对边界误差的影响：分辨率越低，等距离面越接近阶梯状，
   误差上限约等于体素对角线的 1/2。

3. collision_cost_accuracy_map.png
   各近似方案在"几何保真度—查询开销"平面上的位置（定性示意）。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BLUE = "#2196F3"
GREEN = "#4CAF50"
PURPLE = "#9C27B0"
ORANGE = "#FF9800"
RED = "#F44336"
GREY = "#607D8B"


def l_shape():
    """L 形凹多边形的顶点（逆时针）。"""
    return np.array([
        [0.0, 0.0], [2.0, 0.0], [2.0, 1.0],
        [1.0, 1.0], [1.0, 2.0], [0.0, 2.0],
    ])


def _panel(ax, idx, pts):
    """把第 idx 号面板画到 ax 上。idx 为全局编号，见 plot_approximations。"""
    def base(ax, title):
        ax.set_xlim(-0.35, 2.45)
        ax.set_ylim(-0.35, 2.45)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        # 真实几何轮廓
        ax.plot(pts[:, 0], pts[:, 1], '--', color=GREY, linewidth=1.6, zorder=1)

    def outline_fill(ax, poly, color, alpha=0.30, label=None):
        ax.add_patch(Polygon(poly, closed=True, facecolor=color,
                             edgecolor=color, alpha=alpha, linewidth=2, zorder=2))

    if idx == 0:                                    # ---- (a) 真实几何 ----
        base(ax, "(a) 真实几何\n（参考）")
        outline_fill(ax, pts, GREY, alpha=0.45)
        ax.text(1.0, -0.28, "凹角处有真实间隙", ha='center', fontsize=11, color=GREY)

    elif idx == 1:                                  # ---- (b) boundingCube ----
        base(ax, "(b) boundingCube")
        outline_fill(ax, pts, GREY, alpha=0.15)
        outline_fill(ax, np.array([[0, 0], [2, 0], [2, 2], [0, 2]]), ORANGE, alpha=0.30)
        ax.annotate("填满整个 AABB\n凹角被完全封死", xy=(0.55, 1.55), xytext=(0.62, 2.18),
                    fontsize=11, color=ORANGE, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.4))

    elif idx == 2:                                  # ---- (c) boundingSphere ----
        base(ax, "(c) boundingSphere")
        outline_fill(ax, pts, GREY, alpha=0.15)
        cx, cy = 1.0, 1.0                       # 3D AABB 中心 (1,1,0.5)，图中展示 z=0.5 的剖面
        r3 = 0.5 * np.sqrt(2.0 ** 2 + 2.0 ** 2 + 1.0 ** 2)  # 3D AABB 体对角线的一半 = 1.5
        ax.add_patch(Circle((cx, cy), r3, facecolor=PURPLE, edgecolor=PURPLE,
                            alpha=0.22, linewidth=2, zorder=2))
        ax.plot([cx], [cy], 'o', color=PURPLE, markersize=4)
        ax.text(1.0, -0.28, f"球半径 r = 0.5√(2²+2²+1²) = {r3:.1f} m（3D 体对角线的一半）",
                ha='center', fontsize=10, color=PURPLE)

    elif idx == 3:                                  # ---- (d) convexHull ----
        base(ax, "(d) convexHull")
        outline_fill(ax, pts, GREY, alpha=0.15)
        outline_fill(ax, np.array([[0, 0], [2, 0], [2, 1], [1, 2], [0, 2]]), BLUE, alpha=0.28)
        ax.annotate("凹角被一条斜面盖住\n(1,1) 附近出现幽灵接触", xy=(1.05, 1.05),
                    xytext=(0.05, 2.18), fontsize=11, color=BLUE, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.4))

    else:                                           # ---- (e) convexDecomposition / mesh ----
        base(ax, "(e) convexDecomposition\n≈ triangleMesh / SDF")
        outline_fill(ax, pts, GREEN, alpha=0.30)
        ax.plot([1, 1], [0, 1], '-', color=GREEN, linewidth=1.2, alpha=0.9, zorder=3)
        ax.plot([0, 1], [1, 1], '-', color=GREEN, linewidth=1.2, alpha=0.9, zorder=3)
        ax.annotate("两块凸块拼出凹角\n凹角间隙被保留", xy=(1.0, 0.5), xytext=(0.05, 2.18),
                    fontsize=11, color=GREEN, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.4))

    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)


def plot_approximations():
    """五联图：整组对比（用于文章开头）。"""
    pts = l_shape()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    for i, ax in enumerate(axes):
        _panel(ax, i, pts)
    plt.tight_layout()
    out = "public/collision_approximation_lshape.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {out}")


def plot_single_panels():
    """按小节切出的单幅图：让每个小节看到的就是那一幅近似，而不是重复的整组图。"""
    pts = l_shape()
    singles = [
        (0, "collision_approx_real_geometry.png", (4.8, 4.8)),
        (2, "collision_approx_bounding_sphere.png", (4.8, 4.8)),
        (3, "collision_approx_convex_hull.png", (4.8, 4.8)),
        (4, "collision_approx_convex_decomposition.png", (4.8, 4.8)),
    ]
    for idx, name, size in singles:
        fig, ax = plt.subplots(figsize=size)
        _panel(ax, idx, pts)
        ax.set_box_aspect(1)
        plt.tight_layout()
        out = f"public/{name}"
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ {out}")


def plot_sdf_resolution():
    """一维切片：SDF 网格分辨率与零等值面重建误差。

    真实表面位于 x=0。SDF 只在体素中心采样，因此把采样值线性插值回零时，
    重建出的表面位置会被量化到相邻两个体素中心之间；最坏情况误差为 h/2。
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    L = 1.0
    xs = np.linspace(-0.55, 0.55, 400)

    for ax, N in zip(axes, [8, 16, 64]):
        h = L / N
        # 体素中心坐标：表面 x=0 正好落在两个体素中心的中点上（最坏情况）
        centers = (np.arange(-N // 2, N // 2) + 0.5) * h
        samples = centers.copy()  # 真实带符号距离在体素中心的值
        for c in centers:
            ax.axvline(c, color=GREY, alpha=0.18, linewidth=0.8)
        ax.plot(xs, xs, color=BLUE, linewidth=2.2, label="真实距离 d = x")
        ax.step(centers, samples, where='mid', color=RED, linewidth=2.2,
                label="SDF 采样重建（体素中心取值）")
        err = h / 2
        ax.axhspan(-err, err, color=ORANGE, alpha=0.18)
        ax.axhline(0, color='k', linewidth=0.8)
        ax.axvline(0, color='k', linewidth=0.8, linestyle=':')
        ax.set_title(f"分辨率 N = {N}（体素边长 h = {h:.3f} m）", fontsize=12, fontweight='bold')
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)
        ax.set_xlabel("到表面的位置 x [m]", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.text(0.30, (err + 0.02), f"零等值面误差带 ±h/2 = ±{err:.3f} m",
                fontsize=9, color=ORANGE, fontweight='bold')

    axes[0].set_ylabel("带符号距离 [m]", fontsize=11)
    axes[0].legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    out = "public/sdf_voxel_resolution.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {out}")


def plot_cost_accuracy():
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.set_xlabel("形状保真度（对凹几何的还原程度，越高越好）", fontsize=12)
    ax.set_ylabel("每帧碰撞查询相对开销（定性，越高越贵）", fontsize=12)
    ax.set_title("碰撞近似方案的保真度—开销定位（定性示意）", fontsize=13, fontweight='bold')

    data = [
        ("boundingCube", 0.06, 0.05, ORANGE),
        ("boundingSphere", 0.05, 0.04, PURPLE),
        ("convexHull", 0.35, 0.35, BLUE),
        ("convexDecomposition", 0.75, 1.00, RED),
        ("meshSimplification", 0.82, 1.35, GREEN),
        ("triangleMesh (SDF)", 0.96, 0.60, "#009688"),
    ]
    # 轻微错位避免重叠
    for i, (name, acc, cost, color) in enumerate(data):
        ax.scatter(acc, cost, s=180, color=color, alpha=0.85, edgecolor='white', zorder=5)
        dy = 0.06 if name not in ("triangleMesh (SDF)",) else -0.09
        ax.annotate(name, xy=(acc, cost), xytext=(acc + 0.015, cost + dy),
                    fontsize=10, color=color, fontweight='bold')

    ax.plot([0.05, 0.96], [0.04, 0.60], '--', color=GREY, alpha=0.45,
            label="静态碰撞体：SDF 只需 cook 一次，查询便宜")
    ax.annotate("动态凹几何的\n安全区", xy=(0.9, 0.35), xytext=(0.66, 0.18),
                fontsize=10, color="#009688", fontweight='bold',
                arrowprops=dict(arrowstyle='->', color="#009688", lw=1.4))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.55)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    out = "public/collision_cost_accuracy_map.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {out}")


if __name__ == "__main__":
    plot_approximations()
    plot_single_panels()
    plot_sdf_resolution()
    plot_cost_accuracy()
