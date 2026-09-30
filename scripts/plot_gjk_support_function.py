#!/usr/bin/env python3
"""
为「碰撞近似类型」文章画三张 GJK / 支持函数示意图：

1. gjk_support_query.png
   二维凸形体 A 上，给定方向 d，找到最远点（支持点）的几何含义：
   把所有顶点向 d 轴投影，取最大投影对应的点。

2. gjk_minkowski_diff.png
   两个不相交的凸形体 A、B，以及它们的 Minkowski 差 C = A ⊖ B：
   原点在 C 外部 → 不相交。

3. gjk_iteration.png
   GJK 的三步迭代：单纯形如何夹逼原点，最终确定「不相交」。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import ConvexHull

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 颜色 ──────────────────────────────────────────────────────────────────────
C_A      = "#2196F3"   # shape A 蓝
C_B      = "#FF9800"   # shape B 橙
C_DIFF   = "#9C27B0"   # Minkowski 差 紫
C_ORIGIN = "#F44336"   # 原点 红
C_ARROW  = "#1B5E20"   # 方向向量 深绿
C_PROJ   = "#607D8B"   # 投影辅助线 灰蓝
C_SUPP   = "#F44336"   # 支持点 红
C_SIMPLEX= "#E91E63"   # 单纯形 粉红

# ── 形体定义 ──────────────────────────────────────────────────────────────────
# A：三角形，左侧
A_verts = np.array([[0.0, 0.0],
                    [2.0, 0.0],
                    [1.0, 2.0]])

# B：四边形（不规则），右侧
B_verts = np.array([[3.0, 0.2],
                    [4.2, 0.0],
                    [4.5, 1.2],
                    [3.2, 1.5]])


def poly_patch(verts, color, alpha=0.25, lw=2, zorder=2):
    from matplotlib.patches import Polygon as MPoly
    return MPoly(verts, closed=True,
                 facecolor=color, edgecolor=color,
                 alpha=alpha, linewidth=lw, zorder=zorder)


def support(verts, d):
    """支持函数：返回顶点中沿 d 方向投影最大的那个点。"""
    dots = verts @ d
    idx = np.argmax(dots)
    return verts[idx], idx, dots


def mink_diff_verts(A, B):
    """计算 Minkowski 差 A ⊖ B 的凸包顶点。"""
    pts = []
    for a in A:
        for b in B:
            pts.append(a - b)
    pts = np.array(pts)
    hull = ConvexHull(pts)
    return pts[hull.vertices]


# ══════════════════════════════════════════════════════════════════════════════
# 图 1：支持函数——「给一个方向，找最远点」
# ══════════════════════════════════════════════════════════════════════════════
def plot_support_query(path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("支持函数：给一个方向 d，找形体在该方向上的最远点",
                 fontsize=14, fontweight='bold', y=1.01)

    # ── 左图：投影示意 ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("(a) 把所有顶点向方向 d 投影，取最大值", fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(-1.2, 3.0)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)

    # 画形体 A
    ax.add_patch(poly_patch(A_verts, C_A, alpha=0.20))
    tri_closed = np.vstack([A_verts, A_verts[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color=C_A, lw=2, zorder=3)
    ax.text(1.0, 0.55, "A", fontsize=16, color=C_A, fontweight='bold',
            ha='center', va='center')

    # 方向向量 d（指向右上方）
    d_raw = np.array([1.0, 0.5])
    d = d_raw / np.linalg.norm(d_raw)
    ax.annotate("", xy=(0.0 + d[0]*1.4, 0.0 + d[1]*1.4),
                xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.5))
    ax.text(0.0 + d[0]*1.5, 0.0 + d[1]*1.5 + 0.05,
            r"$\mathbf{d}$", fontsize=14, color=C_ARROW)

    # 计算支持点及各顶点投影
    supp_pt, supp_idx, dots = support(A_verts, d)

    # 投影轴（沿 d 方向的射线）
    t_vals = np.linspace(-0.3, 2.5, 2)
    proj_line_origin = np.array([2.0, -0.8])
    ax.plot(proj_line_origin[0] + t_vals * d[0],
            proj_line_origin[1] + t_vals * d[1],
            '--', color=C_PROJ, lw=1.2, alpha=0.7, zorder=1)
    ax.text(proj_line_origin[0] + 2.5*d[0] + 0.05,
            proj_line_origin[1] + 2.5*d[1],
            "投影轴", fontsize=10, color=C_PROJ, alpha=0.8)

    # 每个顶点的投影脚
    labels = ['$v_1$', '$v_2$', '$v_3$']
    for i, (v, dot) in enumerate(zip(A_verts, dots)):
        foot = proj_line_origin + dot * d
        ax.plot([v[0], foot[0]], [v[1], foot[1]],
                ':', color=C_PROJ, lw=1.0, alpha=0.6, zorder=1)
        ax.plot(*foot, 'o', color=C_PROJ, ms=5, zorder=4)
        ax.plot(*v, 'o', color=C_A, ms=7, zorder=5)
        ax.annotate(f"{labels[i]}\n投影={dot:.2f}",
                    xy=v, xytext=(v[0] - 0.55, v[1] + 0.18),
                    fontsize=9, color=C_A,
                    arrowprops=dict(arrowstyle='->', color=C_A, lw=1.0))

    # 高亮支持点
    ax.plot(*supp_pt, '*', color=C_SUPP, ms=18, zorder=6,
            label=f"支持点 $h_A(\\mathbf{{d}})$（投影最大={dots[supp_idx]:.2f}）")
    ax.annotate(f"支持点\n投影最大 = {dots[supp_idx]:.2f}",
                xy=supp_pt, xytext=(supp_pt[0] + 0.3, supp_pt[1] + 0.4),
                fontsize=10, color=C_SUPP, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_SUPP, lw=1.5))
    ax.legend(fontsize=9, loc='upper left')

    # ── 右图：几何直觉——「把形体压扁到 d 轴，取最右端」 ─────────────────────
    ax2 = axes[1]
    ax2.set_title("(b) 直觉：把整个形体「压」到方向轴，取最远端", fontsize=12)
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.5, 3.0)
    ax2.set_ylim(-1.5, 3.0)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlabel("x", fontsize=11)
    ax2.set_ylabel("y", fontsize=11)

    ax2.add_patch(poly_patch(A_verts, C_A, alpha=0.20))
    ax2.plot(tri_closed[:, 0], tri_closed[:, 1], color=C_A, lw=2, zorder=3)
    ax2.text(1.0, 0.55, "A", fontsize=16, color=C_A, fontweight='bold',
             ha='center', va='center')

    # 方向 d 的阴影区域（半平面边界与支持超平面）
    # 支持超平面：{ x : d^T x = h_A(d) }，过支持点，法线为 d
    h = dots[supp_idx]
    # 超平面上两点（垂直于 d）
    perp = np.array([-d[1], d[0]])
    p1 = supp_pt + perp * 2.0
    p2 = supp_pt - perp * 2.0
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
             '-', color=C_SUPP, lw=2.0, alpha=0.8, zorder=3,
             label="支持超平面（形体在此平面的「最前沿」）")
    ax2.fill_betweenx([p2[1], p1[1]], [p2[0], p1[0]],
                      [p2[0] + d[0]*3, p1[0] + d[0]*3],
                      alpha=0.06, color=C_SUPP)

    ax2.annotate("", xy=(0.0 + d[0]*1.4, 0.0 + d[1]*1.4),
                 xytext=(0.0, 0.0),
                 arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.5))
    ax2.text(0.0 + d[0]*1.6, 0.0 + d[1]*1.6,
             r"$\mathbf{d}$", fontsize=14, color=C_ARROW)

    ax2.plot(*supp_pt, '*', color=C_SUPP, ms=18, zorder=6)
    ax2.annotate("支持点\n整个 A 都在这条\n超平面的后方",
                 xy=supp_pt, xytext=(supp_pt[0] + 0.2, supp_pt[1] + 0.5),
                 fontsize=10, color=C_SUPP, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_SUPP, lw=1.5))
    ax2.legend(fontsize=9, loc='upper left')

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 图 2：Minkowski 差——「A ⊖ B，原点在内 ↔ 相交」
# ══════════════════════════════════════════════════════════════════════════════
def plot_minkowski_diff(path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("Minkowski 差：把「A 和 B 是否相交」转化为「原点是否在 C = A ⊖ B 内」",
                 fontsize=13, fontweight='bold', y=1.01)

    d_query = np.array([1.0, 0.3])
    d_query /= np.linalg.norm(d_query)

    # ── 左图：A 和 B（不相交）────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("(a) 形体 A（三角形）和 B（四边形）\n互不相交", fontsize=11)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(True, alpha=0.2)

    ax.add_patch(poly_patch(A_verts, C_A, alpha=0.30))
    tri_c = np.vstack([A_verts, A_verts[0]])
    ax.plot(tri_c[:, 0], tri_c[:, 1], color=C_A, lw=2)
    ax.text(1.0, 0.65, "A", fontsize=16, color=C_A, fontweight='bold',
            ha='center', va='center')

    ax.add_patch(poly_patch(B_verts, C_B, alpha=0.30))
    quad_c = np.vstack([B_verts, B_verts[0]])
    ax.plot(quad_c[:, 0], quad_c[:, 1], color=C_B, lw=2)
    ax.text(3.85, 0.7, "B", fontsize=16, color=C_B, fontweight='bold',
            ha='center', va='center')

    # 标注支持查询方向
    ax.annotate("", xy=(1.0 + d_query[0]*0.9, 0.65 + d_query[1]*0.9),
                xytext=(1.0, 0.65),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.0))
    ax.text(1.0 + d_query[0]*1.05, 0.65 + d_query[1]*1.05 + 0.1,
            r"$\mathbf{d}$", fontsize=13, color=C_ARROW)

    # 支持点标注
    sa, _, _ = support(A_verts, d_query)
    sb, _, _ = support(B_verts, -d_query)
    ax.plot(*sa, '*', color=C_A, ms=14, zorder=5)
    ax.annotate(r"$a^*$ = support$_A(\mathbf{d})$",
                xy=sa, xytext=(sa[0] - 1.0, sa[1] + 0.3),
                fontsize=9, color=C_A,
                arrowprops=dict(arrowstyle='->', color=C_A, lw=1.0))
    ax.plot(*sb, '*', color=C_B, ms=14, zorder=5)
    ax.annotate(r"$b^*$ = support$_B(-\mathbf{d})$",
                xy=sb, xytext=(sb[0] + 0.05, sb[1] - 0.35),
                fontsize=9, color=C_B,
                arrowprops=dict(arrowstyle='->', color=C_B, lw=1.0))

    # ── 中图：A 与 B 相交的反例 ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("(b) 若 B 移近（两形体相交）\n原点落入 C 内部", fontsize=11)
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.5, 5.0)
    ax2.set_ylim(-0.5, 2.5)
    ax2.grid(True, alpha=0.2)

    B_shifted = B_verts - np.array([2.0, 0.0])  # 向左移 2，使其与 A 重叠
    ax2.add_patch(poly_patch(A_verts, C_A, alpha=0.30))
    ax2.plot(tri_c[:, 0], tri_c[:, 1], color=C_A, lw=2)
    ax2.text(1.0, 0.65, "A", fontsize=16, color=C_A, fontweight='bold',
             ha='center', va='center')
    ax2.add_patch(poly_patch(B_shifted, C_B, alpha=0.30))
    quad_cs = np.vstack([B_shifted, B_shifted[0]])
    ax2.plot(quad_cs[:, 0], quad_cs[:, 1], color=C_B, lw=2)
    ax2.text(1.85, 0.7, "B'", fontsize=14, color=C_B, fontweight='bold',
             ha='center', va='center')
    ax2.text(2.3, 2.1, "← 两形体现在重叠", fontsize=10, color='gray',
             style='italic')

    mink2 = mink_diff_verts(A_verts, B_shifted)
    m2c = np.vstack([mink2, mink2[0]])
    ax2.add_patch(poly_patch(mink2, C_DIFF, alpha=0.15))
    # （此图放小版 Minkowski 差以节省空间）
    # 画在右上角
    offset = np.array([2.8, 1.2])
    sc = 0.35
    m2_small = mink2 * sc + offset
    ax2.add_patch(poly_patch(m2_small, C_DIFF, alpha=0.30))
    m2s_c = np.vstack([m2_small, m2_small[0]])
    ax2.plot(m2s_c[:, 0], m2s_c[:, 1], color=C_DIFF, lw=1.5)
    ax2.plot(*offset, 'o', color=C_ORIGIN, ms=9, zorder=6)
    ax2.text(offset[0] + 0.08, offset[1] + 0.08, "原点\n（在 C 内）",
             fontsize=8, color=C_ORIGIN, fontweight='bold')
    ax2.text(offset[0] + 0.35, offset[1] - 0.25,
             "C = A ⊖ B'\n（缩略）", fontsize=8, color=C_DIFF)

    # ── 右图：Minkowski 差（不相交情形，完整） ────────────────────────────────
    ax3 = axes[2]
    ax3.set_title("(c) 不相交时：C = A ⊖ B\n原点在 C 外部", fontsize=11)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.2)

    mink = mink_diff_verts(A_verts, B_verts)
    ax3.add_patch(poly_patch(mink, C_DIFF, alpha=0.25))
    mink_c = np.vstack([mink, mink[0]])
    ax3.plot(mink_c[:, 0], mink_c[:, 1], color=C_DIFF, lw=2)

    # 标注所有 a - b 点
    for a in A_verts:
        for b in B_verts:
            ax3.plot(*(a - b), '.', color=C_DIFF, ms=5, alpha=0.5, zorder=3)

    # 原点
    ax3.plot(0, 0, 'o', color=C_ORIGIN, ms=12, zorder=7,
             label="原点（0, 0）")
    ax3.text(0.05, 0.08, "原点\n（在 C 外 → 不相交）",
             fontsize=10, color=C_ORIGIN, fontweight='bold')

    # 标注支持点对应的 Minkowski 差上的点
    sa2, _, _ = support(A_verts, d_query)
    sb2, _, _ = support(B_verts, -d_query)
    p = sa2 - sb2
    ax3.plot(*p, '*', color=C_ARROW, ms=14, zorder=6)
    ax3.annotate(r"$p = a^* - b^*$" + f"\n= ({p[0]:.1f}, {p[1]:.1f})",
                 xy=p, xytext=(p[0] + 0.4, p[1] + 0.3),
                 fontsize=9, color=C_ARROW,
                 arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))

    # C 的标注
    cx, cy = mink.mean(axis=0)
    ax3.text(cx, cy, "C = A ⊖ B", fontsize=13, color=C_DIFF,
             fontweight='bold', ha='center', va='center')
    ax3.legend(fontsize=9, loc='upper right')

    xmin, xmax = mink[:, 0].min() - 0.5, mink[:, 0].max() + 0.5
    ymin, ymax = mink[:, 1].min() - 0.5, mink[:, 1].max() + 0.5
    ax3.set_xlim(xmin, xmax)
    ax3.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 图 3：GJK 迭代——单纯形如何夹逼原点
# ══════════════════════════════════════════════════════════════════════════════
def plot_gjk_iteration(path):
    """在 Minkowski 差 C 上画 3 步 GJK 迭代（不相交情形）。"""
    mink = mink_diff_verts(A_verts, B_verts)
    mink_hull_closed = np.vstack([mink, mink[0]])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("GJK 三步迭代：在 Minkowski 差 C 上找「距原点最近的点」",
                 fontsize=13, fontweight='bold', y=1.01)

    def base_ax(ax, title, step):
        ax.set_title(f"步骤 {step}：{title}", fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(-5.0, 0.8)
        ax.set_ylim(-2.2, 2.2)
        ax.grid(True, alpha=0.2)
        ax.add_patch(poly_patch(mink, C_DIFF, alpha=0.15))
        ax.plot(mink_hull_closed[:, 0], mink_hull_closed[:, 1],
                color=C_DIFF, lw=1.5, alpha=0.6)
        ax.plot(0, 0, 'o', color=C_ORIGIN, ms=10, zorder=7)
        ax.text(0.05, 0.1, "原点", fontsize=9, color=C_ORIGIN)

    # 迭代 1：初始方向 d = (1, 0)，找到 C 上的第一个点
    d1 = np.array([1.0, 0.0])
    # support_C(d1) = support_A(d1) - support_B(-d1)
    sa1, _, _ = support(A_verts, d1)
    sb1, _, _ = support(B_verts, -d1)
    p1 = sa1 - sb1   # 第一个 simplex 点

    ax = axes[0]
    base_ax(ax, "初始查询方向 d = (1, 0)", 1)

    # 画方向箭头（从原点出发）
    ax.annotate("", xy=(0 + d1[0]*0.8, 0 + d1[1]*0.8),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.0))
    ax.text(0 + d1[0]*0.85, d1[1]*0.85 + 0.12,
            r"$\mathbf{d}_1=(1,0)$", fontsize=10, color=C_ARROW)

    # 画 p1
    ax.plot(*p1, 'D', color=C_SIMPLEX, ms=10, zorder=6, label=f"$p_1$=({p1[0]:.1f},{p1[1]:.1f})")
    ax.annotate(f"$p_1$=({p1[0]:.1f},{p1[1]:.1f})\n← C 在 d₁ 方向最远",
                xy=p1, xytext=(p1[0] - 1.5, p1[1] + 0.5),
                fontsize=9, color=C_SIMPLEX, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_SIMPLEX, lw=1.2))

    # 检查：p1·d1 > 0，继续
    check = np.dot(p1, d1)
    ax.text(-4.5, -1.8,
            f"检查：$p_1 \\cdot d_1$ = {check:.2f} > 0\n→ 原点方向还有探索空间，继续",
            fontsize=9, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    ax.legend(fontsize=9, loc='upper left')

    # 迭代 2：新方向从 p1 指向原点
    d2 = -p1 / np.linalg.norm(p1)  # 从 p1 指向原点
    sa2, _, _ = support(A_verts, d2)
    sb2, _, _ = support(B_verts, -d2)
    p2 = sa2 - sb2   # 第二个 simplex 点
    # 单纯形现在是线段 [p1, p2]
    # 距原点最近的点在线段上
    def closest_on_segment(p, q, o=np.array([0.0, 0.0])):
        pq = q - p
        t = np.clip(np.dot(o - p, pq) / np.dot(pq, pq), 0.0, 1.0)
        return p + t * pq

    closest2 = closest_on_segment(p1, p2)
    d3 = -closest2 / np.linalg.norm(closest2)

    ax2 = axes[1]
    base_ax(ax2, "单纯形扩展为线段 [p₁, p₂]", 2)

    ax2.annotate("", xy=(0 + d2[0]*0.8, 0 + d2[1]*0.8),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.0))
    ax2.text(d2[0]*0.85 - 0.2, d2[1]*0.85 + 0.12,
             r"$\mathbf{d}_2$（指向原点）", fontsize=10, color=C_ARROW)

    # 旧点
    ax2.plot(*p1, 'D', color=C_SIMPLEX, ms=8, zorder=5, alpha=0.5)
    ax2.text(p1[0] + 0.1, p1[1] - 0.25, "$p_1$", fontsize=9, color=C_SIMPLEX, alpha=0.7)
    # 新点
    ax2.plot(*p2, 'D', color=C_SIMPLEX, ms=10, zorder=6)
    ax2.annotate(f"$p_2$=({p2[0]:.1f},{p2[1]:.1f})",
                 xy=p2, xytext=(p2[0] - 2.0, p2[1] + 0.4),
                 fontsize=9, color=C_SIMPLEX, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_SIMPLEX, lw=1.2))
    # 线段
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=C_SIMPLEX, lw=2.5, zorder=5)
    # 最近点
    ax2.plot(*closest2, 's', color='#FF5722', ms=8, zorder=7)
    ax2.annotate("线段上距原点\n最近的点",
                 xy=closest2, xytext=(closest2[0] - 1.8, closest2[1] - 0.6),
                 fontsize=9, color='#FF5722',
                 arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.2))
    # 虚线连到原点
    ax2.plot([closest2[0], 0], [closest2[1], 0], '--', color='#FF5722', lw=1.0, alpha=0.6)

    check2 = np.dot(p2, d2)
    ax2.text(-4.5, -1.8,
             f"检查：$p_2 \\cdot d_2$ = {check2:.2f} > 0\n→ 继续，新方向从最近点指向原点",
             fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50'))

    # 迭代 3：新方向从最近点指向原点
    sa3, _, _ = support(A_verts, d3)
    sb3, _, _ = support(B_verts, -d3)
    p3 = sa3 - sb3

    ax3 = axes[2]
    base_ax(ax3, "检查终止条件：无法再靠近原点", 3)

    ax3.annotate("", xy=(0 + d3[0]*0.8, 0 + d3[1]*0.8),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.0))
    ax3.text(d3[0]*0.9, d3[1]*0.9 + 0.15,
             r"$\mathbf{d}_3$", fontsize=10, color=C_ARROW)

    # 旧点
    ax3.plot(*p1, 'D', color=C_SIMPLEX, ms=7, zorder=5, alpha=0.4)
    ax3.plot(*p2, 'D', color=C_SIMPLEX, ms=7, zorder=5, alpha=0.4)
    ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=C_SIMPLEX, lw=1.5, zorder=4, alpha=0.4)

    # 新点
    ax3.plot(*p3, 'D', color=C_SIMPLEX, ms=10, zorder=6)
    ax3.annotate(f"$p_3$=({p3[0]:.1f},{p3[1]:.1f})",
                 xy=p3, xytext=(p3[0] - 2.3, p3[1] + 0.5),
                 fontsize=9, color=C_SIMPLEX, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_SIMPLEX, lw=1.2))

    # 终止判断：p3·d3
    check3 = np.dot(p3, d3)
    color_check = '#F44336' if check3 <= 0 else '#4CAF50'
    terminate = check3 <= 0

    if terminate:
        msg = (f"终止！$p_3 \\cdot d_3$ = {check3:.2f} ≤ 0\n"
               "C 在方向 d₃ 上已无法更靠近原点\n"
               "→ 原点不在 C 内 → A 和 B 不相交")
        box_color = '#FFEBEE'
        edge_color = '#F44336'
    else:
        # 若未终止，找最近点继续
        closest3 = closest_on_segment(p2, p3)
        ax3.plot(*closest3, 's', color='#FF5722', ms=8, zorder=7)
        ax3.plot([closest3[0], 0], [closest3[1], 0], '--', color='#FF5722', lw=1.0, alpha=0.6)
        msg = (f"$p_3 \\cdot d_3$ = {check3:.2f} > 0\n继续迭代（通常 2-6 步收敛）")
        box_color = '#E8F5E9'
        edge_color = '#4CAF50'

    ax3.text(-4.5, -1.8, msg, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=box_color, edgecolor=edge_color))

    # 最终距离示意线（最近点到原点）
    closest_final = closest_on_segment(p2, p3) if not terminate else closest_on_segment(p1, p3)
    ax3.plot(*closest_final, 's', color='#FF5722', ms=8, zorder=7)
    ax3.plot([closest_final[0], 0], [closest_final[1], 0],
             '-', color='#FF5722', lw=2.0, zorder=6, alpha=0.8,
             label=f"最近距离 ≈ {np.linalg.norm(closest_final):.2f}")
    ax3.legend(fontsize=9, loc='upper left')

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {path}")


# ── 主程序 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    out = "/home/flageval/projects/paper/public"
    os.makedirs(out, exist_ok=True)

    plot_support_query(f"{out}/gjk_support_query.png")
    plot_minkowski_diff(f"{out}/gjk_minkowski_diff.png")
    plot_gjk_iteration(f"{out}/gjk_iteration.png")
