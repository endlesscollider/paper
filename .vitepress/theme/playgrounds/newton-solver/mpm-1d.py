import math

# 1D MPM 演示：5 个粒子在重力下自由下落。
# 展示完整的 P2G → 网格计算 → G2P 循环。
# 粒子用圆表示，网格节点用小方块表示，P2G 权重用连线粗细可视化。

def bspline_w(dx, h):
    """二次 B-spline 权重函数"""
    x = abs(dx / h)
    if x < 0.5:
        return 0.75 - x * x
    elif x < 1.5:
        return 0.5 * (1.5 - x) ** 2
    return 0.0

def mpm_step(px, pv, pm, gx, dt, grav, h):
    """一步 1D MPM: P2G → 网格求解(重力) → G2P"""
    ng = len(gx)
    g_mass = [0.0] * ng
    g_mom = [0.0] * ng

    for xp, vp, mp in zip(px, pv, pm):
        for i in range(ng):
            w = bspline_w(xp - gx[i], h)
            if w > 0:
                g_mass[i] += w * mp
                g_mom[i] += w * mp * vp

    g_v = [0.0] * ng
    g_v_new = [0.0] * ng
    for i in range(ng):
        if g_mass[i] > 1e-9:
            g_v[i] = g_mom[i] / g_mass[i]
            g_v_new[i] = g_v[i] + dt * grav

    new_x = []
    new_v = []
    for xp in px:
        vn = 0.0
        for i in range(ng):
            w = bspline_w(xp - gx[i], h)
            vn += w * g_v_new[i]
        new_x.append(xp + dt * vn)
        new_v.append(vn)

    return new_x, new_v, g_mass, g_v_new

# ============ 仿真设置 ============

n_particles = 5
px = [2.1, 2.4, 2.7, 3.0, 3.3]
pv = [0.0] * n_particles
pm = [0.1] * n_particles
gx = [float(i) for i in range(7)]
h = 1.0
dt = 0.02
grav = -2.0
n_steps = 60

p_colors = ["#D32F2F", "#E65100", "#1976D2", "#388E3C", "#7B1FA2"]

# 可视化参数: 把 1D 位置映射到 2D Canvas
# x 轴 = 粒子编号方向（水平展开），y 轴 = 1D 物理位置
# 网格节点显示在左侧，粒子在右侧
VIEW_X_LEFT = 0.3   # 网格节点 x 位置
VIEW_X_RIGHT = 2.5  # 粒子起始 x 位置
VIEW_X_SPAN = 2.0   # 粒子水平分散宽度

frames = []

for step in range(n_steps + 1):
    objs = []

    # 网格节点（左侧竖排）
    for i, xi in enumerate(gx):
        objs.append({"type": "polygon",
            "points": [[VIEW_X_LEFT - 0.08, xi - 0.08],
                       [VIEW_X_LEFT + 0.08, xi - 0.08],
                       [VIEW_X_LEFT + 0.08, xi + 0.08],
                       [VIEW_X_LEFT - 0.08, xi + 0.08]],
            "fill": "#E0E0E0", "stroke": "#9E9E9E", "width": 1})

    # 网格线（竖排参考线）
    objs.append({"type": "segment",
        "from": [VIEW_X_LEFT, gx[0] - 0.3],
        "to": [VIEW_X_LEFT, gx[-1] + 0.3],
        "stroke": "#E0E0E0", "width": 1})

    # 粒子 + P2G 权重连线
    for p in range(n_particles):
        p_x_view = VIEW_X_RIGHT + p * (VIEW_X_SPAN / (n_particles - 1))
        p_y = px[p]

        # P2G 权重连线
        for i, xi in enumerate(gx):
            w = bspline_w(px[p] - xi, h)
            if w > 0.01:
                lw = max(0.5, w * 4)
                objs.append({"type": "segment",
                    "from": [p_x_view, p_y],
                    "to": [VIEW_X_LEFT, xi],
                    "stroke": p_colors[p], "width": lw})

        # 粒子圆
        objs.append({"type": "circle",
            "x": p_x_view, "y": p_y, "radius": 0.12,
            "fill": p_colors[p], "stroke": p_colors[p]})

    # 地面参考线
    objs.append({"type": "segment",
        "from": [0.0, 0.0], "to": [5.0, 0.0],
        "stroke": "#5D4037", "width": 2})

    avg_y = sum(px) / n_particles
    max_v = max(abs(v) for v in pv) if any(abs(v) > 0 for v in pv) else 0

    frames.append({
        "objects": objs,
        "metrics": {
            "步": f"{step}/{n_steps}",
            "t": f"{step * dt:.2f}s",
            "平均位置": f"{avg_y:.3f}",
            "最大速度": f"{max_v:.3f}",
        },
    })

    if step < n_steps:
        px, pv, _, _ = mpm_step(px, pv, pm, gx, dt, grav, h)
        # 简单地面碰撞
        for i in range(n_particles):
            if px[i] < 0.1:
                px[i] = 0.1
                if pv[i] < 0:
                    pv[i] *= -0.2

# 停留
for _ in range(10):
    frames.append(frames[-1])

print(f"frames: {len(frames)}")
print(f"final positions: {[f'{x:.3f}' for x in px]}")

simulation = {
    "bounds": [-0.2, -0.5, 5.2, 6.5],
    "fps": 8,
    "loop": True,
    "frames": frames,
}
