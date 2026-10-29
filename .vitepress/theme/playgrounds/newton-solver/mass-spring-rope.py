import math

# 两端固定的质点弹簧绳索仿真。
# 半隐式 Euler 积分，从水平直线松手后自然下垂到悬链线平衡态。

# ============ 参数 ============
n_pts = 12
total_len = 1.0
mass = 0.05
k_spring = 500.0
damping = 0.995
grav = -9.8
rest = total_len / (n_pts - 1)

# 初始：水平直线
px = [i * rest for i in range(n_pts)]
py = [0.0] * n_pts
vx = [0.0] * n_pts
vy = [0.0] * n_pts

dt = 0.0005
total_steps = 4000

frames = []

def dist(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    return math.sqrt(dx * dx + dy * dy)

for step in range(total_steps + 1):
    # 每 20 步存一帧
    if step % 20 == 0:
        objs = []
        # 弹簧线段
        for i in range(n_pts - 1):
            objs.append({"type": "segment",
                "from": [px[i], py[i] + 1.2], "to": [px[i + 1], py[i + 1] + 1.2],
                "stroke": "#1565C0", "width": 2})
        # 质点
        for i in range(n_pts):
            clr = "#D32F2F" if i == 0 or i == n_pts - 1 else "#1E88E5"
            r = 0.025 if i == 0 or i == n_pts - 1 else 0.018
            objs.append({"type": "circle", "x": px[i], "y": py[i] + 1.2,
                "radius": r, "fill": clr, "stroke": clr})
        # 固定端标记
        objs.append({"type": "segment", "from": [-0.02, 1.21], "to": [-0.02, 1.25],
            "stroke": "#795548", "width": 3})
        objs.append({"type": "segment", "from": [total_len + 0.02, 1.21],
            "to": [total_len + 0.02, 1.25], "stroke": "#795548", "width": 3})

        # 计算当前能量和最大拉伸率
        E_kin = 0.0
        E_pot = 0.0
        E_spring = 0.0
        max_stretch = 0.0
        for i in range(n_pts):
            E_kin += 0.5 * mass * (vx[i] * vx[i] + vy[i] * vy[i])
            E_pot += mass * (-grav) * (py[i] + 1.2)
        for i in range(n_pts - 1):
            d = dist(px[i], py[i], px[i + 1], py[i + 1])
            stretch_pct = (d - rest) / rest * 100
            if abs(stretch_pct) > abs(max_stretch):
                max_stretch = stretch_pct
            E_spring += 0.5 * k_spring * (d - rest) ** 2

        # 最低点
        min_y = min(py)

        frames.append({
            "objects": objs,
            "metrics": {
                "t": f"{step * dt:.3f} s",
                "最低点y": f"{min_y:.3f} m",
                "最大拉伸": f"{max_stretch:.1f}%",
                "动能": f"{E_kin:.4f}",
            },
        })

    if step == total_steps:
        break

    # ---- 物理步进 ----
    # 弹簧力
    fx = [0.0] * n_pts
    fy = [0.0] * n_pts
    for i in range(n_pts - 1):
        dx = px[i + 1] - px[i]
        dy = py[i + 1] - py[i]
        d = math.sqrt(dx * dx + dy * dy) + 1e-9
        nx, ny = dx / d, dy / d
        stretch = d - rest
        f = k_spring * stretch
        fx[i] += f * nx
        fy[i] += f * ny
        fx[i + 1] -= f * nx
        fy[i + 1] -= f * ny

    # 重力 + 阻尼 + 积分
    for i in range(n_pts):
        ax = fx[i] / mass
        ay = fy[i] / mass + grav
        vx[i] = (vx[i] + dt * ax) * damping
        vy[i] = (vy[i] + dt * ay) * damping
        px[i] += dt * vx[i]
        py[i] += dt * vy[i]

    # 固定两端
    px[0], py[0] = 0.0, 0.0
    vx[0], vy[0] = 0.0, 0.0
    px[n_pts - 1], py[n_pts - 1] = total_len, 0.0
    vx[n_pts - 1], vy[n_pts - 1] = 0.0, 0.0

print(f"frames: {len(frames)}")
print(f"final lowest y: {min(py):.4f}")

simulation = {
    "bounds": [-0.1, 0.55, 1.1, 1.3],
    "fps": 30,
    "loop": True,
    "frames": frames,
}
