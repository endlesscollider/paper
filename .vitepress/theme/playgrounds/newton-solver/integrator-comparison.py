import math

# 三种数值积分方法（显式 Euler / 半隐式 Euler / Velocity Verlet）
# 同时仿真简谐振子，用相空间轨迹和弹簧动画展示能量稳定性差异。

k, m = 1.0, 1.0
omega = math.sqrt(k / m)

def accel(x):
    return -k / m * x

def explicit_euler(x, v, dt):
    a = accel(x)
    return x + dt * v, v + dt * a

def symplectic_euler(x, v, dt):
    a = accel(x)
    v1 = v + dt * a
    x1 = x + dt * v1
    return x1, v1

def velocity_verlet(x, v, dt):
    a0 = accel(x)
    x1 = x + dt * v + 0.5 * dt * dt * a0
    a1 = accel(x1)
    v1 = v + 0.5 * dt * (a0 + a1)
    return x1, v1

# ============ 仿真 ============
dt = 0.3
n_steps = 80

def simulate(step_fn):
    x, v = 1.0, 0.0
    traj = [(x, v)]
    for _ in range(n_steps):
        x, v = step_fn(x, v, dt)
        traj.append((x, v))
    return traj

traj_exp = simulate(explicit_euler)
traj_sym = simulate(symplectic_euler)
traj_ver = simulate(velocity_verlet)

# ============ 构建动画帧 ============
# 左侧：三个质点弹簧实时动画（竖排）
# 右侧：相空间轨迹（用历史轨迹线段画出）

frames = []

# 三条轨迹的颜色
C_EXP = "#E53935"
C_SYM = "#1E88E5"
C_VER = "#43A047"

# 布局常量
# 弹簧动画区域 x in [-1, 3]
# 每个弹簧：固定端在 x=0，质点在 x = 1 + position
SPRING_Y = [3.0, 1.8, 0.6]  # 三条弹簧的 y 坐标
WALL_X = -0.3

# 相空间区域：用轨迹段叠加
# 用弹簧质点位置动画 + 度量面板替代相空间图

for step in range(n_steps + 1):
    xe, ve = traj_exp[step]
    xs, vs = traj_sym[step]
    xv, vv = traj_ver[step]

    Ee = 0.5 * k * xe * xe + 0.5 * m * ve * ve
    Es = 0.5 * k * xs * xs + 0.5 * m * vs * vs
    Ev = 0.5 * k * xv * xv + 0.5 * m * vv * vv

    objs = []

    # 精确解参考
    t = step * dt
    x_exact = math.cos(omega * t)

    data = [
        (xe, C_EXP, SPRING_Y[0], "显式Euler"),
        (xs, C_SYM, SPRING_Y[1], "半隐式Euler"),
        (xv, C_VER, SPRING_Y[2], "Verlet"),
    ]

    for xpos, clr, sy, label in data:
        # 固定墙壁
        objs.append({"type": "segment", "from": [WALL_X, sy - 0.3],
                      "to": [WALL_X, sy + 0.3], "stroke": "#795548", "width": 3})
        # 弹簧线（从墙到质点）
        ball_x = 1.0 + xpos
        # 画弹簧为之字形线段
        n_coils = 8
        coil_pts = [[WALL_X, sy]]
        seg_len = (ball_x - WALL_X) / (n_coils * 2 + 1)
        for ci in range(n_coils):
            cx = WALL_X + (2 * ci + 1) * seg_len
            coil_pts.append([cx, sy + 0.15])
            coil_pts.append([cx + seg_len, sy - 0.15])
        coil_pts.append([ball_x, sy])
        for pi in range(len(coil_pts) - 1):
            objs.append({"type": "segment",
                "from": coil_pts[pi], "to": coil_pts[pi + 1],
                "stroke": "#9E9E9E", "width": 1})
        # 质点球
        objs.append({"type": "circle", "x": ball_x, "y": sy,
                      "radius": 0.18, "fill": clr, "stroke": clr})
        # 精确解参考线（虚线用小圆点表示）
        ref_x = 1.0 + x_exact
        objs.append({"type": "circle", "x": ref_x, "y": sy,
                      "radius": 0.06, "fill": "#BDBDBD", "stroke": "#BDBDBD"})

    # 相空间轨迹：画到当前步为止的线段
    # 限制显式 Euler 轨迹范围避免画面太大
    trajs = [(traj_exp, C_EXP), (traj_sym, C_SYM), (traj_ver, C_VER)]
    # 映射到右侧区域：x_phase in [3.5, 6.5], y_phase in [0, 3.6]
    # 相空间坐标 (x, v) → screen: px = 5.0 + x * 1.0, py = 1.8 + v * 1.0
    PX0, PY0 = 5.0, 1.8
    PS = 0.9  # 缩放

    # 精确解椭圆（参考）
    n_circ = 60
    for ci in range(n_circ):
        a1 = 2.0 * math.pi * ci / n_circ
        a2 = 2.0 * math.pi * (ci + 1) / n_circ
        objs.append({"type": "segment",
            "from": [PX0 + math.cos(a1) * PS, PY0 - math.sin(a1) * omega * PS],
            "to": [PX0 + math.cos(a2) * PS, PY0 - math.sin(a2) * omega * PS],
            "stroke": "#E0E0E0", "width": 1})

    # 画已走过的轨迹
    for traj, clr in trajs:
        end = min(step, len(traj) - 1)
        # 只画最近 40 步避免太密
        start = max(0, end - 40)
        for si in range(start, end):
            x1, v1 = traj[si]
            x2, v2 = traj[si + 1]
            # 截断极端值
            if abs(x1) > 4 or abs(v1) > 4 or abs(x2) > 4 or abs(v2) > 4:
                continue
            objs.append({"type": "segment",
                "from": [PX0 + x1 * PS, PY0 + v1 * PS],
                "to": [PX0 + x2 * PS, PY0 + v2 * PS],
                "stroke": clr, "width": 1.5})
        # 当前点
        xc, vc = traj[end]
        if abs(xc) <= 4 and abs(vc) <= 4:
            objs.append({"type": "circle",
                "x": PX0 + xc * PS, "y": PY0 + vc * PS,
                "radius": 0.06, "fill": clr, "stroke": clr})

    # 坐标轴
    objs.append({"type": "segment", "from": [PX0 - 1.5, PY0],
                  "to": [PX0 + 1.5, PY0], "stroke": "#BDBDBD", "width": 1})
    objs.append({"type": "segment", "from": [PX0, PY0 - 1.5],
                  "to": [PX0, PY0 + 1.5], "stroke": "#BDBDBD", "width": 1})

    # 分隔线
    objs.append({"type": "segment", "from": [3.3, -0.2], "to": [3.3, 3.8],
                  "stroke": "#E0E0E0", "width": 1})

    frames.append({
        "objects": objs,
        "metrics": {
            "步": f"{step}/{n_steps}",
            "E_显式": f"{Ee:.3f}",
            "E_辛": f"{Es:.3f}",
            "E_Verlet": f"{Ev:.3f}",
            "精确E": "0.500",
        },
    })

print(f"frames: {len(frames)}")
print(f"objects/frame: ~{len(frames[40]['objects'])}")

simulation = {
    "bounds": [-0.8, -0.2, 6.8, 3.8],
    "fps": 8,
    "loop": True,
    "frames": frames,
}
