import math

# 演示三维旋转的核心操作：绕 z 轴旋转坐标系 + 角速度积分轨迹。
# 左侧：坐标架从世界坐标系绕 z 轴逐渐旋转到 90°。
# 右侧：以 ω=(0,0,2) rad/s 积分，x 轴端点在单位圆上的轨迹。
# 二维投影（俯视图），用线段画坐标轴，用 segment 连线画轨迹。

# ============ 工具函数 ============

def rot_z(theta):
    """绕 z 轴旋转的 2D 投影（俯视图 = xy 平面）"""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def mv(M, v):
    return [M[0][0]*v[0] + M[0][1]*v[1], M[1][0]*v[0] + M[1][1]*v[1]]

# ============ 参数 ============

axis_len = 0.7
LX, LY = 0.0, 0.0    # 左坐标架中心
RX, RY = 2.5, 0.0     # 右轨迹中心
trail_radius = 0.7     # x 轴端点在单位圆上的投影半径

# ============ 预计算角速度积分轨迹 ============
# ω=(0,0,2) rad/s, dt=0.01, 200 steps → 总转角 = 4 rad ≈ 229°
# 俯视图中 x 轴端点 = (cos θ, sin θ) 在 xy 平面上画圆弧

omega_z = 2.0
sim_dt = 0.01
sim_steps = 200
trail_pts = []
for i in range(sim_steps + 1):
    angle = omega_z * sim_dt * i
    trail_pts.append([math.cos(angle), math.sin(angle)])

# ============ 动画帧 ============

frames = []
total = 240  # 8s at 30fps

# Phase 1 (0-90): 坐标架从 0° 旋转到 90°
# Phase 2 (90-120): 保持 90°
# Phase 3 (120-210): 角速度积分轨迹逐步绘制
# Phase 4 (210-240): 保持

def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

for frame in range(total):
    objs = []

    # ---- 左侧：旋转坐标架 ----

    if frame <= 90:
        angle = smooth(frame / 90.0) * math.pi / 2.0
    else:
        angle = math.pi / 2.0

    R = rot_z(angle)

    # 世界坐标系（淡灰）
    objs.append({"type": "segment", "from": [LX, LY],
        "to": [LX + axis_len, LY], "stroke": "#D0D0D0", "width": 2})
    objs.append({"type": "segment", "from": [LX, LY],
        "to": [LX, LY + axis_len], "stroke": "#D0D0D0", "width": 2})

    # 旋转后坐标系
    x_end = mv(R, [axis_len, 0])
    y_end = mv(R, [0, axis_len])
    objs.append({"type": "segment", "from": [LX, LY],
        "to": [LX + x_end[0], LY + x_end[1]], "stroke": "#D32F2F", "width": 3})
    objs.append({"type": "circle", "x": LX + x_end[0], "y": LY + x_end[1],
        "radius": 0.03, "fill": "#D32F2F", "stroke": "#D32F2F"})
    objs.append({"type": "segment", "from": [LX, LY],
        "to": [LX + y_end[0], LY + y_end[1]], "stroke": "#388E3C", "width": 3})
    objs.append({"type": "circle", "x": LX + y_end[0], "y": LY + y_end[1],
        "radius": 0.03, "fill": "#388E3C", "stroke": "#388E3C"})

    # 旋转弧线
    n_arc = max(2, int(abs(angle) / (math.pi / 2) * 20))
    for i in range(n_arc - 1):
        a1 = angle * i / (n_arc - 1)
        a2 = angle * (i + 1) / (n_arc - 1)
        r_arc = 0.35
        objs.append({"type": "segment",
            "from": [LX + r_arc * math.cos(a1), LY + r_arc * math.sin(a1)],
            "to": [LX + r_arc * math.cos(a2), LY + r_arc * math.sin(a2)],
            "stroke": "#FF6F00", "width": 2})

    # 原点
    objs.append({"type": "circle", "x": LX, "y": LY,
        "radius": 0.035, "fill": "#333", "stroke": "#333"})

    # ---- 右侧：角速度积分轨迹 ----

    # 单位圆参考线（淡灰）
    n_circ = 40
    for i in range(n_circ):
        a1 = 2 * math.pi * i / n_circ
        a2 = 2 * math.pi * (i + 1) / n_circ
        objs.append({"type": "segment",
            "from": [RX + trail_radius * math.cos(a1), RY + trail_radius * math.sin(a1)],
            "to": [RX + trail_radius * math.cos(a2), RY + trail_radius * math.sin(a2)],
            "stroke": "#E0E0E0", "width": 1})

    # 世界坐标系轴（淡灰）
    objs.append({"type": "segment", "from": [RX, RY],
        "to": [RX + trail_radius * 1.15, RY], "stroke": "#D0D0D0", "width": 1})
    objs.append({"type": "segment", "from": [RX, RY],
        "to": [RX, RY + trail_radius * 1.15], "stroke": "#D0D0D0", "width": 1})

    if frame >= 120:
        # 逐步绘制轨迹
        progress = smooth((frame - 120) / 90.0) if frame <= 210 else 1.0
        n_pts = max(1, int(progress * sim_steps))

        # 轨迹线（红色）
        for i in range(min(n_pts, len(trail_pts) - 1)):
            p1 = trail_pts[i]
            p2 = trail_pts[i + 1]
            objs.append({"type": "segment",
                "from": [RX + trail_radius * p1[0], RY + trail_radius * p1[1]],
                "to": [RX + trail_radius * p2[0], RY + trail_radius * p2[1]],
                "stroke": "#D32F2F", "width": 2})

        # 起点（绿）
        objs.append({"type": "circle",
            "x": RX + trail_radius * trail_pts[0][0],
            "y": RY + trail_radius * trail_pts[0][1],
            "radius": 0.04, "fill": "#388E3C", "stroke": "#388E3C"})

        # 当前端点（红）
        if n_pts < len(trail_pts):
            cp = trail_pts[n_pts]
        else:
            cp = trail_pts[-1]
        objs.append({"type": "circle",
            "x": RX + trail_radius * cp[0], "y": RY + trail_radius * cp[1],
            "radius": 0.04, "fill": "#D32F2F", "stroke": "#D32F2F"})

        # 当前旋转后的坐标架
        cur_angle = omega_z * sim_dt * n_pts
        Rc = rot_z(cur_angle)
        cx_end = mv(Rc, [trail_radius * 0.85, 0])
        cy_end = mv(Rc, [0, trail_radius * 0.85])
        objs.append({"type": "segment", "from": [RX, RY],
            "to": [RX + cx_end[0], RY + cx_end[1]], "stroke": "#D32F2F", "width": 2})
        objs.append({"type": "segment", "from": [RX, RY],
            "to": [RX + cy_end[0], RY + cy_end[1]], "stroke": "#388E3C", "width": 2})

        total_deg = math.degrees(cur_angle)
    else:
        total_deg = 0.0
        # 初始坐标架
        objs.append({"type": "segment", "from": [RX, RY],
            "to": [RX + trail_radius * 0.85, RY], "stroke": "#D32F2F", "width": 2})
        objs.append({"type": "segment", "from": [RX, RY],
            "to": [RX, RY + trail_radius * 0.85], "stroke": "#388E3C", "width": 2})

    # 右侧原点
    objs.append({"type": "circle", "x": RX, "y": RY,
        "radius": 0.03, "fill": "#333", "stroke": "#333"})

    # 分隔线
    objs.append({"type": "segment", "from": [1.25, -1.1], "to": [1.25, 1.1],
        "stroke": "#E0E0E0", "width": 1})

    # 四元数计算
    half = angle / 2.0
    qw = math.cos(half)
    qz = math.sin(half)

    if frame <= 90:
        phase = "① 绕z轴旋转"
    elif frame <= 120:
        phase = "② 保持90°"
    elif frame <= 210:
        phase = "③ 角速度积分"
    else:
        phase = "④ 积分完成"

    frames.append({
        "objects": objs,
        "metrics": {
            "阶段": phase,
            "转角": f"{math.degrees(angle):.1f}°",
            "q(w,z)": f"({qw:.3f}, {qz:.3f})",
            "ω": "(0, 0, 2) rad/s",
            "积分转角": f"{total_deg:.1f}°",
        },
    })

print(f"frames: {len(frames)}")
print(f"objects/frame (mid): ~{len(frames[150]['objects'])}")

simulation = {
    "bounds": [-1.0, -1.1, 3.8, 1.1],
    "fps": 30,
    "loop": True,
    "frames": frames,
}
