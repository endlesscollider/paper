import math

# PBD（基于位置的动力学）绳索仿真。
# 展示距离约束投影如何让绳索保持自然长度，对比不同迭代次数的效果。

# ============ 参数 ============
n_pts = 20
total_len = 1.0
rest = total_len / (n_pts - 1)
grav = -9.8
dt_sim = 0.02
n_steps = 200
n_iters = 10

def run_pbd(iters, steps=200):
    """运行 PBD 仿真，返回每帧的位置列表"""
    px_list = [i * rest for i in range(n_pts)]
    py_list = [0.0] * n_pts
    vx_list = [0.0] * n_pts
    vy_list = [0.0] * n_pts
    inv_m = [1.0] * n_pts
    inv_m[0] = 0.0
    inv_m[-1] = 0.0

    history = [list(zip(px_list[:], py_list[:]))]

    for step in range(steps):
        # 预测
        for i in range(n_pts):
            vx_list[i] += 0.0  # no horizontal gravity
            vy_list[i] += dt_sim * grav
            if inv_m[i] == 0:
                vx_list[i] = 0.0
                vy_list[i] = 0.0
        pred_x = [px_list[i] + dt_sim * vx_list[i] for i in range(n_pts)]
        pred_y = [py_list[i] + dt_sim * vy_list[i] for i in range(n_pts)]

        # 约束投影
        for _ in range(iters):
            for i in range(n_pts - 1):
                j = i + 1
                w1, w2 = inv_m[i], inv_m[j]
                if w1 + w2 == 0:
                    continue
                dx = pred_x[j] - pred_x[i]
                dy = pred_y[j] - pred_y[i]
                d = math.sqrt(dx * dx + dy * dy) + 1e-9
                nx, ny = dx / d, dy / d
                diff = d - rest
                c = diff / (w1 + w2)
                pred_x[i] += w1 * c * nx
                pred_y[i] += w1 * c * ny
                pred_x[j] -= w2 * c * nx
                pred_y[j] -= w2 * c * ny

        # 反推速度 + 更新位置
        for i in range(n_pts):
            vx_list[i] = (pred_x[i] - px_list[i]) / dt_sim
            vy_list[i] = (pred_y[i] - py_list[i]) / dt_sim
            px_list[i] = pred_x[i]
            py_list[i] = pred_y[i]

        history.append(list(zip(px_list[:], py_list[:])))

    # 总长度误差
    actual = 0.0
    for i in range(n_pts - 1):
        dx = px_list[i + 1] - px_list[i]
        dy = py_list[i + 1] - py_list[i]
        actual += math.sqrt(dx * dx + dy * dy)
    err = abs(actual - total_len) / total_len * 100
    return history, err


# ============ 主仿真（迭代=10）============
history_main, err_main = run_pbd(n_iters, n_steps)

# 不同迭代次数对比用的最终形状
iter_counts = [1, 5, 10, 30]
iter_finals = {}
iter_errors = {}
for ni in iter_counts:
    h, e = run_pbd(ni, n_steps)
    iter_finals[ni] = h[-1]
    iter_errors[ni] = e

# ============ 构建帧 ============
# 阶段 1 (0~200)：实时动画（迭代=10）
# 阶段 2 (201~280)：叠加展示不同迭代次数的最终形状

frames = []
Y_OFF = 0.65  # 垂直偏移使绳索可见

C_MAIN = "#1565C0"
C_ITER = {1: "#EF5350", 5: "#FF9800", 10: "#1565C0", 30: "#7B1FA2"}

def draw_rope(objs, pts, clr, width=2, dot_r=0.012):
    for i in range(len(pts) - 1):
        objs.append({"type": "segment",
            "from": [pts[i][0], pts[i][1] + Y_OFF],
            "to": [pts[i + 1][0], pts[i + 1][1] + Y_OFF],
            "stroke": clr, "width": width})
    for i in range(len(pts)):
        fc = "#D32F2F" if i == 0 or i == len(pts) - 1 else clr
        objs.append({"type": "circle",
            "x": pts[i][0], "y": pts[i][1] + Y_OFF,
            "radius": dot_r, "fill": fc, "stroke": fc})

# 阶段 1：主仿真动画
for step in range(0, n_steps + 1, 1):
    pts = history_main[step]
    objs = []

    # 固定端
    objs.append({"type": "segment", "from": [-0.015, Y_OFF],
        "to": [-0.015, Y_OFF + 0.05], "stroke": "#795548", "width": 3})
    objs.append({"type": "segment", "from": [total_len + 0.015, Y_OFF],
        "to": [total_len + 0.015, Y_OFF + 0.05], "stroke": "#795548", "width": 3})

    draw_rope(objs, pts, C_MAIN)

    # 总长度
    actual = 0.0
    for i in range(n_pts - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        actual += math.sqrt(dx * dx + dy * dy)
    cur_err = abs(actual - total_len) / total_len * 100

    min_y = min(p[1] for p in pts)

    frames.append({
        "objects": objs,
        "metrics": {
            "阶段": f"仿真 {step}/{n_steps}",
            "迭代次数": f"{n_iters}",
            "长度误差": f"{cur_err:.1f}%",
            "最低点y": f"{min_y:.3f} m",
        },
    })

# 阶段 2：不同迭代次数对比（80帧保持）
for hold in range(80):
    objs = []
    objs.append({"type": "segment", "from": [-0.015, Y_OFF],
        "to": [-0.015, Y_OFF + 0.05], "stroke": "#795548", "width": 3})
    objs.append({"type": "segment", "from": [total_len + 0.015, Y_OFF],
        "to": [total_len + 0.015, Y_OFF + 0.05], "stroke": "#795548", "width": 3})

    for ni in iter_counts:
        pts = iter_finals[ni]
        draw_rope(objs, pts, C_ITER[ni], width=2, dot_r=0.010)

    # 用 iter=10 的误差作为对比文字
    metric_parts = []
    for ni in iter_counts:
        metric_parts.append(f"{ni}次={iter_errors[ni]:.1f}%")

    frames.append({
        "objects": objs,
        "metrics": {
            "阶段": "迭代次数对比",
            "红1/橙5/蓝10/紫30": "四种迭代",
            "长度误差": " | ".join(metric_parts),
        },
    })

print(f"frames: {len(frames)}")
print(f"objects/frame: ~{len(frames[100]['objects'])}")
print(f"error at iter=10: {err_main:.1f}%")

simulation = {
    "bounds": [-0.08, -0.15, 1.08, 0.8],
    "fps": 25,
    "loop": True,
    "frames": frames,
}
