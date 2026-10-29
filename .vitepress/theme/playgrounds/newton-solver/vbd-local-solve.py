import math

# VBD 逐顶点局部牛顿法演示：一个中心顶点被四个固定邻居通过弹簧连接。
# 动画展示牛顿迭代中顶点的运动轨迹和能量收敛。
# 简化为 2D 以适配 Canvas 可视化。

def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy) + 1e-12

def vbd_local_solve_2d(xi, neighbors, rest_lengths, weights, x_tilde, mass_i, dt, n_iter):
    """
    VBD 局部牛顿法求解单顶点最优位置 (2D 版本)。
    固定所有邻居位置，只优化 xi 的 2 个坐标。
    """
    inertia = mass_i / (dt * dt)
    x = xi[:]
    trajectory = [x[:]]
    energies = []

    for _ in range(n_iter + 1):
        E = 0.5 * inertia * ((x[0] - x_tilde[0]) ** 2 + (x[1] - x_tilde[1]) ** 2)
        for xj, L0, w in zip(neighbors, rest_lengths, weights):
            d = dist2(x, xj)
            E += 0.5 * w * (d - L0) ** 2
        energies.append(E)

        if len(energies) > n_iter:
            break

        gx = inertia * (x[0] - x_tilde[0])
        gy = inertia * (x[1] - x_tilde[1])
        H00 = inertia
        H11 = inertia
        H01 = 0.0

        for xj, L0, w in zip(neighbors, rest_lengths, weights):
            dx = x[0] - xj[0]
            dy = x[1] - xj[1]
            d = dist2(x, xj)
            nx, ny = dx / d, dy / d
            stretch = d - L0
            gx += w * stretch * nx
            gy += w * stretch * ny
            H00 += w * nx * nx
            H11 += w * ny * ny
            H01 += w * nx * ny

        det = H00 * H11 - H01 * H01
        if abs(det) < 1e-15:
            break
        sx = -(H11 * gx - H01 * gy) / det
        sy = -(-H01 * gx + H00 * gy) / det
        x = [x[0] + sx, x[1] + sy]
        trajectory.append(x[:])

    return x, trajectory, energies

# ============ 场景设置 ============

xi_init = [0.3, 0.7]
neighbors = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
rest_lengths = [0.6, 0.6, 0.6, 0.6]
weights = [500.0, 500.0, 500.0, 500.0]
x_tilde = xi_init[:]
mass_i = 0.05
dt = 0.01
n_newton = 8

x_final, traj, energies = vbd_local_solve_2d(
    xi_init, neighbors, rest_lengths, weights, x_tilde, mass_i, dt, n_newton)

# ============ 构建动画帧 ============

frames = []
E_max = energies[0]
E_min = energies[-1]
E_range = max(E_max - E_min, 1e-6)

bar_base_y = -0.3
bar_max_h = 0.5

for step in range(len(traj)):
    x_cur = traj[step]
    E = energies[min(step, len(energies) - 1)]
    objs = []

    # 邻居（固定，蓝色）
    for j, nb in enumerate(neighbors):
        objs.append({"type": "circle", "x": nb[0], "y": nb[1],
            "radius": 0.05, "fill": "#1976D2", "stroke": "#0D47A1"})

    # 弹簧连线（到当前位置）
    for nb in neighbors:
        objs.append({"type": "segment",
            "from": x_cur[:], "to": nb[:],
            "stroke": "#B0BEC5", "width": 1})

    # rest_length 参考圆（用小圆弧段近似）
    for nb in neighbors:
        L0 = 0.6
        n_seg = 36
        for s in range(n_seg):
            a1 = 2 * math.pi * s / n_seg
            a2 = 2 * math.pi * (s + 1) / n_seg
            objs.append({"type": "segment",
                "from": [nb[0] + L0 * math.cos(a1), nb[1] + L0 * math.sin(a1)],
                "to": [nb[0] + L0 * math.cos(a2), nb[1] + L0 * math.sin(a2)],
                "stroke": "#E3F2FD", "width": 1})

    # 迭代轨迹（已走过的部分）
    for k in range(step):
        objs.append({"type": "segment",
            "from": traj[k][:], "to": traj[k + 1][:],
            "stroke": "#E65100", "width": 2})
        objs.append({"type": "circle", "x": traj[k][0], "y": traj[k][1],
            "radius": 0.025, "fill": "#E65100", "stroke": "#E65100"})

    # 当前顶点位置
    if step == 0:
        color = "#D32F2F"
        r = 0.06
    elif step == len(traj) - 1:
        color = "#2E7D32"
        r = 0.06
    else:
        color = "#E65100"
        r = 0.045
    objs.append({"type": "circle", "x": x_cur[0], "y": x_cur[1],
        "radius": r, "fill": color, "stroke": color})

    # 能量柱
    frac = (E - E_min) / E_range
    h = bar_max_h * frac
    bar_x = 1.4
    bar_w = 0.12
    objs.append({"type": "polygon",
        "points": [[bar_x, bar_base_y], [bar_x + bar_w, bar_base_y],
                   [bar_x + bar_w, bar_base_y + h], [bar_x, bar_base_y + h]],
        "fill": "#FFE0B2", "stroke": "#E65100", "width": 1})
    objs.append({"type": "segment",
        "from": [bar_x - 0.03, bar_base_y],
        "to": [bar_x + bar_w + 0.03, bar_base_y],
        "stroke": "#888", "width": 1})

    # 到各邻居距离
    dists = [dist2(x_cur, nb) for nb in neighbors]
    avg_dist = sum(dists) / len(dists)

    frames.append({
        "objects": objs,
        "metrics": {
            "牛顿步": f"{step}/{n_newton}",
            "能量": f"{E:.1f}",
            "平均距离": f"{avg_dist:.4f}",
            "目标距离": "0.600",
        },
    })

# 多停留
for _ in range(10):
    frames.append(frames[-1])

print(f"frames: {len(frames)}")
print(f"energy: {energies[0]:.1f} -> {energies[-1]:.1f}")
print(f"final dists: {[dist2(x_final, nb) for nb in neighbors]}")

simulation = {
    "bounds": [-0.4, -0.5, 1.7, 1.4],
    "fps": 2,
    "loop": True,
    "frames": frames,
}
