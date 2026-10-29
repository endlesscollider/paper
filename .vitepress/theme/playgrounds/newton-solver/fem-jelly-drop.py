# 一块 5×5 网格的果冻从空中掉落，撞到地面后被压扁，弹起来。
# 整个仿真只用本文讲的四个步骤：形函数梯度 → F → 应力 → 顶点力。
import math

# ============ 网格：5×5 顶点，每个方格切成两个三角形 ============
n = 5
verts = [[i / (n - 1), j / (n - 1)] for j in range(n) for i in range(n)]
elems = []
for j in range(n - 1):
    for i in range(n - 1):
        a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
        elems.append([a, b, d])
        elems.append([a, d, c])

# 参考构型 = 未变形的方形网格
ref = [v[:] for v in verts]

# 初始位置：整体抬高 0.5
pos = [[v[0], v[1] + 0.5] for v in ref]
vel = [[0.0, 0.0] for _ in ref]

# ============ 预计算形函数梯度（只依赖参考构型，算一次就行） ============
grad_Ns = []  # 每个三角形三个形函数的梯度
areas = []    # 每个三角形的面积

for e in elems:
    x0, x1, x2 = ref[e[0]], ref[e[1]], ref[e[2]]
    area2 = (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x2[0] - x0[0]) * (x1[1] - x0[1])
    gN = [
        [(x1[1] - x2[1]) / area2, (x2[0] - x1[0]) / area2],
        [(x2[1] - x0[1]) / area2, (x0[0] - x2[0]) / area2],
        [(x0[1] - x1[1]) / area2, (x1[0] - x0[0]) / area2],
    ]
    grad_Ns.append(gN)
    areas.append(abs(area2) / 2)

# ============ 物理参数——可以直接改！ ============
mu = 400.0        # 弹性模量（调大果冻变硬，调小变软）
lam = 600.0       # 第二弹性模量
mass = 0.4        # 每个顶点的质量
dt = 0.0003       # 时间步长
damping = 0.06    # 阻尼系数（越大停得越快）

# ============ 仿真主循环 ============
frames = []
total_steps = 3000

for step in range(total_steps + 1):
    # --- 每 12 步存一帧给播放器 ---
    if step % 12 == 0:
        objects = [{"type": "segment", "from": [-0.3, 0], "to": [1.3, 0],
                    "stroke": "#5D4037", "width": 3}]
        for e in elems:
            objects.append({
                "type": "polygon",
                "points": [pos[e[0]][:], pos[e[1]][:], pos[e[2]][:]],
                "fill": "#BBDEFB", "stroke": "#1565C0", "width": 1,
            })
        for p in pos:
            objects.append({"type": "circle", "x": p[0], "y": p[1],
                            "radius": 0.018, "fill": "#0D47A1"})
        ys = [p[1] for p in pos]
        frames.append({
            "objects": objects,
            "metrics": {"t": f"{step * dt * 1000:.0f} ms",
                        "height": f"{max(ys) - min(ys):.3f}",
                        "y_min": f"{min(ys):.3f}"},
        })

    if step == total_steps:
        break

    # --- 第 1 步：算所有三角形的弹性力 ---
    forces = [[0.0, 0.0] for _ in pos]
    for ei in range(len(elems)):
        e = elems[ei]
        gN = grad_Ns[ei]
        area = areas[ei]
        p0, p1, p2 = pos[e[0]], pos[e[1]], pos[e[2]]

        # F = sum_i x_i ⊗ grad_N_i  (2×2 矩阵，用四个分量表示)
        F00 = p0[0]*gN[0][0] + p1[0]*gN[1][0] + p2[0]*gN[2][0]
        F01 = p0[0]*gN[0][1] + p1[0]*gN[1][1] + p2[0]*gN[2][1]
        F10 = p0[1]*gN[0][0] + p1[1]*gN[1][0] + p2[1]*gN[2][0]
        F11 = p0[1]*gN[0][1] + p1[1]*gN[1][1] + p2[1]*gN[2][1]

        # det(F) = 体积变化比
        det = F00 * F11 - F01 * F10
        J = max(det, 0.05)

        # F 逆转置（2×2 显式公式）
        inv_det = 1.0 / det if abs(det) > 1e-12 else 0.0
        Fit00, Fit01 = F11 * inv_det, -F10 * inv_det
        Fit10, Fit11 = -F01 * inv_det, F00 * inv_det

        # Neo-Hookean PK1 应力: P = mu*(F - Fit) + lam*(J-1)*J*Fit
        c = lam * (J - 1) * J
        P00 = mu * (F00 - Fit00) + c * Fit00
        P01 = mu * (F01 - Fit01) + c * Fit01
        P10 = mu * (F10 - Fit10) + c * Fit10
        P11 = mu * (F11 - Fit11) + c * Fit11

        # 节点力 f_i = -area * P @ grad_N_i，并累加到全局（组装）
        for i in range(3):
            fx = -area * (P00 * gN[i][0] + P01 * gN[i][1])
            fy = -area * (P10 * gN[i][0] + P11 * gN[i][1])
            forces[e[i]][0] += fx
            forces[e[i]][1] += fy

    # --- 第 2 步：加重力和阻尼，更新速度和位置 ---
    for i in range(len(pos)):
        forces[i][1] += mass * (-9.8)
        forces[i][0] -= damping * vel[i][0] * mass
        forces[i][1] -= damping * vel[i][1] * mass

        vel[i][0] += dt * forces[i][0] / mass
        vel[i][1] += dt * forces[i][1] / mass
        pos[i][0] += dt * vel[i][0]
        pos[i][1] += dt * vel[i][1]

        # 地面碰撞
        if pos[i][1] < 0:
            pos[i][1] = 0
            if vel[i][1] < 0:
                vel[i][1] *= -0.15

ys = [p[1] for p in pos]
print(f"final height range: [{min(ys):.3f}, {max(ys):.3f}]")
print(f"total frames: {len(frames)}")

# PhysicsPlayground 重放预计算好的帧。
simulation = {"bounds": [-0.3, -0.15, 1.3, 1.8], "fps": 50, "loop": True, "frames": frames}
