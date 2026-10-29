import math

# 三角形弹簧网络的 Projective Dynamics 局部-全局迭代。
# 三个顶点被弹簧连接，初始位置偏离等边目标形状。
# 动画展示逐轮迭代收敛过程 + 能量和边长的实时变化。

# ============ 2x2 线性系统求解（纯 Python） ============

def solve2x2(A, b):
    """解 2x2 线性系统 Ax = b"""
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-15:
        return [0.0, 0.0]
    return [(A[1][1] * b[0] - A[0][1] * b[1]) / det,
            (A[0][0] * b[1] - A[1][0] * b[0]) / det]

def solve_system(A, b, n):
    """高斯消元求解 n 维线性系统"""
    Ab = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]
    for col in range(n):
        best = col
        for row in range(col + 1, n):
            if abs(Ab[row][col]) > abs(Ab[best][col]):
                best = row
        Ab[col], Ab[best] = Ab[best], Ab[col]
        if abs(Ab[col][col]) < 1e-15:
            continue
        for row in range(col + 1, n):
            factor = Ab[row][col] / Ab[col][col]
            for j in range(col, n + 1):
                Ab[row][j] -= factor * Ab[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = Ab[i][n]
        for j in range(i + 1, n):
            s -= Ab[i][j] * x[j]
        x[i] = s / Ab[i][i] if abs(Ab[i][i]) > 1e-15 else 0.0
    return x

# ============ Projective Dynamics 核心 ============

def local_step(x, edges, rest_lengths):
    """局部投影：每根弹簧独立计算心愿边向量"""
    projs = []
    for (i, j), L0 in zip(edges, rest_lengths):
        dx = x[j][0] - x[i][0]
        dy = x[j][1] - x[i][1]
        dist = math.sqrt(dx * dx + dy * dy) + 1e-12
        projs.append([dx / dist * L0, dy / dist * L0])
    return projs

def build_global_matrix(n_verts, edges, weights, mass, dt):
    """构建全局线性系统矩阵 (2n x 2n)"""
    nd = n_verts * 2
    A = [[0.0] * nd for _ in range(nd)]
    inertia = mass / (dt * dt)
    for i in range(nd):
        A[i][i] = inertia
    for (vi, vj), w in zip(edges, weights):
        for a, b, s in [(vi, vi, 1), (vj, vj, 1), (vi, vj, -1), (vj, vi, -1)]:
            A[2 * a][2 * b] += s * w
            A[2 * a + 1][2 * b + 1] += s * w
    return A

def pd_step(x0, v0, edges, rest_lengths, weights, mass, dt, gravity, n_iters):
    """一次仿真步：预测 → 若干轮局部-全局迭代 → 返回历史"""
    nv = len(x0)
    nd = nv * 2
    inertia = mass / (dt * dt)

    x_flat = []
    for p in x0:
        x_flat.extend(p)

    x_tilde = []
    for i in range(nv):
        x_tilde.append(x_flat[2 * i] + dt * (v0[i][0] + dt * gravity[0]))
        x_tilde.append(x_flat[2 * i + 1] + dt * (v0[i][1] + dt * gravity[1]))

    A = build_global_matrix(nv, edges, weights, mass, dt)

    x_new = x_tilde[:]
    history = []
    energies = []
    edge_lengths_hist = []

    for it in range(n_iters + 1):
        pts = [[x_new[2 * i], x_new[2 * i + 1]] for i in range(nv)]
        history.append(pts)

        E_inertia = 0.0
        for k in range(nd):
            E_inertia += 0.5 * inertia * (x_new[k] - x_tilde[k]) ** 2
        E_elastic = 0.0
        elens = []
        for (vi, vj), L0, w in zip(edges, rest_lengths, weights):
            dx = pts[vj][0] - pts[vi][0]
            dy = pts[vj][1] - pts[vi][1]
            d = math.sqrt(dx * dx + dy * dy)
            E_elastic += 0.5 * w * (d - L0) ** 2
            elens.append(d)
        energies.append(E_inertia + E_elastic)
        edge_lengths_hist.append(elens)

        if it == n_iters:
            break

        projs = local_step(pts, edges, rest_lengths)

        b = [inertia * x_tilde[k] for k in range(nd)]
        for idx, ((vi, vj), w) in enumerate(zip(edges, weights)):
            px, py = projs[idx]
            b[2 * vi] -= w * px
            b[2 * vi + 1] -= w * py
            b[2 * vj] += w * px
            b[2 * vj + 1] += w * py

        x_new = solve_system(A, b, nd)

    return history, energies, edge_lengths_hist

# ============ 仿真设置 ============

x0 = [[0.0, 0.0], [1.3, 0.3], [0.6, 1.4]]
v0 = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
edges = [(0, 1), (1, 2), (0, 2)]
rest_lengths = [1.0, 1.0, 1.0]
weights = [2000.0, 2000.0, 2000.0]
mass = 0.1
dt = 0.01
gravity = [0.0, -9.8]
n_iters = 15

history, energies, edge_lengths_hist = pd_step(
    x0, v0, edges, rest_lengths, weights, mass, dt, gravity, n_iters)

# ============ 构建动画帧 ============

frames = []
edge_colors = ["#D32F2F", "#388E3C", "#1976D2"]

E_max = energies[0]
E_min = energies[-1]
E_range = max(E_max - E_min, 1e-6)

# 能量柱状图位置参数
bar_base_y = -0.6
bar_height = 0.6

for it in range(n_iters + 1):
    pts = history[it]
    elens = edge_lengths_hist[it]
    E = energies[it]

    objs = []

    # 三角形网格边
    for idx, (vi, vj) in enumerate(edges):
        objs.append({"type": "segment",
            "from": pts[vi][:], "to": pts[vj][:],
            "stroke": edge_colors[idx], "width": 2.5})

    # 顶点
    for i, p in enumerate(pts):
        alpha_color = "#1565C0" if it > 0 else "#B0BEC5"
        objs.append({"type": "circle", "x": p[0], "y": p[1],
            "radius": 0.06, "fill": alpha_color, "stroke": "#0D47A1"})

    # 初始三角形（淡灰参考）
    if it > 0:
        p0 = history[0]
        for vi, vj in edges:
            objs.append({"type": "segment",
                "from": p0[vi][:], "to": p0[vj][:],
                "stroke": "#E0E0E0", "width": 1.5})

    # 目标等边三角形参考（虚线效果：短线段）
    cx = sum(p[0] for p in pts) / 3
    cy = sum(p[1] for p in pts) / 3

    # 能量柱（右侧）
    bar_x = 2.0
    bar_w = 0.15
    frac = (E - E_min) / E_range
    h = bar_height * frac
    objs.append({"type": "polygon",
        "points": [[bar_x, bar_base_y], [bar_x + bar_w, bar_base_y],
                   [bar_x + bar_w, bar_base_y + h], [bar_x, bar_base_y + h]],
        "fill": "#FFCDD2", "stroke": "#D32F2F", "width": 1})
    objs.append({"type": "segment",
        "from": [bar_x - 0.05, bar_base_y],
        "to": [bar_x + bar_w + 0.05, bar_base_y],
        "stroke": "#888", "width": 1})

    # 边长指示柱（右侧）
    for idx in range(3):
        bx = 2.4 + idx * 0.18
        elen = elens[idx]
        elen_frac = min(max((elen - 0.8) / 0.6, 0), 1)
        eh = bar_height * elen_frac
        objs.append({"type": "polygon",
            "points": [[bx, bar_base_y], [bx + 0.12, bar_base_y],
                       [bx + 0.12, bar_base_y + eh], [bx, bar_base_y + eh]],
            "fill": "#C8E6C9", "stroke": edge_colors[idx], "width": 1})
    # 目标线
    target_frac = (1.0 - 0.8) / 0.6
    target_h = bar_height * target_frac
    objs.append({"type": "segment",
        "from": [2.35, bar_base_y + target_h],
        "to": [3.0, bar_base_y + target_h],
        "stroke": "#888", "width": 1})

    frames.append({
        "objects": objs,
        "metrics": {
            "迭代": f"{it}/{n_iters}",
            "能量": f"{E:.1f}",
            "边01": f"{elens[0]:.4f}",
            "边12": f"{elens[1]:.4f}",
            "边02": f"{elens[2]:.4f}",
            "目标": "1.000",
        },
    })

# 在最终状态多停留一些帧
for _ in range(8):
    frames.append(frames[-1])

print(f"frames: {len(frames)}, iters: {n_iters}")
print(f"energy: {energies[0]:.1f} -> {energies[-1]:.1f}")
print(f"final edges: {edge_lengths_hist[-1]}")

simulation = {
    "bounds": [-0.5, -0.8, 3.2, 1.8],
    "fps": 3,
    "loop": True,
    "frames": frames,
}
