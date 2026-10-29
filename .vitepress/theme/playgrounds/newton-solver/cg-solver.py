import math

# 演示共轭梯度法 (CG) 求解过程：
# 在一个小型对称正定系统 Ax = b 上逐步迭代，实时显示：
# - 左：当前近似解 x 的分量（柱状图），逐步逼近真解
# - 右：残差范数的对数下降曲线
# - 上方：对应的一维弹簧网格变形可视化（解向量 = 各质点位移）

# ============ 构造一个小型三对角 SPD 系统（模拟弹簧网格刚度矩阵） ============

n = 8
# A = tridiagonal: 2 on diagonal, -0.5 on off-diagonal → SPD
A = [[0.0] * n for _ in range(n)]
for i in range(n):
    A[i][i] = 3.0
    if i > 0:
        A[i][i - 1] = -0.8
    if i < n - 1:
        A[i][i + 1] = -0.8

b = [0.0] * n
b[0] = 1.0
b[n // 2] = 2.0
b[n - 1] = -1.0

# True solution via forward-substitution (Thomas algorithm for tridiagonal)
def solve_tridiag(A, b):
    nn = len(b)
    c = [0.0] * nn
    d = [0.0] * nn
    x = [0.0] * nn
    c[0] = A[0][1] / A[0][0]
    d[0] = b[0] / A[0][0]
    for i in range(1, nn):
        m = A[i][i] - A[i][i - 1] * c[i - 1]
        if i < nn - 1:
            c[i] = A[i][i + 1] / m
        d[i] = (b[i] - A[i][i - 1] * d[i - 1]) / m
    x[nn - 1] = d[nn - 1]
    for i in range(nn - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x

x_true = solve_tridiag(A, b)

# ============ CG iteration, recording all intermediate states ============

def matvec(A, v):
    nn = len(v)
    r = [0.0] * nn
    for i in range(nn):
        for j in range(nn):
            r[i] += A[i][j] * v[j]
    return r

def dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def vadd(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def vsub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]

def vscl(a, s):
    return [ai * s for ai in a]

x = [0.0] * n
r = b[:]  # r = b - A@0 = b
p = r[:]
rs_old = dot(r, r)

cg_states = [(x[:], math.sqrt(rs_old))]

for k in range(n + 2):
    Ap = matvec(A, p)
    alpha = rs_old / max(dot(p, Ap), 1e-15)
    x = vadd(x, vscl(p, alpha))
    r = vsub(r, vscl(Ap, alpha))
    rs_new = dot(r, r)
    res = math.sqrt(rs_new)
    cg_states.append((x[:], res))
    if res < 1e-10:
        break
    beta = rs_new / max(rs_old, 1e-15)
    p = vadd(r, vscl(p, beta))
    rs_old = rs_new

total_iters = len(cg_states) - 1

# ============ 动画 ============

# Layout
# Spring chain: top band y=[2.0, 2.6], x=[0.5, 4.5]
# Bar chart (solution): left, y=[0.2, 1.7], x=[0.3, 2.3]
# Residual curve: right, y=[0.2, 1.7], x=[2.7, 4.7]

SCX0, SCX1 = 0.5, 4.5
SCY = 2.3
BAR_X0, BAR_X1 = 0.3, 2.3
BAR_Y0, BAR_Y1 = 0.2, 1.6
RES_X0, RES_X1 = 2.7, 4.7
RES_Y0, RES_Y1 = 0.2, 1.6

max_x = max(abs(v) for v in x_true) * 1.2
max_res_log = 1.0  # log10 of initial residual (approx)
min_res_log = -11.0

frames_per_iter = 15
hold_frames = 30
total_frames = (total_iters + 1) * frames_per_iter + hold_frames

frames = []

for f in range(total_frames):
    iter_f = f / frames_per_iter
    iter_idx = min(int(iter_f), total_iters)
    frac = min(1.0, iter_f - iter_idx) if iter_idx < total_iters else 1.0

    # Interpolate between iter_idx and iter_idx+1
    x_cur, res_cur = cg_states[iter_idx]
    if iter_idx < total_iters and frac > 0:
        x_next, res_next = cg_states[iter_idx + 1]
        t = frac * frac * (3 - 2 * frac)
        x_interp = [x_cur[i] + t * (x_next[i] - x_cur[i]) for i in range(n)]
        res_interp = res_cur + t * (res_next - res_cur)
    else:
        x_interp = x_cur[:]
        res_interp = res_cur

    o = []

    # --- Spring chain visualization ---
    spacing = (SCX1 - SCX0) / (n + 1)
    node_y = SCY
    rest_positions = [SCX0 + spacing * (i + 1) for i in range(n)]
    disp_scale = 0.3
    cur_positions = [rest_positions[i] + disp_scale * x_interp[i] for i in range(n)]

    # Fixed walls
    o.append({"type": "segment", "from": [SCX0, SCY - 0.15], "to": [SCX0, SCY + 0.15],
        "stroke": "#5D4037", "width": 3})
    o.append({"type": "segment", "from": [SCX1, SCY - 0.15], "to": [SCX1, SCY + 0.15],
        "stroke": "#5D4037", "width": 3})

    # Springs as zigzag segments (simplified as straight segments)
    all_pts = [SCX0] + cur_positions + [SCX1]
    for i in range(len(all_pts) - 1):
        o.append({"type": "segment", "from": [all_pts[i], node_y], "to": [all_pts[i + 1], node_y],
            "stroke": "#90CAF9", "width": 1.5})

    # Nodes
    for i in range(n):
        # rest position marker
        o.append({"type": "segment",
            "from": [rest_positions[i], node_y - 0.08],
            "to": [rest_positions[i], node_y + 0.08],
            "stroke": "#E0E0E0", "width": 1})
        # current position
        clr = "#D32F2F" if abs(x_interp[i]) > 0.01 else "#1565C0"
        o.append({"type": "circle", "x": cur_positions[i], "y": node_y,
            "radius": 0.06, "fill": clr, "stroke": clr})

    # External force indicators
    for i in range(n):
        if abs(b[i]) > 0.01:
            arr_len = 0.12 * b[i] / max(abs(v) for v in b)
            o.append({"type": "segment",
                "from": [cur_positions[i], node_y + 0.1],
                "to": [cur_positions[i], node_y + 0.1 + arr_len],
                "stroke": "#F57C00", "width": 2})

    # --- Bar chart: current solution vs true solution ---
    bar_w = (BAR_X1 - BAR_X0) / (n * 2 + 1)
    for i in range(n):
        bx = BAR_X0 + bar_w * (2 * i + 0.5)
        zero_y = BAR_Y0 + (BAR_Y1 - BAR_Y0) * 0.5
        # true solution (gray background bar)
        h_true = (BAR_Y1 - BAR_Y0) * 0.45 * x_true[i] / max_x
        o.append({"type": "polygon", "points": [
            [bx, zero_y], [bx + bar_w * 0.8, zero_y],
            [bx + bar_w * 0.8, zero_y + h_true], [bx, zero_y + h_true]],
            "fill": "#E0E0E0", "stroke": "#BBBBBB", "width": 1})
        # current approximation (blue bar)
        h_cur = (BAR_Y1 - BAR_Y0) * 0.45 * x_interp[i] / max_x
        o.append({"type": "polygon", "points": [
            [bx + bar_w * 0.1, zero_y], [bx + bar_w * 0.7, zero_y],
            [bx + bar_w * 0.7, zero_y + h_cur], [bx + bar_w * 0.1, zero_y + h_cur]],
            "fill": "#1565C0", "stroke": "#0D47A1", "width": 1})

    # zero line for bar chart
    zero_y = BAR_Y0 + (BAR_Y1 - BAR_Y0) * 0.5
    o.append({"type": "segment", "from": [BAR_X0, zero_y], "to": [BAR_X1, zero_y],
        "stroke": "#AAAAAA", "width": 1})
    # bar chart axes
    o.append({"type": "segment", "from": [BAR_X0, BAR_Y0], "to": [BAR_X0, BAR_Y1],
        "stroke": "#AAAAAA", "width": 1})
    o.append({"type": "segment", "from": [BAR_X0, BAR_Y0], "to": [BAR_X1, BAR_Y0],
        "stroke": "#AAAAAA", "width": 1})

    # --- Residual convergence curve ---
    # axes
    o.append({"type": "segment", "from": [RES_X0, RES_Y0], "to": [RES_X0, RES_Y1],
        "stroke": "#AAAAAA", "width": 1})
    o.append({"type": "segment", "from": [RES_X0, RES_Y0], "to": [RES_X1, RES_Y0],
        "stroke": "#AAAAAA", "width": 1})

    # convergence threshold line
    thr_log = -10.0
    thr_y = RES_Y0 + (RES_Y1 - RES_Y0) * (thr_log - min_res_log) / (max_res_log - min_res_log)
    o.append({"type": "segment", "from": [RES_X0, thr_y], "to": [RES_X1, thr_y],
        "stroke": "#E65100", "width": 1})

    # plot all residual points up to current iteration
    for k in range(min(iter_idx + 2, len(cg_states))):
        _, rk = cg_states[k]
        rk_log = math.log10(max(rk, 1e-15))
        rx = RES_X0 + (RES_X1 - RES_X0) * k / max(total_iters, 1)
        ry = RES_Y0 + (RES_Y1 - RES_Y0) * (rk_log - min_res_log) / (max_res_log - min_res_log)
        ry = max(RES_Y0, min(RES_Y1, ry))
        if k > 0:
            _, rk_prev = cg_states[k - 1]
            rk_prev_log = math.log10(max(rk_prev, 1e-15))
            rx_prev = RES_X0 + (RES_X1 - RES_X0) * (k - 1) / max(total_iters, 1)
            ry_prev = RES_Y0 + (RES_Y1 - RES_Y0) * (rk_prev_log - min_res_log) / (max_res_log - min_res_log)
            ry_prev = max(RES_Y0, min(RES_Y1, ry_prev))
            o.append({"type": "segment", "from": [rx_prev, ry_prev], "to": [rx, ry],
                "stroke": "#1565C0", "width": 2})
        o.append({"type": "circle", "x": rx, "y": ry, "radius": 0.03,
            "fill": "#1565C0", "stroke": "#1565C0"})

    # current residual marker (larger)
    res_log = math.log10(max(res_interp, 1e-15))
    crx = RES_X0 + (RES_X1 - RES_X0) * min(iter_idx + frac, total_iters) / max(total_iters, 1)
    cry = RES_Y0 + (RES_Y1 - RES_Y0) * (res_log - min_res_log) / (max_res_log - min_res_log)
    cry = max(RES_Y0, min(RES_Y1, cry))
    o.append({"type": "circle", "x": crx, "y": cry, "radius": 0.05,
        "fill": "#D32F2F", "stroke": "#D32F2F"})

    # separator
    o.append({"type": "segment", "from": [2.5, 0.1], "to": [2.5, 1.7],
        "stroke": "#E0E0E0", "width": 1})

    err = math.sqrt(sum((x_interp[i] - x_true[i]) ** 2 for i in range(n)))

    frames.append({
        "objects": o,
        "metrics": {
            "迭代": f"{iter_idx}/{total_iters}",
            "残差": f"{res_interp:.2e}",
            "误差": f"{err:.2e}",
            "维度n": str(n),
        },
    })

print(f"frames: {len(frames)}")
print(f"CG converged in {total_iters} iterations (n={n})")
print(f"objects/frame: ~{len(frames[50]['objects'])}")

simulation = {
    "bounds": [0.0, 0.0, 5.0, 2.7],
    "fps": 25,
    "loop": True,
    "frames": frames,
}
