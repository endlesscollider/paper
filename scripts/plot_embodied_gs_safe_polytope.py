#!/usr/bin/env python3
"""具身高斯综述配图 3：把高斯椭球变成安全走廊。

复现 Splat-Nav 的几何构造链：
  1. 沿一条粗路径取种子点，对每个种子点找出附近的高斯椭球
  2. 从种子点向椭球求最近点，得到一张支撑超平面，切出安全半空间
  3. 每段的半空间求交 = 一段凸多面体；相邻多面体叠在一起 = 安全走廊
  4. 分段贝塞尔曲线：所有控制点落在各自多面体内，
     由贝塞尔曲线的凸包性质保证整条曲线安全
  5. 用二次规划求解控制点（目标：路径长度代理最小）

脚本末尾对求解结果做安全断言：任何采样点越界就直接报错。

输出：public/embodied_gs_safe_polytope.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

R_ROBOT = 0.12
DEG = 6                                     # 每段贝塞尔曲线的次数
RADIUS = 0.95                               # 收集附近椭球的半径

START = np.array([0.50, 0.40])
GOAL = np.array([4.40, 1.60])
WAY = np.array([START,
                [1.35, 0.28],
                [2.35, 1.10],
                [3.30, 2.15],
                GOAL])

# (cx, cy, a, b, angle_deg)
OBSTACLES = [
    (1.60, 1.05, 0.30, 0.55, 0.0),
    (2.60, 0.60, 0.45, 0.22, -20.0),
    (2.30, 2.05, 0.50, 0.22, 10.0),
    (3.60, 2.45, 0.30, 0.30, 0.0),
]
P = len(WAY) - 1


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def ell_boundary(ob, n=1200):
    cx, cy, a, b, deg = ob
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    local = np.stack([a * np.cos(phi), b * np.sin(phi)])
    return (rot(deg) @ local).T + np.array([cx, cy])


def closest_on(ob, p):
    """椭球边界上距 p 最近的点。"""
    cx, cy, a, b, deg = ob
    R = rot(deg)

    def d2(phi):
        local = np.array([a * np.cos(phi), b * np.sin(phi)])
        return float(np.sum((p - (R @ local + np.array([cx, cy]))) ** 2))

    res = minimize_scalar(d2, bounds=(0, 2 * np.pi), method="bounded",
                          options={"xatol": 1e-12})
    local = np.array([a * np.cos(res.x), b * np.sin(res.x)])
    return R @ local + np.array([cx, cy])


def dist_point_seg(p, a, b):
    d = b - a
    t = np.clip(np.dot(p - a, d) / np.dot(d, d), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * d)))


# ---- 1. 每段构造安全多面体 ----
# 只保留两端点都落在半空间内的约束，确保 QP 初始解可行
planes = []          # planes[p] = [(n, q), ...]，约束 n·x >= n·q + R
for p in range(P):
    a, b = WAY[p], WAY[p + 1]
    seed = 0.5 * (a + b)
    plist = []
    for ob in OBSTACLES:
        if dist_point_seg(np.array([ob[0], ob[1]]), a, b) < RADIUS:
            q = closest_on(ob, seed)
            n = (seed - q) / np.linalg.norm(seed - q)
            # 跳过会把端点切掉的半平面（端点是外部给定的可行点）
            ma = float(np.dot(n, a - q) - R_ROBOT)
            mb = float(np.dot(n, b - q) - R_ROBOT)
            if ma >= -1e-6 and mb >= -1e-6:
                plist.append((n, q))
    planes.append(plist)
    print(f"[诊断] 第 {p} 段：{len(plist)} 个有效半平面约束")

# ---- 2. 分段贝塞尔控制点的二次规划 ----
n_ctrl = (DEG + 1) * P
x0 = np.zeros(n_ctrl * 2)
for p in range(P):
    for m in range(DEG + 1):
        t = m / DEG
        x0[(p * (DEG + 1) + m) * 2: (p * (DEG + 1) + m) * 2 + 2] = \
            (1 - t) * WAY[p] + t * WAY[p + 1]


def unpack(X):
    return X.reshape(P, DEG + 1, 2)


def cost(X):
    C = unpack(X)
    return float(np.sum((C[:, 1:, :] - C[:, :-1, :]) ** 2))


def eq_res(X):
    C = unpack(X)
    out = [C[0, 0] - START, C[P - 1, DEG] - GOAL]
    out += [C[p, DEG] - C[p + 1, 0] for p in range(P - 1)]
    return np.concatenate(out)


def ineq_vals(X):
    C = unpack(X)
    vals = []
    for p in range(P):
        for m in range(DEG + 1):
            for n, q in planes[p]:
                vals.append(float(np.dot(n, C[p, m] - q)) - R_ROBOT)
    return np.array(vals)


cons = [{"type": "eq", "fun": eq_res},
        {"type": "ineq", "fun": ineq_vals}]
res = minimize(cost, x0, method="SLSQP", constraints=cons,
               options={"maxiter": 600, "ftol": 1e-12})
print(f"[诊断] QP 收敛: {res.success}, 目标值 = {res.fun:.5f}")
assert res.success, res.message
CTRL = unpack(res.x)


def bezier(c, s):
    s = np.asarray(s)[:, None]
    n = c.shape[0] - 1
    from math import comb
    out = np.zeros((len(s), 2))
    for i in range(n + 1):
        out += comb(n, i) * ((1 - s) ** (n - i)) * (s ** i) * c[i]
    return out


# ---- 3. 安全断言 ----
s_fine = np.linspace(0, 1, 400)
min_margin = np.inf
for p in range(P):
    pts = bezier(CTRL[p], s_fine)
    for n, q in planes[p]:
        min_margin = min(min_margin,
                         float(np.min(np.dot(pts - q, n))) - R_ROBOT)
assert min_margin >= -1e-6, f"曲线越出多面体，最小裕度 {min_margin:+.4f}"

end_err = max(np.linalg.norm(CTRL[0, 0] - START),
              np.linalg.norm(CTRL[P - 1, DEG] - GOAL))
print(f"[诊断] 曲线最小裕度 = {min_margin:+.5f}")
print(f"[诊断] 端点误差 = {end_err:.2e}")
print(f"[诊断] 控制点越界数 = {int((ineq_vals(res.x) < -1e-9).sum())}")

# ---- 4. 绘制 ----
gx, gy = np.meshgrid(np.linspace(0.0, 4.95, 620), np.linspace(-0.30, 2.95, 620))
grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

# 走廊 = 各段多面体之并
corridor = np.full(gx.size, -1e9)
for p in range(P):
    mp = np.full(gx.size, 1e9)
    for n, q in planes[p]:
        mp = np.minimum(mp, n[0] * (grid_pts[:, 0] - q[0])
                        + n[1] * (grid_pts[:, 1] - q[1]) - R_ROBOT)
    corridor = np.maximum(corridor, mp)
corridor = corridor.reshape(gx.shape)

# 障碍物膨胀区（用采样点 KD 树近似距离场）
bnd = np.concatenate([ell_boundary(ob, 3000) for ob in OBSTACLES])
tree = cKDTree(bnd)
dd = tree.query(grid_pts)[0].reshape(gx.shape)
safe = (corridor >= 0.0) & (dd >= R_ROBOT)

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.contourf(gx, gy, safe.astype(float), levels=[0.5, 1.5],
            colors=["#E8F5E9"], alpha=0.9)
ax.contour(gx, gy, corridor, levels=[0.0], colors=["#2E7D32"], linewidths=1.3)

for ob in OBSTACLES:
    pts = ell_boundary(ob, 500)
    ax.fill(pts[:, 0], pts[:, 1], color="#B0BEC5", alpha=0.55, zorder=2)
    ax.plot(pts[:, 0], pts[:, 1], color="#455A64", lw=1.5, zorder=3)
    cen = np.array([ob[0], ob[1]])
    nr = pts - cen
    nr /= np.linalg.norm(nr, axis=1, keepdims=True)
    off = pts + R_ROBOT * nr
    ax.plot(off[:, 0], off[:, 1], color="#F44336", lw=1.2, ls="--", zorder=3)

# 支撑超平面
for p in range(P):
    for n, q in planes[p]:
        b = np.dot(n, q) + R_ROBOT
        xs = np.array([0.0, 4.95])
        if abs(n[1]) > 1e-6:
            ys = (b - n[0] * xs) / n[1]
            ax.plot(xs, ys, color="#E65100", lw=1.6, alpha=0.85, zorder=4)
        else:
            ax.axvline(b, color="#E65100", lw=1.6, alpha=0.85, zorder=4)

for p in range(P):
    pts = bezier(CTRL[p], np.linspace(0, 1, 200))
    ax.plot(pts[:, 0], pts[:, 1], color="#1565C0", lw=3.0, zorder=6,
            label="piecewise Bézier" if p == 0 else None)

ax.plot(*START, "o", color="#4CAF50", ms=13, zorder=7, label="start")
ax.plot(*GOAL, "*", color="#E91E63", ms=20, zorder=7, label="goal")
ax.plot(np.array(WAY)[1:-1, 0], np.array(WAY)[1:-1, 1], "x", color="#212121",
        ms=9, mew=2.2, zorder=7, label="seed waypoints")
ax.plot([np.nan], [np.nan], color="#E65100", lw=1.6,
        label="supporting half-planes")

ax.set_xlim(0.0, 4.95)
ax.set_ylim(-0.30, 2.95)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Gaussian ellipsoids -> half-planes -> corridor -> Bézier inside")
ax.legend(loc="upper left", fontsize=8)

fig.tight_layout()
fig.savefig("public/embodied_gs_safe_polytope.png", dpi=150, bbox_inches="tight")
print("saved public/embodied_gs_safe_polytope.png")
