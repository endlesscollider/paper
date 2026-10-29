import math

# 左侧网格展示完整变形 F = RS（拉伸 + 旋转），右侧只展示 S（纯拉伸）。
# 动画循环六个阶段，核心对比：叠加旋转时右侧网格不变 → S 不含旋转信息。

# ============ 2×2 矩阵工具 ============

def eye():
    return [[1.0, 0.0], [0.0, 1.0]]

def mm(A, B):
    return [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
            [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]

def mv(M, v):
    return [M[0][0]*v[0]+M[0][1]*v[1], M[1][0]*v[0]+M[1][1]*v[1]]

def mT(A):
    return [[A[0][0], A[1][0]], [A[0][1], A[1][1]]]

def det2(A):
    return A[0][0]*A[1][1] - A[0][1]*A[1][0]

def inv2(A):
    d = det2(A)
    if abs(d) < 1e-12:
        return eye()
    return [[A[1][1]/d, -A[0][1]/d], [-A[1][0]/d, A[0][0]/d]]

def madd(A, B):
    return [[A[0][0]+B[0][0], A[0][1]+B[0][1]],
            [A[1][0]+B[1][0], A[1][1]+B[1][1]]]

def mscl(A, s):
    return [[A[0][0]*s, A[0][1]*s], [A[1][0]*s, A[1][1]*s]]

def msub(A, B):
    return [[A[0][0]-B[0][0], A[0][1]-B[0][1]],
            [A[1][0]-B[1][0], A[1][1]-B[1][1]]]

def rot(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s], [s, c]]

def polar(F):
    """2×2 极分解 F = R S (R 正交, S 对称正定)"""
    C = mm(mT(F), F)
    dc = det2(C)
    if dc < 1e-12:
        return eye(), eye()
    s = math.sqrt(dc)
    t = C[0][0] + C[1][1]
    d = math.sqrt(max(1e-12, t + 2.0 * s))
    S = mscl(madd(C, mscl(eye(), s)), 1.0 / d)
    R = mm(F, inv2(S))
    return R, S

def eigS(M):
    """2×2 对称矩阵特征值与特征向量"""
    a, b, d = M[0][0], M[0][1], M[1][1]
    t = a + d
    dt = a * d - b * b
    disc = math.sqrt(max(0.0, t * t / 4.0 - dt))
    vs = [t / 2.0 - disc, t / 2.0 + disc]
    vecs = []
    for lam in vs:
        if abs(b) > 1e-10:
            v = [lam - d, b]
        else:
            v = [1.0, 0.0] if abs(a - lam) < abs(d - lam) else [0.0, 1.0]
        n = math.sqrt(v[0]**2 + v[1]**2)
        if n > 1e-12:
            v = [v[0] / n, v[1] / n]
        vecs.append(v)
    return vs, vecs

# ============ 5×5 参考网格，中心在原点 ============

gn = 5
ref = []
for j in range(gn):
    for i in range(gn):
        ref.append([i / (gn - 1) - 0.5, j / (gn - 1) - 0.5])

# ============ 动画时间线 ============

def sm(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def lp(a, b, t):
    return a + (b - a) * t

def lpm(A, B, t):
    return [[lp(A[i][j], B[i][j], t) for j in range(2)] for i in range(2)]

S_tgt = [[1.4, 0.15], [0.15, 0.7]]
th1 = math.radians(25)
th2 = math.radians(40)

def get_RS(f):
    """根据帧号返回 (R, S)"""
    I = eye()
    if f <= 40:
        return I, lpm(I, S_tgt, sm(f / 40.0))
    if f <= 80:
        return rot(lp(0, th1, sm((f - 40) / 40.0))), S_tgt
    if f <= 100:
        return rot(th1), S_tgt
    if f <= 140:
        t = sm((f - 100) / 40.0)
        return rot(lp(th1, th2, t)), lpm(S_tgt, I, t)
    if f <= 160:
        return rot(th2), I
    if f <= 200:
        return rot(lp(th2, 0, sm((f - 160) / 40.0))), I
    return I, I

def draw_grid(objs, cx, cy, M, fc, sc):
    """绘制参考网格（灰线）+ 变形后多边形"""
    for j in range(gn):
        for i in range(gn):
            x0, y0 = ref[j * gn + i]
            if i < gn - 1:
                x1, y1 = ref[j * gn + i + 1]
                objs.append({"type": "segment",
                    "from": [cx + x0, cy + y0], "to": [cx + x1, cy + y1],
                    "stroke": "#D0D0D0", "width": 1})
            if j < gn - 1:
                x1, y1 = ref[(j + 1) * gn + i]
                objs.append({"type": "segment",
                    "from": [cx + x0, cy + y0], "to": [cx + x1, cy + y1],
                    "stroke": "#D0D0D0", "width": 1})
    dp = [mv(M, p) for p in ref]
    for j in range(gn - 1):
        for i in range(gn - 1):
            a = j * gn + i
            pts = [[cx + dp[k][0], cy + dp[k][1]]
                   for k in [a, a + 1, a + gn + 1, a + gn]]
            objs.append({"type": "polygon", "points": pts,
                "fill": fc, "stroke": sc, "width": 1.5})

# ============ 构建帧 ============

LX, LY = 0.9, 1.0
RX, RY = 3.3, 1.0
frames = []

for f in range(220):
    Rc, Sc = get_RS(f)
    Fc = mm(Rc, Sc)

    Rd, Sd = polar(Fc)
    J = det2(Fc)
    C = mm(mT(Fc), Fc)
    E = mscl(msub(C, eye()), 0.5)
    vs, vcs = eigS(Sd)
    tr = math.atan2(Rd[1][0], Rd[0][0])

    o = []

    draw_grid(o, LX, LY, Fc, "#BBDEFB", "#1565C0")
    draw_grid(o, RX, RY, Sd, "#FFE0B2", "#E65100")

    for i in range(2):
        clr = "#D32F2F" if vs[i] > 1.01 else "#1565C0" if vs[i] < 0.99 else "#999"
        ex = RX + 0.4 * vs[i] * vcs[i][0]
        ey = RY + 0.4 * vs[i] * vcs[i][1]
        o.append({"type": "segment", "from": [RX, RY], "to": [ex, ey],
            "stroke": clr, "width": 3})
        o.append({"type": "circle", "x": ex, "y": ey,
            "radius": 0.025, "fill": clr, "stroke": clr})

    o.append({"type": "segment", "from": [2.1, -0.3], "to": [2.1, 2.3],
        "stroke": "#E0E0E0", "width": 1})

    if f <= 40:   ph = "① 纯拉伸"
    elif f <= 80: ph = "② +旋转"
    elif f <= 100: ph = "③ 保持"
    elif f <= 140: ph = "④ →纯旋转"
    elif f <= 160: ph = "⑤ 纯旋转"
    elif f <= 200: ph = "⑥ →原状"
    else:          ph = "⑦ 无变形"

    frames.append({
        "objects": o,
        "metrics": {
            "阶段": ph,
            "旋转θ": f"{math.degrees(tr):.1f}°",
            "主拉伸": f"{vs[0]:.3f}, {vs[1]:.3f}",
            "J=detF": f"{J:.3f}",
            "E₁₁": f"{E[0][0]:.3f}",
            "E₂₂": f"{E[1][1]:.3f}",
        },
    })

print(f"frames: {len(frames)}")
print(f"objects/frame: ~{len(frames[50]['objects'])}")

simulation = {
    "bounds": [-0.5, -0.5, 4.7, 2.5],
    "fps": 20,
    "loop": True,
    "frames": frames,
}
