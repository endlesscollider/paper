import math

# 演示：从形变梯度 F 计算应力张量，在变形四边形上画出牵引力箭头与主应力方向。
# 动画平滑地从无变形到压缩+剪切，再到拉伸+剪切。

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

def eigS(M):
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
        n = math.sqrt(v[0] ** 2 + v[1] ** 2)
        if n > 1e-12:
            v = [v[0] / n, v[1] / n]
        vecs.append(v)
    return vs, vecs

# ============ 应力计算 ============

def linear_elastic_pk1(F, mu, lam):
    I = eye()
    E = mscl(msub(mm(mT(F), F), I), 0.5)
    trE = E[0][0] + E[1][1]
    S2 = madd(mscl(E, 2.0 * mu), mscl(I, lam * trE))
    P = mm(F, S2)
    return P, E

def pk1_to_cauchy(P, F):
    J = det2(F)
    if abs(J) < 1e-12:
        return eye()
    PFt = mm(P, mT(F))
    return mscl(PFt, 1.0 / J)

# ============ 动画 ============

def sm(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def lp(a, b, t):
    return a + (b - a) * t

def lpm(A, B, t):
    return [[lp(A[i][j], B[i][j], t) for j in range(2)] for i in range(2)]

I2 = eye()
F_comp = [[0.7, 0.15], [0.0, 1.2]]
F_tens = [[1.4, -0.1], [0.1, 0.85]]
F_shear = [[1.0, 0.35], [0.0, 1.0]]

def get_F(f):
    if f <= 50:
        return lpm(I2, F_comp, sm(f / 50.0))
    if f <= 90:
        return F_comp
    if f <= 140:
        return lpm(F_comp, F_tens, sm((f - 90) / 50.0))
    if f <= 180:
        return F_tens
    if f <= 230:
        return lpm(F_tens, F_shear, sm((f - 180) / 50.0))
    if f <= 270:
        return F_shear
    if f <= 320:
        return lpm(F_shear, I2, sm((f - 270) / 50.0))
    return I2

ref_sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

CX, CY = 1.6, 1.5
traction_scale = 0.18
face_colors = ["#D32F2F", "#388E3C", "#F57C00", "#7B1FA2"]

frames = []

for f in range(350):
    Fc = get_F(f)
    P, E = linear_elastic_pk1(Fc, 1.0, 2.0)
    sigma = pk1_to_cauchy(P, Fc)
    J = det2(Fc)
    evals, evecs = eigS(sigma)

    def_sq = [mv(Fc, p) for p in ref_sq]
    o = []

    # reference quad (gray dashed as segments)
    for i in range(4):
        ni = (i + 1) % 4
        o.append({"type": "segment",
            "from": [CX + ref_sq[i][0] - 0.5, CY + ref_sq[i][1] - 0.5],
            "to": [CX + ref_sq[ni][0] - 0.5, CY + ref_sq[ni][1] - 0.5],
            "stroke": "#CCCCCC", "width": 1})

    # deformed quad
    pts = [[CX + p[0] - 0.5, CY + p[1] - 0.5] for p in def_sq]
    o.append({"type": "polygon", "points": pts, "fill": "#BBDEFB", "stroke": "#1565C0", "width": 2})

    # traction vectors on each face
    for i in range(4):
        ni = (i + 1) % 4
        p1 = def_sq[i]
        p2 = def_sq[ni]
        mid = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
        edge = [p2[0] - p1[0], p2[1] - p1[1]]
        n = [edge[1], -edge[0]]
        nl = math.sqrt(n[0] ** 2 + n[1] ** 2)
        if nl > 1e-12:
            n = [n[0] / nl, n[1] / nl]
        t = mv(sigma, n)
        mx = CX + mid[0] - 0.5
        my = CY + mid[1] - 0.5
        tx = mx + traction_scale * t[0]
        ty = my + traction_scale * t[1]
        o.append({"type": "segment", "from": [mx, my], "to": [tx, ty],
            "stroke": face_colors[i], "width": 2.5})
        o.append({"type": "circle", "x": tx, "y": ty, "radius": 0.025,
            "fill": face_colors[i], "stroke": face_colors[i]})

    # principal stress directions
    cx_d = sum(p[0] for p in pts) / 4.0
    cy_d = sum(p[1] for p in pts) / 4.0
    max_ev = max(abs(evals[0]), abs(evals[1]), 0.01)
    for i in range(2):
        clr = "#D32F2F" if evals[i] < -0.01 else "#388E3C" if evals[i] > 0.01 else "#999"
        sc = 0.3 * abs(evals[i]) / max_ev
        for sign in [1, -1]:
            ex = cx_d + sign * sc * evecs[i][0]
            ey = cy_d + sign * sc * evecs[i][1]
            o.append({"type": "segment", "from": [cx_d, cy_d], "to": [ex, ey],
                "stroke": clr, "width": 2})

    # Mohr's circle (drawn on the right side)
    mcx = 3.5
    mcy = 1.5
    s1, s2 = evals
    ctr = (s1 + s2) / 2.0
    rad = abs(s2 - s1) / 2.0
    mohr_sc = 0.5 / max(rad, 0.01)
    mohr_sc = min(mohr_sc, 2.0)

    n_pts = 36
    circle_pts = []
    for k in range(n_pts):
        ang = 2.0 * math.pi * k / n_pts
        px = mcx + mohr_sc * (ctr + rad * math.cos(ang))
        py = mcy + mohr_sc * rad * math.sin(ang)
        circle_pts.append([px, py])
    o.append({"type": "polygon", "points": circle_pts, "fill": "#E3F2FD", "stroke": "#1976D2", "width": 1.5})

    # principal stress points on Mohr's circle
    o.append({"type": "circle", "x": mcx + mohr_sc * s1, "y": mcy,
        "radius": 0.04, "fill": "#D32F2F", "stroke": "#D32F2F"})
    o.append({"type": "circle", "x": mcx + mohr_sc * s2, "y": mcy,
        "radius": 0.04, "fill": "#388E3C", "stroke": "#388E3C"})
    # max shear point
    o.append({"type": "circle", "x": mcx + mohr_sc * ctr, "y": mcy + mohr_sc * rad,
        "radius": 0.035, "fill": "#F57C00", "stroke": "#F57C00"})
    # axis
    o.append({"type": "segment", "from": [mcx + mohr_sc * (ctr - rad * 1.3), mcy],
        "to": [mcx + mohr_sc * (ctr + rad * 1.3), mcy], "stroke": "#AAAAAA", "width": 1})
    o.append({"type": "segment", "from": [mcx + mohr_sc * ctr, mcy - mohr_sc * rad * 1.3],
        "to": [mcx + mohr_sc * ctr, mcy + mohr_sc * rad * 1.3], "stroke": "#AAAAAA", "width": 1})

    # separator
    o.append({"type": "segment", "from": [2.6, 0.2], "to": [2.6, 2.8],
        "stroke": "#E0E0E0", "width": 1})

    P_sym = abs(P[0][1] - P[1][0]) < 0.005
    s_sym = abs(sigma[0][1] - sigma[1][0]) < 0.01

    if f <= 50:   ph = "① →压缩+剪切"
    elif f <= 90: ph = "② 压缩(保持)"
    elif f <= 140: ph = "③ →拉伸"
    elif f <= 180: ph = "④ 拉伸(保持)"
    elif f <= 230: ph = "⑤ →纯剪切"
    elif f <= 270: ph = "⑥ 纯剪切(保持)"
    elif f <= 320: ph = "⑦ →原状"
    else:          ph = "⑧ 无变形"

    frames.append({
        "objects": o,
        "metrics": {
            "阶段": ph,
            "σ₁": f"{s1:.3f}",
            "σ₂": f"{s2:.3f}",
            "τ_max": f"{rad:.3f}",
            "J": f"{J:.3f}",
            "P对称": "否" if not P_sym else "是",
            "σ对称": "是" if s_sym else "否",
        },
    })

print(f"frames: {len(frames)}")
print(f"objects/frame: ~{len(frames[60]['objects'])}")

simulation = {
    "bounds": [0.0, 0.0, 5.0, 3.0],
    "fps": 25,
    "loop": True,
    "frames": frames,
}
