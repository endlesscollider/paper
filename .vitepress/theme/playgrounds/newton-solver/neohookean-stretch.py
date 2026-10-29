import math

# 演示 Neo-Hookean 超弹性材料：一维拉伸/压缩时能量与应力的响应。
# 左：一根条状单元被拉伸/压缩，实时显示 Neo-Hookean 和 StVK 能量值。
# 右：对应的 PK1 应力（红=Neo-Hookean，灰=StVK），可直观看到两者在极端压缩下的差异。
# 动画：拉伸比 λ 从 1.0 → 0.15（极端压缩）→ 3.0（极端拉伸）→ 1.0 循环。

# ============ 材料模型 ============

def nh_energy(lam, mu, la):
    if lam <= 0:
        return 1e6
    J = lam
    I1 = lam * lam + 1.0
    return 0.5 * mu * (I1 - 2.0) - mu * math.log(J) + 0.5 * la * (J - 1.0) ** 2

def stvk_energy(lam, mu, la):
    E11 = 0.5 * (lam * lam - 1.0)
    return mu * E11 * E11 + 0.5 * la * E11 * E11

def nh_pk1(lam, mu, la):
    if abs(lam) < 1e-12:
        return -1e6
    J = lam
    return mu * lam - mu / lam + la * (J - 1.0) * J / lam

def stvk_pk1(lam, mu, la):
    E11 = 0.5 * (lam * lam - 1.0)
    S = 2.0 * mu * E11 + la * E11
    return lam * S

# ============ 动画参数 ============

mu, la = 1.0, 2.0

def sm(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def lp(a, b, t):
    return a + (b - a) * t

def get_lambda(f):
    if f <= 60:
        return lp(1.0, 0.15, sm(f / 60.0))
    if f <= 100:
        return 0.15
    if f <= 180:
        return lp(0.15, 3.0, sm((f - 100) / 80.0))
    if f <= 220:
        return 3.0
    if f <= 300:
        return lp(3.0, 1.0, sm((f - 220) / 80.0))
    return 1.0

# ============ Pre-compute curve for the energy/stress plot ============

N_CURVE = 80
lam_min, lam_max = 0.12, 3.2
curve_lams = [lam_min + (lam_max - lam_min) * i / (N_CURVE - 1) for i in range(N_CURVE)]
curve_nh_e = [nh_energy(l, mu, la) for l in curve_lams]
curve_st_e = [stvk_energy(l, mu, la) for l in curve_lams]
curve_nh_p = [nh_pk1(l, mu, la) for l in curve_lams]
curve_st_p = [stvk_pk1(l, mu, la) for l in curve_lams]

# Normalize for display
max_e = 20.0
max_p = 15.0

# Plot area coordinates
# Energy plot: left area
EX, EY = 0.3, 0.3
EW, EH = 1.8, 1.6
# Stress plot: right area
PX, PY = 2.8, 0.3
PW, PH = 1.8, 1.6
# Bar: top center
BX, BY = 2.5, 2.35
BAR_H = 0.2
BAR_REF_W = 1.0

def plot_to_xy(px, py, ox, oy, w, h, xmin, xmax, ymin, ymax):
    x = ox + w * (px - xmin) / (xmax - xmin)
    y = oy + h * (py - ymin) / (ymax - ymin)
    return [x, y]

frames = []

for f in range(330):
    cur_lam = get_lambda(f)
    e_nh = nh_energy(cur_lam, mu, la)
    e_st = stvk_energy(cur_lam, mu, la)
    p_nh = nh_pk1(cur_lam, mu, la)
    p_st = stvk_pk1(cur_lam, mu, la)
    J = cur_lam

    o = []

    # --- bar visualization ---
    cur_w = BAR_REF_W * cur_lam
    ref_left = BX - BAR_REF_W / 2.0
    ref_right = BX + BAR_REF_W / 2.0
    cur_left = BX - cur_w / 2.0
    cur_right = BX + cur_w / 2.0

    # reference bar (gray)
    o.append({"type": "polygon", "points": [
        [ref_left, BY - BAR_H / 2], [ref_right, BY - BAR_H / 2],
        [ref_right, BY + BAR_H / 2], [ref_left, BY + BAR_H / 2]],
        "fill": "#E0E0E0", "stroke": "#AAAAAA", "width": 1})

    # deformed bar (colored by compression/tension)
    bar_clr = "#BBDEFB" if cur_lam >= 0.99 else "#FFCDD2" if cur_lam < 0.5 else "#FFE0B2"
    bar_stroke = "#1565C0" if cur_lam >= 0.99 else "#C62828" if cur_lam < 0.5 else "#E65100"
    cur_h = BAR_H / max(cur_lam, 0.15)
    cur_h = min(cur_h, BAR_H * 3.0)
    o.append({"type": "polygon", "points": [
        [cur_left, BY - cur_h / 2], [cur_right, BY - cur_h / 2],
        [cur_right, BY + cur_h / 2], [cur_left, BY + cur_h / 2]],
        "fill": bar_clr, "stroke": bar_stroke, "width": 2})

    # force arrows on bar ends
    arr_sc = 0.008 * abs(p_nh)
    arr_sc = min(arr_sc, 0.35)
    if abs(p_nh) > 0.05:
        if p_nh > 0:
            o.append({"type": "segment", "from": [cur_right, BY], "to": [cur_right + arr_sc, BY],
                "stroke": "#D32F2F", "width": 3})
            o.append({"type": "circle", "x": cur_right + arr_sc, "y": BY,
                "radius": 0.02, "fill": "#D32F2F", "stroke": "#D32F2F"})
            o.append({"type": "segment", "from": [cur_left, BY], "to": [cur_left - arr_sc, BY],
                "stroke": "#D32F2F", "width": 3})
            o.append({"type": "circle", "x": cur_left - arr_sc, "y": BY,
                "radius": 0.02, "fill": "#D32F2F", "stroke": "#D32F2F"})
        else:
            o.append({"type": "segment", "from": [cur_right + arr_sc, BY], "to": [cur_right, BY],
                "stroke": "#1565C0", "width": 3})
            o.append({"type": "circle", "x": cur_right, "y": BY,
                "radius": 0.02, "fill": "#1565C0", "stroke": "#1565C0"})
            o.append({"type": "segment", "from": [cur_left - arr_sc, BY], "to": [cur_left, BY],
                "stroke": "#1565C0", "width": 3})
            o.append({"type": "circle", "x": cur_left, "y": BY,
                "radius": 0.02, "fill": "#1565C0", "stroke": "#1565C0"})

    # --- energy curve plot ---
    # axes
    o.append({"type": "segment", "from": [EX, EY], "to": [EX + EW, EY], "stroke": "#AAAAAA", "width": 1})
    o.append({"type": "segment", "from": [EX, EY], "to": [EX, EY + EH], "stroke": "#AAAAAA", "width": 1})
    # λ=1 vertical line
    x1 = EX + EW * (1.0 - lam_min) / (lam_max - lam_min)
    o.append({"type": "segment", "from": [x1, EY], "to": [x1, EY + EH], "stroke": "#DDDDDD", "width": 1})

    # NH curve (red)
    for i in range(N_CURVE - 1):
        e1 = min(curve_nh_e[i], max_e)
        e2 = min(curve_nh_e[i + 1], max_e)
        x1c = EX + EW * i / (N_CURVE - 1)
        x2c = EX + EW * (i + 1) / (N_CURVE - 1)
        y1c = EY + EH * e1 / max_e
        y2c = EY + EH * e2 / max_e
        o.append({"type": "segment", "from": [x1c, y1c], "to": [x2c, y2c],
            "stroke": "#D32F2F", "width": 2})

    # StVK curve (gray dashed - just thinner)
    for i in range(N_CURVE - 1):
        e1 = min(curve_st_e[i], max_e)
        e2 = min(curve_st_e[i + 1], max_e)
        x1c = EX + EW * i / (N_CURVE - 1)
        x2c = EX + EW * (i + 1) / (N_CURVE - 1)
        y1c = EY + EH * e1 / max_e
        y2c = EY + EH * e2 / max_e
        o.append({"type": "segment", "from": [x1c, y1c], "to": [x2c, y2c],
            "stroke": "#AAAAAA", "width": 1})

    # current position marker on energy plot
    cx_e = EX + EW * (cur_lam - lam_min) / (lam_max - lam_min)
    cy_nh = EY + EH * min(e_nh, max_e) / max_e
    cy_st = EY + EH * min(e_st, max_e) / max_e
    o.append({"type": "circle", "x": cx_e, "y": cy_nh, "radius": 0.04,
        "fill": "#D32F2F", "stroke": "#D32F2F"})
    o.append({"type": "circle", "x": cx_e, "y": cy_st, "radius": 0.03,
        "fill": "#888888", "stroke": "#888888"})

    # --- stress curve plot ---
    o.append({"type": "segment", "from": [PX, PY], "to": [PX + PW, PY], "stroke": "#AAAAAA", "width": 1})
    o.append({"type": "segment", "from": [PX, PY], "to": [PX, PY + PH], "stroke": "#AAAAAA", "width": 1})
    # zero line
    y_zero = PY + PH * (max_p) / (2.0 * max_p)
    o.append({"type": "segment", "from": [PX, y_zero], "to": [PX + PW, y_zero],
        "stroke": "#DDDDDD", "width": 1})
    # λ=1 line
    x1p = PX + PW * (1.0 - lam_min) / (lam_max - lam_min)
    o.append({"type": "segment", "from": [x1p, PY], "to": [x1p, PY + PH], "stroke": "#DDDDDD", "width": 1})

    # NH stress curve (red)
    for i in range(N_CURVE - 1):
        p1 = max(-max_p, min(curve_nh_p[i], max_p))
        p2 = max(-max_p, min(curve_nh_p[i + 1], max_p))
        x1c = PX + PW * i / (N_CURVE - 1)
        x2c = PX + PW * (i + 1) / (N_CURVE - 1)
        y1c = PY + PH * (p1 + max_p) / (2.0 * max_p)
        y2c = PY + PH * (p2 + max_p) / (2.0 * max_p)
        o.append({"type": "segment", "from": [x1c, y1c], "to": [x2c, y2c],
            "stroke": "#D32F2F", "width": 2})

    # StVK stress curve (gray)
    for i in range(N_CURVE - 1):
        p1 = max(-max_p, min(curve_st_p[i], max_p))
        p2 = max(-max_p, min(curve_st_p[i + 1], max_p))
        x1c = PX + PW * i / (N_CURVE - 1)
        x2c = PX + PW * (i + 1) / (N_CURVE - 1)
        y1c = PY + PH * (p1 + max_p) / (2.0 * max_p)
        y2c = PY + PH * (p2 + max_p) / (2.0 * max_p)
        o.append({"type": "segment", "from": [x1c, y1c], "to": [x2c, y2c],
            "stroke": "#AAAAAA", "width": 1})

    # current marker on stress plot
    cx_p = PX + PW * (cur_lam - lam_min) / (lam_max - lam_min)
    p_nh_clamp = max(-max_p, min(p_nh, max_p))
    p_st_clamp = max(-max_p, min(p_st, max_p))
    cy_pnh = PY + PH * (p_nh_clamp + max_p) / (2.0 * max_p)
    cy_pst = PY + PH * (p_st_clamp + max_p) / (2.0 * max_p)
    o.append({"type": "circle", "x": cx_p, "y": cy_pnh, "radius": 0.04,
        "fill": "#D32F2F", "stroke": "#D32F2F"})
    o.append({"type": "circle", "x": cx_p, "y": cy_pst, "radius": 0.03,
        "fill": "#888888", "stroke": "#888888"})

    if f <= 60:   ph = "① →极端压缩"
    elif f <= 100: ph = "② 极端压缩(保持)"
    elif f <= 180: ph = "③ →极端拉伸"
    elif f <= 220: ph = "④ 极端拉伸(保持)"
    elif f <= 300: ph = "⑤ →原状"
    else:          ph = "⑥ 无变形"

    tag = "压缩" if cur_lam < 0.99 else "拉伸" if cur_lam > 1.01 else "无变形"

    frames.append({
        "objects": o,
        "metrics": {
            "阶段": ph,
            "λ": f"{cur_lam:.3f}",
            "Ψ_NH": f"{min(e_nh, 999):.2f}",
            "Ψ_StVK": f"{min(e_st, 999):.2f}",
            "P_NH": f"{p_nh:.2f}",
            "P_StVK": f"{p_st:.2f}",
            "状态": tag,
        },
    })

print(f"frames: {len(frames)}")
print(f"objects/frame: ~{len(frames[80]['objects'])}")

simulation = {
    "bounds": [0.0, 0.0, 5.0, 2.8],
    "fps": 25,
    "loop": True,
    "frames": frames,
}
