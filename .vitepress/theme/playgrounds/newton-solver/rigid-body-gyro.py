import math

# 自由旋转刚体仿真：积木绕主轴旋转（带扰动），展示陀螺进动。
# 左侧：刚体的 3D 投影动画（一个旋转的矩形 + 角速度向量）。
# 右侧：ω_x-ω_y 相平面上的进动椭圆轨迹逐步绘出。
# 度量面板显示 ω 三分量、动能和能量漂移。

# ============ 欧拉方程仿真 ============

I_body = [0.00104, 0.00354, 0.00417]  # 主转动惯量

def cross3(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def euler_rhs(w):
    Iw = [I_body[0]*w[0], I_body[1]*w[1], I_body[2]*w[2]]
    c = cross3(w, Iw)
    return [-c[0]/I_body[0], -c[1]/I_body[1], -c[2]/I_body[2]]

omega = [0.1, 0.0, 5.0]
dt = 0.001
n_steps = 3000

omega_hist = []
energy_hist = []

for step in range(n_steps):
    omega_hist.append(omega[:])
    E = 0.5 * (I_body[0]*omega[0]**2 + I_body[1]*omega[1]**2 + I_body[2]*omega[2]**2)
    energy_hist.append(E)

    dw = euler_rhs(omega)
    omega = [omega[i] + dt * dw[i] for i in range(3)]

omega_hist.append(omega[:])
energy_hist.append(0.5*(I_body[0]*omega[0]**2+I_body[1]*omega[1]**2+I_body[2]*omega[2]**2))

E0 = energy_hist[0]

# ============ 旋转矩阵累积（用于刚体可视化） ============

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1,0,0],[0,c,-s],[0,s,c]]

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c,-s,0],[s,c,0],[0,0,1]]

def mm3(A, B):
    r = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r[i][j] += A[i][k]*B[k][j]
    return r

def mv3(M, v):
    return [M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
            M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
            M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]]

def rot_from_rotvec(rv):
    """Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²"""
    angle = math.sqrt(rv[0]**2 + rv[1]**2 + rv[2]**2)
    if angle < 1e-12:
        return [[1,0,0],[0,1,0],[0,0,1]]
    ax = [rv[0]/angle, rv[1]/angle, rv[2]/angle]
    s = math.sin(angle)
    c = math.cos(angle)
    t = 1.0 - c
    x, y, z = ax
    return [[t*x*x+c,   t*x*y-s*z, t*x*z+s*y],
            [t*x*y+s*z, t*y*y+c,   t*y*z-s*x],
            [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]

# 积累姿态四元数（存为旋转矩阵列表）
R_cur = [[1,0,0],[0,1,0],[0,0,1]]
R_hist = [R_cur]
for step in range(n_steps):
    w = omega_hist[step]
    drv = [w[0]*dt, w[1]*dt, w[2]*dt]
    dR = rot_from_rotvec(drv)
    R_cur = mm3(dR, R_cur)
    R_hist.append(R_cur)

# ============ 3D → 2D 投影 ============

# 等轴测投影：先绕 x 转 35°，再绕 z 转 -45°
view_R = mm3(rot_z(-0.78), rot_x(0.6))

def proj(p3):
    """3D → 2D 等轴测投影"""
    vp = mv3(view_R, p3)
    return [vp[0], vp[1]]

# ============ 刚体几何：一个 4cm x 6cm x 7cm 的积木 ============

hw, hh, hd = 0.02, 0.03, 0.035
box_verts = [
    [-hw,-hh,-hd],[ hw,-hh,-hd],[ hw, hh,-hd],[-hw, hh,-hd],
    [-hw,-hh, hd],[ hw,-hh, hd],[ hw, hh, hd],[-hw, hh, hd],
]
box_faces = [
    [0,1,2,3], [4,5,6,7],  # front/back
    [0,1,5,4], [2,3,7,6],  # bottom/top
    [0,3,7,4], [1,2,6,5],  # left/right
]
face_colors = ["#BBDEFB", "#90CAF9", "#E3F2FD", "#64B5F6", "#42A5F5", "#BBDEFB"]

# ============ 构建动画帧 ============

# 布局：左侧刚体动画，右侧相平面
BODY_CX, BODY_CY = -0.6, 0.0
PHASE_CX, PHASE_CY = 0.7, 0.0
body_scale = 12.0  # 放大刚体以在画布上可见
phase_scale = 2.5  # ω_x, ω_y 范围约 ±0.1, 放大到画布

frames = []
sample_every = 10  # 每 10 步存一帧

# 预计算 ω 范围
wx_range = max(abs(w[0]) for w in omega_hist)
wy_range = max(abs(w[1]) for w in omega_hist)

for fi in range(0, n_steps + 1, sample_every):
    R_body = R_hist[fi]
    w = omega_hist[min(fi, len(omega_hist)-1)]
    E = energy_hist[min(fi, len(energy_hist)-1)]
    rel_err = (E - E0) / E0 * 100.0

    objs = []

    # ---- 左侧：旋转刚体 ----

    # 变换刚体顶点
    world_verts = [mv3(R_body, v) for v in box_verts]
    proj_verts = [proj(v) for v in world_verts]

    # 按面的平均 z 排序（画家算法）
    face_z = []
    for fi_face, face in enumerate(box_faces):
        vz = sum(mv3(view_R, world_verts[v])[2] for v in face) / len(face)
        face_z.append((vz, fi_face))
    face_z.sort()

    for _, fi_face in face_z:
        face = box_faces[fi_face]
        pts = [[BODY_CX + proj_verts[v][0]*body_scale,
                BODY_CY + proj_verts[v][1]*body_scale] for v in face]
        objs.append({"type": "polygon", "points": pts,
            "fill": face_colors[fi_face], "stroke": "#1565C0", "width": 1})

    # 角速度向量（橙色箭头）
    w_dir = mv3(R_body, [0, 0, 1])  # z 主轴在世界坐标系下的方向
    w_proj = proj(w_dir)
    arrow_len = 0.5
    objs.append({"type": "segment",
        "from": [BODY_CX, BODY_CY],
        "to": [BODY_CX + w_proj[0]*arrow_len, BODY_CY + w_proj[1]*arrow_len],
        "stroke": "#FF6F00", "width": 3})
    objs.append({"type": "circle",
        "x": BODY_CX + w_proj[0]*arrow_len,
        "y": BODY_CY + w_proj[1]*arrow_len,
        "radius": 0.025, "fill": "#FF6F00", "stroke": "#FF6F00"})

    # ---- 右侧：ω_x-ω_y 相平面 ----

    # 坐标轴
    ax_len = 0.45
    objs.append({"type": "segment",
        "from": [PHASE_CX - ax_len, PHASE_CY],
        "to": [PHASE_CX + ax_len, PHASE_CY],
        "stroke": "#D0D0D0", "width": 1})
    objs.append({"type": "segment",
        "from": [PHASE_CX, PHASE_CY - ax_len],
        "to": [PHASE_CX, PHASE_CY + ax_len],
        "stroke": "#D0D0D0", "width": 1})

    # 绘制到当前步的轨迹（每 30 步一个点，限制线段数）
    end_idx = min(fi, len(omega_hist) - 1)
    trail_step = 30
    prev = None
    for ti in range(0, end_idx + 1, trail_step):
        wt = omega_hist[ti]
        px = PHASE_CX + wt[0] * phase_scale
        py = PHASE_CY + wt[1] * phase_scale
        if prev is not None:
            objs.append({"type": "segment",
                "from": prev, "to": [px, py],
                "stroke": "#7B1FA2", "width": 1.5})
        prev = [px, py]

    # 当前点
    cur_px = PHASE_CX + w[0] * phase_scale
    cur_py = PHASE_CY + w[1] * phase_scale
    objs.append({"type": "circle", "x": cur_px, "y": cur_py,
        "radius": 0.025, "fill": "#D32F2F", "stroke": "#D32F2F"})

    # 起点标记
    w0 = omega_hist[0]
    objs.append({"type": "circle",
        "x": PHASE_CX + w0[0] * phase_scale,
        "y": PHASE_CY + w0[1] * phase_scale,
        "radius": 0.025, "fill": "#388E3C", "stroke": "#388E3C"})

    # 分隔线
    objs.append({"type": "segment", "from": [0.05, -0.75], "to": [0.05, 0.75],
        "stroke": "#E0E0E0", "width": 1})

    t_sec = fi * dt

    frames.append({
        "objects": objs,
        "metrics": {
            "t": f"{t_sec:.2f} s",
            "ω_x": f"{w[0]:.4f}",
            "ω_y": f"{w[1]:.4f}",
            "ω_z": f"{w[2]:.4f}",
            "E": f"{E:.6f} J",
            "ΔE": f"{rel_err:+.4f}%",
        },
    })

print(f"frames: {len(frames)}")
print(f"objects/frame: ~{len(frames[len(frames)//2]['objects'])}")

simulation = {
    "bounds": [-1.3, -0.8, 1.35, 0.8],
    "fps": 30,
    "loop": True,
    "frames": frames,
}
