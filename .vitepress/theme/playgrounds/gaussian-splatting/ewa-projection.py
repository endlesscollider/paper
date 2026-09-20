import math

# 演示 EWA splatting：一个三维高斯(固定的三维协方差)在不同深度、不同朝向下，
# 投影到屏幕上的二维椭圆形状如何变化。这里手动实现 W(旋转)和 J(雅可比)两步
# 变换，对应第 6 章公式 Sigma_2D = J W Sigma_3D W^T J^T。

fx = fy = 1000.0  # 焦距（像素）

# 三维协方差（相机自身坐标系下，暂不旋转）：一个略微拉长的椭球
sx, sy, sz = 0.03, 0.03, 0.08  # 三个方向的标准差（米），沿 z 方向更长
Sigma3d_local = [[sx*sx, 0, 0], [0, sy*sy, 0], [0, 0, sz*sz]]


def rotation_matrix_y(theta):
    # 绕世界 Y 轴（竖直轴）转动，模拟"高斯朝向"变化
    c, s = math.cos(theta), math.sin(theta)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            if A[i][k] == 0:
                continue
            for j in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def jacobian(X0, Y0, Z0):
    # 第5章推出的透视投影雅可比矩阵（2x3）
    return [
        [fx / Z0, 0, -fx * X0 / (Z0 ** 2)],
        [0, fy / Z0, -fy * Y0 / (Z0 ** 2)],
    ]


def ellipse_from_cov2d(Sigma2d):
    # 从 2x2 协方差矩阵反解出椭圆的两条半轴长度和旋转角，供渲染器画图使用。
    a, b, c, d = Sigma2d[0][0], Sigma2d[0][1], Sigma2d[1][0], Sigma2d[1][1]
    # 特征值：对 2x2 对称矩阵直接用公式求解
    tr = a + d
    det = a * d - b * c
    disc = max(0.0, (tr / 2) ** 2 - det)
    root = math.sqrt(disc)
    lam1 = tr / 2 + root
    lam2 = tr / 2 - root
    # 主轴角度：atan2(lam1 - a, b)，即特征向量方向
    if b != 0:
        angle = math.atan2(lam1 - a, b)
    else:
        angle = 0.0 if a >= d else math.pi / 2
    r1 = math.sqrt(max(lam1, 1e-9))
    r2 = math.sqrt(max(lam2, 1e-9))
    return r1, r2, angle


frames = []
n_frames = 72
for i in range(n_frames):
    t = i / n_frames
    # 深度在 [1.0, 3.0] 米之间来回摆动
    depth = 2.0 + 1.0 * math.sin(2 * math.pi * t)
    # 朝向角在 [-60°, 60°] 之间来回摆动
    theta = math.radians(60 * math.sin(2 * math.pi * t + 1.0))

    X0, Y0, Z0 = 0.0, 0.0, depth
    W = rotation_matrix_y(theta)
    Sigma_cam = matmul(matmul(W, Sigma3d_local), transpose(W))
    J = jacobian(X0, Y0, Z0)
    Sigma2d = matmul(matmul(J, Sigma_cam), transpose(J))

    r1_m, r2_m, angle = ellipse_from_cov2d(Sigma2d)
    # 转换成用于渲染器的“世界单位”坐标（这里简单地把像素单位缩小显示，
    # 保持画面比例合适；量级不追求物理精确，只用于教学演示相对变化）
    scale_to_view = 1.0 / 250.0
    r1 = r1_m * scale_to_view
    r2 = r2_m * scale_to_view

    objects = [
        {
            "type": "splat",
            "x": 0.0, "y": 0.0,
            "radiusA": max(r1, 0.05), "radiusB": max(r2, 0.05),
            "angle": angle,
            "depth": 1.0,
            "color": "#2196f3",
            "alpha": 0.75,
            "label": "屏幕椭圆 Σ2D",
        },
    ]

    frames.append({
        "objects": objects,
        "metrics": {
            "深度Z0(米)": round(depth, 2),
            "朝向角(度)": round(math.degrees(theta), 1),
            "长轴半径(像素)": round(r1_m, 1),
            "短轴半径(像素)": round(r2_m, 1),
        },
    })

simulation = {
    "bounds": [-2.5, -2.5, 2.5, 2.5],
    "fps": 24,
    "loop": True,
    "frames": frames,
}

print("三维协方差固定为标准差 (0.03, 0.03, 0.08) 米的拉长椭球。")
print("深度在 1~3 米间摆动，朝向在 -60°~60° 间摆动，观察屏幕椭圆的大小和倾斜角如何随之变化。")
