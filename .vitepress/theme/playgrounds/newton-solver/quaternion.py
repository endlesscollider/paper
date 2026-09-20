import math

# 使用恒定世界系角速度；初始姿态是单位四元数 [w, x, y, z]。
dt = 0.016
omega = [0.7, 1.1, 0.35]
q = [1.0, 0.0, 0.0, 0.0]
frames = []


def multiply(a, b):
    # Hamilton 乘积；四元数乘法不可交换，因此左右顺序不能随意改变。
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw]


def rotate(vector):
    # 使用 q * [0, v] * conjugate(q) 旋转三维向量。
    conjugate = [q[0], -q[1], -q[2], -q[3]]
    return multiply(multiply(q, [0.0] + vector), conjugate)[1:]


def project(vector):
    # 把三维点做简单斜投影，只用于二维画布显示。
    x, y, z = vector
    return [x + 0.45 * z, y + 0.25 * z]


for step in range(320):
    # 第 1 步：用显式 Euler 积分 q_dot = 0.5 * [0, omega] * q。
    derivative = multiply([0.0] + omega, q)
    q = [q[i] + 0.5 * dt * derivative[i] for i in range(4)]

    # 第 2 步：Euler 积分会偏离单位球，因此每一步都必须归一化。
    norm = math.sqrt(sum(value * value for value in q))
    q = [value / norm for value in q]

    # 第 3 步：旋转刚体局部 X/Y/Z 轴，再投影到二维画布。
    if step % 2 == 0:
        axes = [("X", [1.6, 0, 0], "#f44336"),
                ("Y", [0, 1.6, 0], "#4caf50"),
                ("Z", [0, 0, 1.6], "#2196f3")]
        objects = []
        for label, axis, color in axes:
            end = project(rotate(axis))
            objects.append({"type": "segment", "from": [0, 0], "to": end,
                            "stroke": color, "width": 4, "label": label})
        objects.append({"type": "circle", "x": 0, "y": 0, "radius": 0.09,
                        "fill": "#334155", "stroke": "#334155"})
        frames.append({"objects": objects,
                       "metrics": {"|q|": f"{norm:.6f}", "qw": f"{q[0]:.3f}", "qxyz": f"{q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}"}})

print("normalized quaternion:", [round(value, 6) for value in q])

# 画面显示的 |q| 是该步归一化之前的模长，用于观察数值漂移。
simulation = {"bounds": [-2.5, -2.5, 2.5, 2.5], "fps": 45, "loop": True, "frames": frames}
