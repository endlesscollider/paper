import math

# 定义矩形质量属性，以及绕固定质心旋转的动态状态。
dt = 0.016
mass = 2.0
width, height = 2.2, 1.0
inertia = mass * (width * width + height * height) / 12.0
angle = 0.0
angular_velocity = 0.0
torque = 3.0
frames = []


def world_corners(cx, cy, theta):
    # 先旋转每个局部顶点，再平移到世界坐标。
    c, s = math.cos(theta), math.sin(theta)
    local = [[-width / 2, -height / 2], [width / 2, -height / 2],
             [width / 2, height / 2], [-width / 2, height / 2]]
    return [[cx + c * x - s * y, cy + s * x + c * y] for x, y in local]


for step in range(300):
    # 第 1 步：前 60 步施加力矩，之后撤掉力矩让刚体自由转动。
    angular_acceleration = torque / inertia if step < 60 else 0.0

    # 第 2 步：半隐式 Euler 先更新角速度 omega，再更新角度。
    angular_velocity += angular_acceleration * dt
    angle += angular_velocity * dt

    # 第 3 步：用更新后的角度重新生成世界坐标顶点，供渲染使用。
    if step % 2 == 0:
        frames.append({
            "objects": [
                {"type": "polygon", "points": world_corners(0, 2.4, angle),
                 "fill": "#9c27b0", "stroke": "#6a1b9a", "label": "rigid body"},
                {"type": "segment", "from": [0, 2.4],
                 "to": [1.2 * math.cos(angle), 2.4 + 1.2 * math.sin(angle)],
                 "stroke": "#f44336", "width": 3},
            ],
            "metrics": {"angle": f"{angle:.2f} rad", "omega": f"{angular_velocity:.2f}",
                        "inertia": f"{inertia:.2f}"},
        })

print(f"moment of inertia = {inertia:.6f}")
print(f"angular velocity  = {angular_velocity:.6f}")

# PhysicsPlayground 只重放已经变换好的矩形顶点。
simulation = {"bounds": [-3, -0.5, 3, 5], "fps": 45, "loop": True, "frames": frames}
