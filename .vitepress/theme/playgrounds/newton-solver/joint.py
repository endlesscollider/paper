import math

# 一个单位质量粒子，通过距离关节连接到固定锚点。
dt = 0.012
gravity = -9.81
rest_length = 2.5
anchor = [0.0, 4.8]
position = [2.0, 3.2]
velocity = [0.0, 0.0]
frames = []

for step in range(520):
    # 第 1 步：在忽略关节时预测重力下的自由运动。
    velocity[1] += gravity * dt
    position[0] += velocity[0] * dt
    position[1] += velocity[1] * dt

    # 第 2 步：沿“锚点到粒子”方向构造一条标量约束行。
    dx, dy = position[0] - anchor[0], position[1] - anchor[1]
    distance = max(1e-9, math.hypot(dx, dy))
    nx, ny = dx / distance, dy / distance
    error = distance - rest_length
    radial_speed = velocity[0] * nx + velocity[1] * ny

    # 第 3 步：单位逆质量使有效质量 K=1。bias 消除一部分位置漂移，
    # joint_impulse 则负责取消沿绳方向的径向速度。
    bias = 0.2 * error / dt
    joint_impulse = -(radial_speed + bias)
    velocity[0] += joint_impulse * nx
    velocity[1] += joint_impulse * ny

    # 第 4 步：再做一次独立位置清理，让距离严格回到目标长度。
    position[0] -= error * nx
    position[1] -= error * ny

    # 关节不删除切向速度，因此粒子仍会形成摆动。
    if step % 3 == 0:
        frames.append({
            "objects": [
                {"type": "segment", "from": anchor, "to": position[:], "stroke": "#607d8b", "width": 3},
                {"type": "circle", "x": anchor[0], "y": anchor[1], "radius": 0.14,
                 "fill": "#f44336", "stroke": "#334155", "label": "anchor"},
                {"type": "circle", "x": position[0], "y": position[1], "radius": 0.32,
                 "fill": "#ff9800", "stroke": "#e65100", "label": "body"},
            ],
            "metrics": {"length": f"{math.dist(anchor, position):.4f}",
                        "C(x)": f"{math.dist(anchor, position) - rest_length:+.5f}",
                        "radial v": f"{radial_speed:+.3f}",
                        "joint J": f"{joint_impulse:+.3f}"},
        })

print(f"joint error = {math.dist(anchor, position) - rest_length:+.8f}")

# 速度与位置求解完成后，PhysicsPlayground 才负责显示帧。
simulation = {"bounds": [-3.5, -0.2, 3.5, 5.5], "fps": 60, "loop": True, "frames": frames}
