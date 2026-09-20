import math

# 一个运动粒子通过具有顺应性的距离约束连接到固定锚点。
dt = 0.016
gravity = -9.81
rest_length = 2.3
compliance = 0.0005
iterations = 8
anchor = [0.0, 4.8]
position = [1.6, 2.8]
previous = position[:]
frames = []

for step in range(300):
    # 第 1 步：用 Verlet 风格的速度估计预测自由位置。
    vx = (position[0] - previous[0]) / dt
    vy = (position[1] - previous[1]) / dt + gravity * dt
    previous = position[:]
    position[0] += vx * dt
    position[1] += vy * dt

    # 第 2 步：lambda 在本时间步的多轮求解迭代之间累计。
    # 进入新时间步后，开始一次新的离散约束求解。
    lagrange = 0.0
    alpha = compliance / (dt * dt)
    for _ in range(iterations):
        dx, dy = position[0] - anchor[0], position[1] - anchor[1]
        distance = max(1e-9, math.hypot(dx, dy))
        constraint = distance - rest_length

        # XPBD 在 PBD 位置投影中加入顺应性和累计乘子，
        # 让材料软硬程度不再过度依赖 dt。
        delta_lambda = (-constraint - alpha * lagrange) / (1.0 + alpha)
        lagrange += delta_lambda
        position[0] += delta_lambda * dx / distance
        position[1] += delta_lambda * dy / distance

    # 第 3 步：约束求解完成后才保存可视化数据。
    if step % 2 == 0:
        distance = math.dist(anchor, position)
        frames.append({
            "objects": [
                {"type": "segment", "from": anchor, "to": position[:], "stroke": "#9c27b0", "width": 3},
                {"type": "circle", "x": anchor[0], "y": anchor[1], "radius": 0.13,
                 "fill": "#f44336", "stroke": "#334155"},
                {"type": "circle", "x": position[0], "y": position[1], "radius": 0.24,
                 "fill": "#9c27b0", "stroke": "#6a1b9a"},
            ],
            "metrics": {"length": f"{distance:.3f}", "lambda": f"{lagrange:.4f}",
                        "compliance": compliance},
        })

print(f"rest length  = {rest_length}")
print(f"final length = {math.dist(anchor, position):.6f}")

# Python 结束后，页面读取这个变量并重放动画。
simulation = {"bounds": [-3, -0.5, 3, 5.4], "fps": 45, "loop": True, "frames": frames}
