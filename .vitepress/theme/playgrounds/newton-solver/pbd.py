import math

# 七个粒子组成一条绳子；0 号点逆质量为 0，因此完全固定。
dt = 0.016
gravity = -9.81
rest_length = 0.8
iterations = 8
points = [[-2.0 + i * rest_length, 4.8] for i in range(7)]
previous = [p[:] for p in points]
inverse_mass = [0.0] + [1.0] * 6
frames = []

for step in range(260):
    # 第 1 步：用当前与上一位置之差重建速度，再预测自由位置。
    for i in range(1, len(points)):
        vx = (points[i][0] - previous[i][0]) / dt
        vy = (points[i][1] - previous[i][1]) / dt + gravity * dt
        previous[i] = points[i][:]
        points[i][0] += vx * dt
        points[i][1] += vy * dt

    # 第 2 步：反复把每对相邻粒子的距离投影回静止长度。
    for _ in range(iterations):
        for i in range(len(points) - 1):
            a, b = points[i], points[i + 1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            distance = max(1e-9, math.hypot(dx, dy))
            error = distance - rest_length
            total_weight = inverse_mass[i] + inverse_mass[i + 1]

            # 归一化边方向就是约束梯度方向；逆质量决定两端各承担多少校正。
            correction_x = error * dx / distance / total_weight
            correction_y = error * dy / distance / total_weight
            a[0] += inverse_mass[i] * correction_x
            a[1] += inverse_mass[i] * correction_y
            b[0] -= inverse_mass[i + 1] * correction_x
            b[1] -= inverse_mass[i + 1] * correction_y

    # 必须在全部迭代结束后测量剩余误差，不能误报迭代前的误差。
    max_error = 0.0
    for i in range(len(points) - 1):
        max_error = max(max_error, abs(math.dist(points[i], points[i + 1]) - rest_length))

    # 第 3 步：把校正后的粒子位置转换成可绘制对象。
    if step % 2 == 0:
        objects = []
        for i in range(len(points) - 1):
            objects.append({"type": "segment", "from": points[i][:], "to": points[i + 1][:],
                            "stroke": "#64748b", "width": 3})
        for i, (x, y) in enumerate(points):
            objects.append({"type": "circle", "x": x, "y": y, "radius": 0.12,
                            "fill": "#f44336" if i == 0 else "#4caf50", "stroke": "#334155"})
        frames.append({"objects": objects, "metrics": {"iteration": iterations,
                      "max error": f"{max_error:.4f}"}})

print(f"constraint iterations = {iterations}")
print(f"last max error         = {max_error:.6f}")

# PhysicsPlayground 只重放 frames，不参与约束求解。
simulation = {"bounds": [-3, -1, 4, 5.5], "fps": 45, "loop": True, "frames": frames}
