# 单位质量方块：分别计算法向反弹与切向摩擦两个标量响应。
dt = 0.01
gravity = -9.81
restitution = 0.65
friction = 0.35
x, y = -4.0, 4.0
vx, vy = 4.0, 0.0
half_w, half_h = 0.6, 0.35
frames = []

for step in range(600):
    # 第 1 步：在没有地面约束时预测自由平动。
    vy += gravity * dt
    x += vx * dt
    y += vy * dt
    normal_impulse = 0.0
    friction_impulse = 0.0

    # 第 2 步：清除地面穿透，并用法向冲量修正竖直速度。
    if y < half_h:
        y = half_h
        if vy < 0:
            normal_impulse = -(1.0 + restitution) * vy
            vy += normal_impulse

            # 第 3 步：库仑摩擦把切向冲量限制在 mu * Jn 以内。
            friction_impulse = min(abs(vx), friction * normal_impulse)
            vx -= friction_impulse if vx > 0 else -friction_impulse

    # 本程序只隔离线速度；偏心摩擦产生的角速度由正文和完整程序讲解。
    if step % 3 == 0:
        corners = [[x - half_w, y - half_h], [x + half_w, y - half_h],
                   [x + half_w, y + half_h], [x - half_w, y + half_h]]
        frames.append({
            "objects": [
                {"type": "segment", "from": [-5, 0], "to": [8, 0], "stroke": "#475569", "width": 3},
                {"type": "polygon", "points": corners, "fill": "#009688", "stroke": "#00695c"},
            ],
            "metrics": {"vx": f"{vx:.2f}", "vy": f"{vy:.2f}",
                        "normal J": f"{normal_impulse:.2f}",
                        "friction J": f"{friction_impulse:.2f}"},
        })

print(f"final horizontal speed = {vx:.4f}")

# 浏览器只重放这些已经完成求解的帧。
simulation = {"bounds": [-5, -0.5, 8, 5], "fps": 60, "loop": True, "frames": frames}
