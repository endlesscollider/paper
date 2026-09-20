# 用一维双物体碰撞单独观察质量与冲量分配。
dt = 0.01
restitution = 0.85
radius_a = 0.45
radius_b = 0.65
mass_a = 1.0
mass_b = 2.5
x_a, v_a = -4.2, 3.2
x_b, v_b = 2.3, -0.6
frames = []
last_collision_impulse = 0.0

for step in range(420):
    # 第 1 步：分别预测两个圆的圆心位置。
    x_a += v_a * dt
    x_b += v_b * dt
    distance = x_b - x_a
    touching = distance < radius_a + radius_b
    applied_impulse = 0.0

    # 第 2 步：只有“已经重叠且仍在靠近”时才求碰撞冲量。
    if touching and v_b - v_a < 0:
        relative_velocity = v_b - v_a
        applied_impulse = -(1.0 + restitution) * relative_velocity / (1.0 / mass_a + 1.0 / mass_b)

        # 两边收到大小相等、方向相反的冲量；速度变化量为 J / mass。
        v_a -= applied_impulse / mass_a
        v_b += applied_impulse / mass_b
        last_collision_impulse = applied_impulse

        # 位置校正按逆质量分摊穿透，轻物体移动得更多。
        penetration = radius_a + radius_b - distance
        x_a -= penetration * mass_b / (mass_a + mass_b)
        x_b += penetration * mass_a / (mass_a + mass_b)

    # applied J 为 0 表示这一帧没有真正施加碰撞冲量。
    if step % 2 == 0:
        frames.append({
            "objects": [
                {"type": "segment", "from": [-5, 0], "to": [5, 0], "stroke": "#94a3b8"},
                {"type": "circle", "x": x_a, "y": radius_a, "radius": radius_a,
                 "fill": "#2196f3", "stroke": "#1565c0", "label": "m=1"},
                {"type": "circle", "x": x_b, "y": radius_b, "radius": radius_b,
                 "fill": "#ff9800", "stroke": "#e65100", "label": "m=2.5"},
            ],
            "metrics": {"vA": f"{v_a:.2f}", "vB": f"{v_b:.2f}",
                        "applied J": f"{applied_impulse:.2f}"},
        })

print(f"v_a = {v_a:.4f}")
print(f"v_b = {v_b:.4f}")
print(f"collision impulse = {last_collision_impulse:.4f}")

# PhysicsPlayground 只显示已经求解好的位置与指标。
simulation = {"bounds": [-5.5, -0.5, 5.5, 3.6], "fps": 60, "loop": True, "frames": frames}
