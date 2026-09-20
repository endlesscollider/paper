# 固定时间步参数与小球初始状态。
dt = 0.02
gravity = -9.81
radius = 0.25
position = 4.5
velocity = 0.0
frames = []

for step in range(180):
    # 第 1 步：预测自由运动。此时故意忽略地面约束。
    velocity_pred = velocity + gravity * dt
    position_pred = position + velocity_pred * dt

    # 第 2 步：检测预测球底是否越过 y=0。
    touching = position_pred < radius

    if touching:
        # 第 3a 步：位置校正，清除已经发生的几何穿透。
        position_next = radius

        # 第 3b 步：速度校正，删除向地面内部运动的法向速度。
        # 本例恢复系数为 0，因此小球停止而不是反弹。
        velocity_next = max(0.0, velocity_pred)
    else:
        # 预测状态合法时，位置和速度都可以直接接受。
        position_next = position_pred
        velocity_next = velocity_pred

    # 第 4 步：构造动画帧。下面的字典只负责显示，不参与物理计算。
    objects = [
        {"type": "segment", "from": [-3, 0], "to": [3, 0], "stroke": "#475569", "width": 3},
        {"type": "circle", "x": 0, "y": position_next, "radius": radius,
         "fill": "#4caf50", "stroke": "#2e7d32", "label": "corrected"},
    ]
    if touching:
        objects.append({"type": "circle", "x": 0.7, "y": position_pred, "radius": radius,
                        "fill": "#fca5a5", "stroke": "#dc2626", "label": "predicted"})

    # 每两个物理步保存一帧，让循环预览更短、数据量更小。
    if step % 2 == 0:
        frames.append({
            "objects": objects,
            "metrics": {
                "contact": "yes" if touching else "no",
                "y predicted": f"{position_pred:.3f}",
                "y corrected": f"{position_next:.3f}",
                "v predicted": f"{velocity_pred:.3f}",
                "v corrected": f"{velocity_next:.3f}",
            },
        })

    # 校正后的最终状态会成为下一固定时间步的输入。
    position, velocity = position_next, velocity_next

print(f"final position = {position:.3f}")
print(f"final velocity = {velocity:.3f}")

# Python 结束后 PhysicsPlayground 才读取 simulation，并在 Canvas 上重放。
# 真正的物理计算已经由上面的 Python 循环全部完成。
simulation = {"bounds": [-3, -0.7, 3, 5.2], "fps": 30, "loop": True, "frames": frames}
