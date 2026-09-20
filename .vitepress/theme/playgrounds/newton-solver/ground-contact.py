# 这个小程序只研究竖直方向的“小球撞地”，暂时不处理摩擦和旋转。
dt = 0.01             # 固定物理时间步：每次循环推进 0.01 秒
gravity = -9.81       # 重力加速度：负号表示竖直向下
restitution = 0.72    # 恢复系数：反弹速度保留入射速度的 72%
radius = 0.28         # 小球半径；球心最低只能到 y = radius
y = 4.5               # 当前球心高度
v = 0.0               # 当前竖直速度；负数向下，正数向上
frames = []           # 保存给网页播放器的动画帧，不参与物理计算
contacts = 0          # 统计实际发生了多少次撞地反弹

for step in range(500):
    # 第 1 步：预测自由运动。半隐式 Euler 必须先更新速度，再更新位置。
    v += gravity * dt
    y += v * dt

    # 第 2 步：检测穿透。若球心低于一个半径，球底已经进入 y = 0 以下。
    touching = y < radius
    if touching:
        # 第 3 步：位置校正。把球心放回 y = radius，清除已有穿透。
        y = radius
        if v < 0:
            # 第 4 步：速度校正。只有仍在向下运动时才产生反弹。
            # 例如入射速度为 -5，校正后为 -0.72 * (-5) = +3.6。
            v = -restitution * v
            contacts += 1

    # 每 3 个物理步保存 1 帧，减少动画数据量；物理仍然每步都计算。
    if step % 3 == 0:
        frames.append({
            "objects": [
                # segment 只负责画地面，不参与上面的碰撞判断。
                {"type": "segment", "from": [-3, 0], "to": [3, 0], "stroke": "#475569", "width": 3},
                # circle 使用校正后的 y；接触帧显示橙色，其他帧显示蓝色。
                {"type": "circle", "x": 0, "y": y, "radius": radius,
                 "fill": "#ff9800" if touching else "#2196f3", "stroke": "#334155"},
            ],
            # metrics 是右上角数值面板，同样不参与物理计算。
            "metrics": {"height": f"{y:.2f}", "velocity": f"{v:.2f}", "contacts": contacts},
        })

# 文本输出用于检查最终结果；切换到“输出”标签即可看到。
print(f"contact count = {contacts}")
print(f"final speed   = {v:.4f}")

# Python 运行结束后，PhysicsPlayground 读取 simulation 并重放 frames。
# bounds 是画布坐标范围，fps 是播放帧率，loop=True 表示循环播放。
simulation = {"bounds": [-3, -0.5, 3, 5.2], "fps": 45, "loop": True, "frames": frames}
