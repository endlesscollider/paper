import math

# 这个紧凑示例只展示完整时间步顺序。
# 真正的 SAT 与顺序冲量实现位于 scripts/newton_solver_from_zero/rigid2d.py。
dt = 0.012
gravity = -9.81
restitution = 0.55
bodies = [
    {"x": -1.2, "y": 4.8, "vx": 1.0, "vy": 0.0, "angle": 0.1, "omega": 1.6,
     "w": 1.2, "h": 0.6, "color": "#2196f3"},
    {"x": 1.5, "y": 6.2, "vx": -0.5, "vy": -0.5, "angle": -0.2, "omega": -1.0,
     "w": 1.0, "h": 0.8, "color": "#ff9800"},
]
frames = []


def corners(body):
    # 把矩形顶点从刚体局部坐标变换到世界坐标。
    c, s = math.cos(body["angle"]), math.sin(body["angle"])
    local = [[-body["w"]/2, -body["h"]/2], [body["w"]/2, -body["h"]/2],
             [body["w"]/2, body["h"]/2], [-body["w"]/2, body["h"]/2]]
    return [[body["x"] + c*x - s*y, body["y"] + s*x + c*y] for x, y in local]


for step in range(650):
    contacts = 0
    for body in bodies:
        # 第 1 步：预测线性状态与角状态。
        body["vy"] += gravity * dt
        body["x"] += body["vx"] * dt
        body["y"] += body["vy"] * dt
        body["angle"] += body["omega"] * dt

        # 第 2 步：计算当前旋转矩形在竖直方向上的准确支撑高度。
        c, s = math.cos(body["angle"]), math.sin(body["angle"])
        support_y = 0.5 * (abs(s) * body["w"] + abs(c) * body["h"])
        if body["y"] < support_y:
            # 第 3 步：简化地面响应，包括位置投影、法向反弹、切向与角速度衰减。
            # 这个网页示例不做两个动态物体之间的 SAT；完整实现见 rigid2d.py。
            body["y"] = support_y
            if body["vy"] < 0:
                body["vy"] = -restitution * body["vy"]
                body["vx"] *= 0.82
                body["omega"] *= 0.72
                contacts += 1

    # 第 4 步：所有刚体完成本时间步后，才构造可视化帧。
    if step % 3 == 0:
        objects = [{"type": "segment", "from": [-5, 0], "to": [5, 0],
                    "stroke": "#475569", "width": 3}]
        for index, body in enumerate(bodies):
            objects.append({"type": "polygon", "points": corners(body), "fill": body["color"],
                            "stroke": "#334155", "label": f"body {index}"})
        frames.append({"objects": objects,
                       "metrics": {"step": step, "contacts": contacts,
                                   "solver": "simplified ground demo"}})

print("final states")
for index, body in enumerate(bodies):
    print(index, {key: round(value, 3) for key, value in body.items() if isinstance(value, float)})

# PhysicsPlayground 是播放器；上面的循环已经完成全部状态计算。
simulation = {"bounds": [-5, -0.5, 5, 7.2], "fps": 60, "loop": True, "frames": frames}
