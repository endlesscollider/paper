# 两个相同小球从同一状态落地，只改变接触力的离散方式。
# 向上为正方向；球心低于 radius 时，penetration = radius - y 为正。
mass = 1.0
gravity = 9.81
radius = 0.25
stiffness = 20000.0
damping = 80.0
dt = 0.02
duration = 3.2
steps = int(duration / dt)
frame_stride = max(1, steps // 160)

explicit_y, explicit_v = 1.8, -1.0
implicit_y, implicit_v = 1.8, -1.0
frames = []
explicit_peak_speed = abs(explicit_v)
implicit_peak_speed = abs(implicit_v)
explicit_peak_force = 0.0
implicit_peak_force = 0.0


def explicit_step(y, v):
    # 显式 Euler 只读取时间步开始时刻的穿透量和速度。
    penetration = max(0.0, radius - y)
    contact_force = max(0.0, stiffness * penetration - damping * v) if penetration > 0.0 else 0.0

    # 力、速度和位置依次推进；本步内新出现的穿透要到下一步才被弹簧看见。
    acceleration = -gravity + contact_force / mass
    y_next = y + dt * v
    v_next = v + dt * acceleration
    return y_next, v_next, contact_force


def implicit_step(y, v):
    # 先检查若完全没有接触力，本步结束时是否会进入地面。
    v_free = v - gravity * dt
    y_free = y + dt * v_free
    if y_free >= radius:
        return y_free, v_free, 0.0

    # 接触成立时，把结束速度、结束位置和结束接触力联立起来。
    # denominator 不是调参，而是把三个方程消元后由求解器算出的系数。
    denominator = mass + damping * dt + stiffness * dt * dt
    v_next = (
        mass * v
        - mass * gravity * dt
        + stiffness * dt * (radius - y)
    ) / denominator
    y_next = y + dt * v_next
    contact_force = stiffness * (radius - y_next) - damping * v_next

    # 单边接触只能推、不能拉；若联立结果要求负力，就接受自由运动。
    if contact_force <= 0.0:
        return y_free, v_free, 0.0
    return y_next, v_next, contact_force


for step in range(steps):
    explicit_y, explicit_v, explicit_force = explicit_step(explicit_y, explicit_v)
    implicit_y, implicit_v, implicit_force = implicit_step(implicit_y, implicit_v)
    explicit_peak_speed = max(explicit_peak_speed, abs(explicit_v))
    implicit_peak_speed = max(implicit_peak_speed, abs(implicit_v))
    explicit_peak_force = max(explicit_peak_force, explicit_force)
    implicit_peak_force = max(implicit_peak_force, implicit_force)

    # 固定总时长后按比例抽帧；减小 dt 不会让实验在接触前提前结束。
    if step % frame_stride != 0:
        continue

    # 显式小球飞出坐标范围后，把红色标记钉在画布顶部，真实数值仍显示在指标中。
    explicit_out = explicit_y > 3.45 or explicit_y < -0.25
    explicit_draw_y = min(3.45, max(-0.15, explicit_y))
    frames.append({
        "time": step * dt,
        "objects": [
            {"type": "segment", "from": [-3.2, 0], "to": [-0.2, 0],
             "stroke": "#607d8b", "width": 3, "label": "显式：读取旧状态"},
            {"type": "segment", "from": [0.2, 0], "to": [3.2, 0],
             "stroke": "#607d8b", "width": 3, "label": "隐式：求解新状态"},
            {"type": "circle", "x": -1.7, "y": explicit_draw_y, "radius": radius,
             "fill": "#f44336", "stroke": "#b71c1c",
             "label": "显式：飞出范围" if explicit_out else "显式"},
            {"type": "circle", "x": 1.7, "y": implicit_y, "radius": radius,
             "fill": "#009688", "stroke": "#00695c", "label": "隐式"},
        ],
        "metrics": {
            "t": f"{step * dt:.2f} s",
            "explicit y": f"{explicit_y:.2f} m",
            "explicit v": f"{explicit_v:.2f} m/s",
            "explicit force": f"{explicit_force:.1f} N",
            "implicit y": f"{implicit_y:.3f} m",
            "implicit v": f"{implicit_v:.3f} m/s",
            "implicit force": f"{implicit_force:.1f} N",
        },
    })

print(f"mass = {mass:g} kg, stiffness = {stiffness:g} N/m, damping = {damping:g} N*s/m")
print(f"dt = {dt:g} s, stiffness*dt^2/mass = {stiffness * dt * dt / mass:.2f}")
print(f"explicit peak speed = {explicit_peak_speed:.3f} m/s")
print(f"implicit peak speed = {implicit_peak_speed:.3f} m/s")
print(f"explicit peak force = {explicit_peak_force:.1f} N")
print(f"implicit peak force = {implicit_peak_force:.1f} N")
print(f"explicit final y = {explicit_y:.3f} m")
print(f"implicit final y = {implicit_y:.3f} m")

# PhysicsPlayground 只重放上面已经计算完成的两套状态，不参与物理求解。
simulation = {"bounds": [-3.5, -0.5, 3.5, 4.0], "fps": 30, "loop": True, "frames": frames}
