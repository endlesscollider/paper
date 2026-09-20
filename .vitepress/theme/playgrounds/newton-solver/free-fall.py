from dataclasses import dataclass


@dataclass
class State:
    # 某一时刻的动态状态；竖直向下为负方向。
    position: float
    velocity: float


def step_semi_implicit(state, dt, gravity):
    # 半隐式 Euler：必须先更新速度，再用新速度更新位置。
    velocity_next = state.velocity + gravity * dt
    position_next = state.position + velocity_next * dt
    return State(position_next, velocity_next)


# 用 100 个固定时间步精确模拟 1.0 秒。
dt = 0.01
gravity = -9.81
state = State(position=5.0, velocity=0.0)
frames = []

for step in range(101):
    # 先记录当前 s_step，再决定是否推进到 s_(step+1)。
    if step % 2 == 0:
        frames.append({
            "time": step * dt,
            "objects": [
                {"type": "segment", "from": [-3, 0], "to": [3, 0], "stroke": "#64748b"},
                {"type": "circle", "x": 0, "y": state.position, "radius": 0.22,
                 "fill": "#2196f3", "stroke": "#1565c0"},
            ],
            "metrics": {"t": f"{step * dt:.2f} s", "y": f"{state.position:.3f} m",
                        "v": f"{state.velocity:.3f} m/s"},
        })

    # 最后一份样本已经是 t=1.0 秒，不能再多走一步到 1.01 秒。
    if step < 100:
        state = step_semi_implicit(state, dt, gravity)

# 把 t=1.0 秒的数值结果与连续解析解进行对照。
exact = 5.0 + 0.5 * gravity * 1.0 ** 2
print(f"semi-implicit y(1s) = {state.position:.6f} m")
print(f"analytic y(1s)      = {exact:.6f} m")

# PhysicsPlayground 只重放预先算好的帧，不参与积分。
simulation = {"bounds": [-3, -0.5, 3, 5.7], "fps": 30, "loop": True, "frames": frames}
