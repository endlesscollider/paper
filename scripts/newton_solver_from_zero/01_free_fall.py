"""The smallest useful rigid-body solver: one particle in gravity.

This script intentionally uses only the Python standard library so the
integration loop is visible before introducing a physics-engine API.
"""

from dataclasses import dataclass


@dataclass
class State:
    position: float
    velocity: float


def step_semi_implicit(state: State, dt: float, gravity: float) -> State:
    """Advance one step with velocity-first Euler integration."""
    acceleration = gravity
    velocity_next = state.velocity + acceleration * dt
    position_next = state.position + velocity_next * dt
    return State(position_next, velocity_next)


def simulate(initial: State, dt: float, gravity: float, steps: int) -> list[State]:
    states = [initial]
    state = initial
    for _ in range(steps):
        state = step_semi_implicit(state, dt, gravity)
        states.append(state)
    return states


if __name__ == "__main__":
    dt = 0.01
    gravity = -9.81
    states = simulate(State(position=5.0, velocity=0.0), dt, gravity, steps=100)
    final = states[-1]
    time = dt * (len(states) - 1)
    exact_position = 5.0 + 0.5 * gravity * time**2
    exact_velocity = gravity * time
    print(f"t={time:.2f}s")
    print(f"semi-implicit position={final.position:.6f}, exact={exact_position:.6f}")
    print(f"semi-implicit velocity={final.velocity:.6f}, exact={exact_velocity:.6f}")
