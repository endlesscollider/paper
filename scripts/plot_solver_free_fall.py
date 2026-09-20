"""Generate the first chapter's free-fall integration comparison figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate(method: str, dt: float, duration: float, x0: float, v0: float, gravity: float):
    steps = int(round(duration / dt))
    times = np.arange(steps + 1) * dt
    positions = np.empty(steps + 1)
    velocities = np.empty(steps + 1)
    positions[0] = x0
    velocities[0] = v0

    for i in range(steps):
        if method == "explicit":
            positions[i + 1] = positions[i] + velocities[i] * dt
            velocities[i + 1] = velocities[i] + gravity * dt
        elif method == "semi-implicit":
            velocities[i + 1] = velocities[i] + gravity * dt
            positions[i + 1] = positions[i] + velocities[i + 1] * dt
        else:
            raise ValueError(f"unknown method: {method}")
    return times, positions


def main() -> None:
    x0, v0, gravity, duration, dt = 5.0, 0.0, -9.81, 1.0, 0.1
    times = np.arange(int(round(duration / dt)) + 1) * dt
    exact = x0 + v0 * times + 0.5 * gravity * times**2
    _, explicit = simulate("explicit", dt, duration, x0, v0, gravity)
    _, semi_implicit = simulate("semi-implicit", dt, duration, x0, v0, gravity)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    axes[0].plot(times, exact, color="#607D8B", linewidth=2.2, label="Analytical")
    axes[0].plot(times, explicit, "o--", color="#F44336", linewidth=1.8, markersize=4, label="Explicit Euler")
    axes[0].plot(
        times,
        semi_implicit,
        "s-",
        color="#2196F3",
        linewidth=1.8,
        markersize=4,
        label="Semi-implicit Euler",
    )
    axes[0].set_title("Free-fall position")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Height (m)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    explicit_error = np.abs(explicit - exact)
    semi_error = np.abs(semi_implicit - exact)
    axes[1].plot(times, explicit_error, "o--", color="#F44336", linewidth=1.8, markersize=4, label="Explicit Euler")
    axes[1].plot(
        times,
        semi_error,
        "s-",
        color="#2196F3",
        linewidth=1.8,
        markersize=4,
        label="Semi-implicit Euler",
    )
    axes[1].set_title("Absolute position error")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (m)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    output = Path(__file__).resolve().parents[1] / "public/solver_free_fall_integrator_comparison.png"
    fig.savefig(output, dpi=150)
    print(output)


if __name__ == "__main__":
    main()
