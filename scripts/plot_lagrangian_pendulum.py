"""Plot potential-energy geometry and energy exchange for a pendulum."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2196F3"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
GRAY = "#607D8B"


def simulate(theta0: float, omega0: float, dt: float, duration: float):
    g, length = 9.81, 1.0
    steps = int(duration / dt)
    theta = np.empty(steps + 1)
    omega = np.empty(steps + 1)
    theta[0], omega[0] = theta0, omega0
    for i in range(steps):
        omega[i + 1] = omega[i] - dt * g / length * np.sin(theta[i])
        theta[i + 1] = theta[i] + dt * omega[i + 1]
    return np.arange(steps + 1) * dt, theta, omega


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    angles = np.linspace(-np.pi, np.pi, 500)
    potential = 9.81 * (1 - np.cos(angles))
    gravity_torque = -9.81 * np.sin(angles)
    ax = axes[0]
    ax.plot(np.rad2deg(angles), potential, color=PURPLE, linewidth=2.4, label="potential energy")
    ax.set(xlabel="Angle (deg)", ylabel="Potential energy (J)", title="Configuration creates an energy landscape")
    ax.grid(alpha=0.22)
    ax2 = ax.twinx()
    ax2.plot(np.rad2deg(angles), gravity_torque, color=ORANGE, linewidth=2.0, linestyle="--",
             label="gravity torque")
    ax2.set_ylabel("Gravity torque (N m)")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="upper center", fontsize=8)

    time, theta, omega = simulate(np.deg2rad(70), 0.0, 0.002, 8.0)
    kinetic = 0.5 * omega**2
    potential = 9.81 * (1 - np.cos(theta))
    total = kinetic + potential
    ax = axes[1]
    ax.plot(time, kinetic, color=BLUE, linewidth=2, label="kinetic T")
    ax.plot(time, potential, color=ORANGE, linewidth=2, label="potential V")
    ax.plot(time, total, color=GREEN, linewidth=2.3, linestyle="--", label="total T + V")
    ax.set(xlabel="Time (s)", ylabel="Energy (J)", title="Lagrangian dynamics exchanges T and V")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=8)

    output = Path(__file__).resolve().parents[1] / "public/robot_lagrangian_pendulum.png"
    fig.savefig(output, dpi=160, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
