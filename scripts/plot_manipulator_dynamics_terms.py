"""Plot torque-term decomposition and configuration-dependent inertia for a 2R arm."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2196F3"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
GRAY = "#607D8B"

L1, LC1, LC2 = 0.8, 0.4, 0.3
M1, M2 = 1.5, 1.0
I1, I2 = 0.08, 0.04
GRAVITY = 9.81


def mass_matrix(q2):
    m11 = I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * np.cos(q2))
    m12 = I2 + M2 * (LC2**2 + L1 * LC2 * np.cos(q2))
    m22 = np.full_like(np.asarray(q2), I2 + M2 * LC2**2, dtype=float)
    return m11, m12, m22


def main() -> None:
    time = np.linspace(0, 6, 600)
    q1 = 0.45 * np.sin(1.2 * time)
    q2 = 0.7 * np.sin(1.2 * time + 0.6)
    qd1 = 0.45 * 1.2 * np.cos(1.2 * time)
    qd2 = 0.7 * 1.2 * np.cos(1.2 * time + 0.6)
    qdd1 = -0.45 * 1.2**2 * np.sin(1.2 * time)
    qdd2 = -0.7 * 1.2**2 * np.sin(1.2 * time + 0.6)

    m11, m12, m22 = mass_matrix(q2)
    inertia = m11 * qdd1 + m12 * qdd2
    h = M2 * L1 * LC2 * np.sin(q2)
    velocity = -h * (2 * qd1 * qd2 + qd2**2)
    gravity = (M1 * LC1 + M2 * L1) * GRAVITY * np.cos(q1) + M2 * LC2 * GRAVITY * np.cos(q1 + q2)
    total = inertia + velocity + gravity

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(time, inertia, color=BLUE, linewidth=2, label="inertia M(q) qdd")
    ax.plot(time, velocity, color=PURPLE, linewidth=2, label="velocity C(q, qd) qd")
    ax.plot(time, gravity, color=ORANGE, linewidth=2, label="gravity g(q)")
    ax.plot(time, total, color="#263238", linewidth=2.4, linestyle="--", label="required torque")
    ax.set(xlabel="Time (s)", ylabel="Joint-1 torque (N m)", title="One command is the sum of physical causes")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8)

    q2_sweep = np.linspace(-np.pi, np.pi, 500)
    m11, m12, m22 = mass_matrix(q2_sweep)
    ax = axes[1]
    ax.plot(np.rad2deg(q2_sweep), m11, color=BLUE, linewidth=2.2, label="M11")
    ax.plot(np.rad2deg(q2_sweep), m12, color=GREEN, linewidth=2.2, label="M12 coupling")
    ax.plot(np.rad2deg(q2_sweep), m22, color=ORANGE, linewidth=2.2, label="M22")
    ax.axvline(0, color=GRAY, linewidth=1.2, linestyle=":")
    ax.set(xlabel="Elbow angle q2 (deg)", ylabel="Inertia coefficient", title="The mass matrix changes with configuration")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=8)

    output = Path(__file__).resolve().parents[1] / "public/robot_manipulator_dynamics_terms.png"
    fig.savefig(output, dpi=160, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
