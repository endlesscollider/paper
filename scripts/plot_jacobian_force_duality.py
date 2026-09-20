"""Plot velocity-force ellipsoid duality and Jacobian-transpose torque mapping."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2196F3"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
GRAY = "#607D8B"


def jacobian(q1: float, q2: float, l1: float = 1.0, l2: float = 0.8) -> np.ndarray:
    return np.array([
        [-l1 * np.sin(q1) - l2 * np.sin(q1 + q2), -l2 * np.sin(q1 + q2)],
        [l1 * np.cos(q1) + l2 * np.cos(q1 + q2), l2 * np.cos(q1 + q2)],
    ])


def main() -> None:
    q1, q2 = np.deg2rad(25), np.deg2rad(65)
    j = jacobian(q1, q2)
    u, sigma, _ = np.linalg.svd(j)
    angle = np.linspace(0, 2 * np.pi, 360)
    unit = np.vstack([np.cos(angle), np.sin(angle)])
    velocity_ellipse = j @ unit
    force_ellipse = u @ np.diag(1 / sigma) @ unit

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(velocity_ellipse[0], velocity_ellipse[1], color=BLUE, linewidth=2.5,
            label="velocity ellipse: unit joint speed")
    ax.fill(velocity_ellipse[0], velocity_ellipse[1], color=BLUE, alpha=0.10)
    ax.plot(force_ellipse[0], force_ellipse[1], color=ORANGE, linewidth=2.5,
            label="force ellipse: unit joint torque")
    ax.fill(force_ellipse[0], force_ellipse[1], color=ORANGE, alpha=0.10)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set(xlabel="Task-space x direction", ylabel="Task-space y direction",
           title="Velocity and force capability are dual")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=8)

    force_angle = np.linspace(0, 2 * np.pi, 500)
    force = 20 * np.vstack([np.cos(force_angle), np.sin(force_angle)])
    torque = j.T @ force
    ax = axes[1]
    ax.plot(np.rad2deg(force_angle), torque[0], color=PURPLE, linewidth=2.2, label="joint torque tau1")
    ax.plot(np.rad2deg(force_angle), torque[1], color=GREEN, linewidth=2.2, label="joint torque tau2")
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set(xlim=(0, 360), xlabel="End-effector force direction (deg)", ylabel="Required torque (N m)",
           title="Force direction decides which joints carry the load")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=8)

    output = Path(__file__).resolve().parents[1] / "public/robot_jacobian_force_duality.png"
    fig.savefig(output, dpi=160, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
