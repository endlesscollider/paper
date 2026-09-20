"""Plot velocity manipulability ellipses and singular values for a 2R arm."""

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


def draw_arm_and_ellipse(ax, q1: float, q2: float, color: str, label: str) -> None:
    l1, l2 = 1.0, 0.8
    elbow = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    end = elbow + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    ax.plot([0, elbow[0], end[0]], [0, elbow[1], end[1]], "-o", color=color,
            linewidth=3, markersize=7, markerfacecolor="white", markeredgewidth=2)
    circle = np.vstack([np.cos(np.linspace(0, 2 * np.pi, 240)), np.sin(np.linspace(0, 2 * np.pi, 240))])
    ellipse = 0.33 * jacobian(q1, q2) @ circle
    ax.plot(end[0] + ellipse[0], end[1] + ellipse[1], color=color, linewidth=2.2)
    ax.fill(end[0] + ellipse[0], end[1] + ellipse[1], color=color, alpha=0.12)
    ax.text(end[0] + 0.05, end[1] + 0.18, label, color=color, fontsize=9, weight="bold")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), constrained_layout=True)
    ax = axes[0]
    draw_arm_and_ellipse(ax, np.deg2rad(25), np.deg2rad(80), BLUE, "bent: 2D motion")
    draw_arm_and_ellipse(ax, np.deg2rad(-18), np.deg2rad(4), ORANGE, "near singular: flattened")
    ax.scatter([0], [0], marker="s", s=80, color="#263238", zorder=5)
    ax.set(xlim=(-0.25, 2.55), ylim=(-1.25, 2.0), xlabel="x (m)", ylabel="y (m)",
           title="Unit joint-speed circle mapped into task space")
    ax.set_aspect("equal")
    ax.grid(alpha=0.22)

    q2_values = np.linspace(-np.pi + 0.01, np.pi - 0.01, 500)
    singular_values = np.array([np.linalg.svd(jacobian(0.0, q2), compute_uv=False) for q2 in q2_values])
    sigma_max = singular_values[:, 0]
    sigma_min = singular_values[:, 1]
    manipulability = sigma_max * sigma_min
    ax = axes[1]
    ax.plot(np.rad2deg(q2_values), sigma_max, color=BLUE, linewidth=2.2, label="largest singular value")
    ax.plot(np.rad2deg(q2_values), sigma_min, color=ORANGE, linewidth=2.2, label="smallest singular value")
    ax.plot(np.rad2deg(q2_values), manipulability, color=GREEN, linewidth=2.2, linestyle="--",
            label="manipulability")
    for angle in (-180, 0, 180):
        ax.axvline(angle, color=PURPLE, linestyle=":", linewidth=1.3)
    ax.set(xlabel="Elbow angle q2 (deg)", ylabel="Scale", title="Singularity is a continuous loss of one direction")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper center", fontsize=8)
    ax.set_xlim(-180, 180)

    output = Path(__file__).resolve().parents[1] / "public/robot_manipulability_singularity.png"
    fig.savefig(output, dpi=160, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
