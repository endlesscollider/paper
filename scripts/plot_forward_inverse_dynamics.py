"""Plot payload-dependent forward dynamics and algorithmic scaling."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2196F3"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"


def mass_matrix(q2: float, payload: float) -> np.ndarray:
    l1, lc1, lc2 = 0.8, 0.4, 0.3
    m1, m2 = 1.5, 1.0 + payload
    i1, i2 = 0.08, 0.04 + payload * 0.3**2
    return np.array([
        [i1 + i2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(q2)),
         i2 + m2 * (lc2**2 + l1 * lc2 * np.cos(q2))],
        [i2 + m2 * (lc2**2 + l1 * lc2 * np.cos(q2)), i2 + m2 * lc2**2],
    ])


def main() -> None:
    payloads = np.array([0.0, 0.5, 1.5, 3.0])
    applied_torque_after_bias = np.array([4.0, 1.0])
    accelerations = np.array([
        np.linalg.solve(mass_matrix(np.deg2rad(60), payload), applied_torque_after_bias)
        for payload in payloads
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    x = np.arange(len(payloads))
    width = 0.34
    ax = axes[0]
    ax.bar(x - width / 2, accelerations[:, 0], width, color=BLUE, label="joint 1 acceleration")
    ax.bar(x + width / 2, accelerations[:, 1], width, color=ORANGE, label="joint 2 acceleration")
    ax.set_xticks(x, [f"{p:.1f} kg" for p in payloads])
    ax.set(xlabel="Payload", ylabel="Acceleration (rad/s^2)",
           title="Same torque, different inertia, different acceleration")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="best", fontsize=8)

    dof = np.arange(2, 101)
    dense = dof**3
    recursive = 35 * dof
    ax = axes[1]
    ax.loglog(dof, dense, color=PURPLE, linewidth=2.3, label="dense solve trend O(n^3)")
    ax.loglog(dof, recursive, color=GREEN, linewidth=2.3, label="recursive rigid-body trend O(n)")
    ax.scatter([7, 30], [7**3, 30**3], color=PURPLE, s=40)
    ax.scatter([7, 30], [35 * 7, 35 * 30], color=GREEN, s=40)
    ax.set(xlabel="Degrees of freedom n", ylabel="Relative work (log scale)",
           title="Structure changes dynamics-computation scaling")
    ax.grid(alpha=0.22, which="both")
    ax.legend(loc="upper left", fontsize=8)

    output = Path(__file__).resolve().parents[1] / "public/robot_forward_inverse_dynamics.png"
    fig.savefig(output, dpi=160, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
