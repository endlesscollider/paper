#!/usr/bin/env python3
"""Draw the contact point, normal, and moment arms for two rigid bodies."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle


BLUE = "#2196F3"
BLUE_LIGHT = "#E3F2FD"
ORANGE = "#FF9800"
ORANGE_LIGHT = "#FFF3E0"
GREEN = "#4CAF50"
PURPLE = "#9C27B0"
GREY = "#607D8B"
INK = "#263238"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "two_body_contact_geometry.png"


def rotated_square(center, half_extent, angle_degrees):
    local = np.array(
        [
            [-half_extent, -half_extent],
            [half_extent, -half_extent],
            [half_extent, half_extent],
            [-half_extent, half_extent],
        ]
    )
    angle = np.deg2rad(angle_degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return local @ rotation.T + np.asarray(center)


def main():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-1.9, 1.8)
    ax.set_ylim(-1.05, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    center_a = np.array([-0.90, 0.20])
    vertices_a = rotated_square(center_a, 0.55, 20)
    contact = vertices_a[1]
    center_b = np.array([contact[0] + 0.60, 0.35])

    ax.add_patch(
        Polygon(
            vertices_a,
            closed=True,
            facecolor=BLUE_LIGHT,
            edgecolor=BLUE,
            linewidth=2.7,
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (contact[0], center_b[1] - 0.60),
            1.20,
            1.20,
            facecolor=ORANGE_LIGHT,
            edgecolor=ORANGE,
            linewidth=2.7,
            zorder=1,
        )
    )

    ax.scatter(*center_a, s=58, color=INK, zorder=5)
    ax.scatter(*center_b, s=58, color=INK, zorder=5)
    ax.scatter(*contact, s=72, color=PURPLE, zorder=6)

    ax.text(center_a[0] - 0.11, center_a[1] + 0.17, r"$\mathbf{c}_A$", fontsize=17,
            color=INK, ha="right", fontweight="bold")
    ax.text(center_b[0] + 0.10, center_b[1] + 0.14, r"$\mathbf{c}_B$", fontsize=17,
            color=INK, ha="left", fontweight="bold")
    ax.text(contact[0] - 0.08, contact[1] - 0.18, r"$\mathbf{p}$", fontsize=18,
            color=PURPLE, ha="right", fontweight="bold")

    ax.annotate(
        "",
        xy=contact,
        xytext=center_a,
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5, mutation_scale=15),
        zorder=4,
    )
    midpoint_a = (center_a + contact) / 2
    ax.text(midpoint_a[0] - 0.08, midpoint_a[1] + 0.08, r"$\mathbf{r}_A$",
            fontsize=17, color=BLUE, ha="right", fontweight="bold")

    ax.annotate(
        "",
        xy=contact,
        xytext=center_b,
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5, mutation_scale=15),
        zorder=4,
    )
    midpoint_b = (center_b + contact) / 2
    ax.text(midpoint_b[0], midpoint_b[1] + 0.14, r"$\mathbf{r}_B$",
            fontsize=17, color="#EF6C00", ha="center", fontweight="bold")

    normal_start = contact + np.array([0.0, -0.36])
    normal_end = normal_start + np.array([1.18, 0.0])
    ax.annotate(
        "",
        xy=normal_end,
        xytext=normal_start,
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3.0, mutation_scale=17),
        zorder=5,
    )
    ax.text(
        normal_end[0] - 0.04,
        normal_end[1] + 0.10,
        r"$\mathbf{n}$ (A to B)",
        fontsize=16,
        color="#2E7D32",
        ha="right",
        fontweight="bold",
    )

    ax.text(center_a[0] - 0.42, 0.97, r"Body $A$", fontsize=18,
            color=BLUE, ha="center", fontweight="bold")
    ax.text(center_b[0] + 0.48, 1.02, r"Body $B$", fontsize=18,
            color="#EF6C00", ha="center", fontweight="bold")

    fig.tight_layout(pad=0.2)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
