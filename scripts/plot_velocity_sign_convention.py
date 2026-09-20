#!/usr/bin/env python3
"""Draw the one-dimensional velocity sign convention used in chapter 6."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BLUE = "#2196F3"
BLUE_LIGHT = "#E3F2FD"
ORANGE = "#FF9800"
ORANGE_LIGHT = "#FFF3E0"
GREEN = "#4CAF50"
GREY = "#607D8B"
INK = "#263238"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "velocity_sign_convention.png"


def draw_body(ax, center_x, label, edge_color, face_color):
    width = 0.58
    height = 0.58
    body = FancyBboxPatch(
        (center_x - width / 2, -height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=2.5,
        zorder=3,
    )
    ax.add_patch(body)
    ax.text(
        center_x,
        0,
        label,
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=INK,
        zorder=4,
    )


def main():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-1.10, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_body(ax, -1.25, r"$A$", BLUE, BLUE_LIGHT)
    draw_body(ax, 1.25, r"$B$", ORANGE, ORANGE_LIGHT)

    ax.annotate(
        "",
        xy=(0.08, 0.55),
        xytext=(-1.25, 0.55),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3.2, mutation_scale=18),
    )
    ax.text(
        -0.58,
        0.73,
        r"$v_A=+3\,\mathrm{m/s}$",
        ha="center",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color=BLUE,
    )

    ax.annotate(
        "",
        xy=(0.78, 0.55),
        xytext=(1.25, 0.55),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3.2, mutation_scale=18),
    )
    ax.text(
        1.25,
        0.73,
        r"$v_B=-1\,\mathrm{m/s}$",
        ha="center",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color="#EF6C00",
    )

    ax.plot([-2.02, 1.85], [-0.72, -0.72], color=GREY, lw=1.8)
    ax.annotate(
        "",
        xy=(2.08, -0.72),
        xytext=(1.82, -0.72),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.6, mutation_scale=17),
    )
    ax.text(
        2.05,
        -0.52,
        r"$+x$",
        ha="right",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="#2E7D32",
    )
    ax.text(
        0,
        -0.98,
        "positive direction: right",
        ha="center",
        va="center",
        fontsize=13,
        color=GREY,
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
