#!/usr/bin/env python3
"""Draw compact symbol maps for the ground-contact worked example."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BLUE = "#2196F3"
BLUE_LIGHT = "#E3F2FD"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
RED = "#F44336"
PURPLE = "#9C27B0"
GREY = "#607D8B"
GREY_LIGHT = "#ECEFF1"
INK = "#263238"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_POSITION = ROOT / "public" / "ground_contact_position_correction.png"
OUTPUT_VELOCITY = ROOT / "public" / "ground_contact_normal_velocity.png"


def setup_axis(ax):
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-0.56, 0.86)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_ground(ax):
    ax.add_patch(Rectangle((-1.62, -0.56), 3.24, 0.56,
                           facecolor=GREY_LIGHT, edgecolor="none", zorder=0))
    ax.plot([-1.62, 1.62], [0, 0], color=GREY, lw=3, zorder=2)
    for x in [-1.45, -1.00, -0.55, -0.10, 0.35, 0.80, 1.25]:
        ax.plot([x, x - 0.17], [-0.06, -0.22], color="#B0BEC5", lw=1.2)


def draw_position_figure():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white")
    setup_axis(ax)
    draw_ground(ax)

    bottom = -0.28
    slop_y = -0.04
    ax.add_patch(Rectangle((-0.48, bottom), 0.96, 0.76,
                           facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=2.8, zorder=3))
    ax.text(0, 0.31, r"Body $A$", ha="center", va="center",
            fontsize=20, fontweight="bold", color=INK)
    ax.text(0, 0.16, r"$m_A=2\,\mathrm{kg}$", ha="center", fontsize=15, color=GREY)
    ax.text(0, 0.04, r"$w_A=0.5\,\mathrm{kg}^{-1}$",
            ha="center", fontsize=15, color=GREY)
    ax.text(-1.48, -0.47, r"$B$ (static)   $w_B=0$",
            ha="left", fontsize=16, color=GREY, fontweight="bold")

    ax.plot([-0.55, 1.22], [slop_y, slop_y], color=ORANGE, lw=2.0, ls="--")
    ax.fill_between([-0.55, 1.22], 0, slop_y, color=ORANGE, alpha=0.15)
    ax.annotate(r"$s=0.005\,\mathrm{m}$", xy=(-0.52, slop_y / 2),
                xytext=(-1.38, 0.13), ha="left", fontsize=16, color="#E65100",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))

    ax.annotate("", xy=(0.73, bottom), xytext=(0.73, 0),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2.6))
    ax.text(0.82, -0.10, r"$d=0.08\,\mathrm{m}$", ha="left", va="center",
            fontsize=16, color=RED, fontweight="bold")
    ax.annotate("", xy=(1.20, bottom), xytext=(1.20, slop_y),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=2.4))
    ax.text(1.29, -0.23, r"$d-s$", ha="left", va="center",
            fontsize=17, color=PURPLE, fontweight="bold")

    contact_y = bottom / 2
    ax.scatter([0], [contact_y], s=55, color=INK, zorder=5)
    ax.text(-0.08, contact_y - 0.03, r"$\mathbf{p}$", ha="right", va="top",
            fontsize=18, color=INK, fontweight="bold")

    ax.annotate("", xy=(-0.18, -0.50), xytext=(-0.18, contact_y + 0.03),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3.0))
    ax.text(-0.27, -0.42, r"$\mathbf{n}=(0,-1)$", ha="right",
            fontsize=16, color="#2E7D32", fontweight="bold")

    ax.annotate("", xy=(0.27, -0.50), xytext=(0.27, contact_y + 0.03),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=3.0))
    ax.text(0.36, -0.52, r"$\mathbf{correction}$", ha="left",
            fontsize=17, color=PURPLE, fontweight="bold")

    ax.text(1.08, 0.69, r"$\beta=0.8$", ha="center", fontsize=17, color=GREY,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#CFD8DC"))

    fig.tight_layout(pad=0.15)
    fig.savefig(OUTPUT_POSITION, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_velocity_figure():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white")
    setup_axis(ax)
    draw_ground(ax)

    ax.add_patch(Rectangle((-0.48, 0), 0.96, 0.72,
                           facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=2.8, zorder=3))
    ax.text(0, 0.57, r"$A$   $m=2\,\mathrm{kg}$", ha="center",
            fontsize=18, fontweight="bold", color=INK)
    ax.scatter([0], [0], s=55, color=INK, zorder=5)
    ax.text(-1.48, -0.47, r"$B$   $\mathbf{v}_B^c=0$", ha="left",
            fontsize=16, color=GREY, fontweight="bold")

    ax.annotate("", xy=(-0.22, 0.08), xytext=(-0.22, 0.46),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3.0))
    ax.text(-0.31, 0.29, r"$\mathbf{v}_A^c$", ha="right", va="center",
            fontsize=18, color=BLUE, fontweight="bold")

    ax.annotate("", xy=(0.19, 0.49), xytext=(0.19, 0.08),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=3.0))
    ax.text(0.28, 0.30, r"$\mathbf{v}_{rel}$", ha="left", va="center",
            fontsize=18, color=PURPLE, fontweight="bold")

    ax.annotate("", xy=(0.82, -0.48), xytext=(0.82, 0.03),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3.0))
    ax.text(0.92, -0.27, r"$\mathbf{n}$", ha="left", va="center",
            fontsize=19, color="#2E7D32", fontweight="bold")

    ax.annotate("", xy=(-0.78, 0.50), xytext=(-0.78, 0.04),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=3.0))
    ax.text(-0.89, 0.29, r"$j$", ha="right", va="center",
            fontsize=20, color=RED, fontweight="bold")

    ax.text(1.10, 0.69, r"$e=0.5$", ha="center", fontsize=17, color=GREY,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#CFD8DC"))

    fig.tight_layout(pad=0.15)
    fig.savefig(OUTPUT_VELOCITY, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw_position_figure()
    draw_velocity_figure()
    print(OUTPUT_POSITION)
    print(OUTPUT_VELOCITY)
