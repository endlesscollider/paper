#!/usr/bin/env python3
"""Draw approaching, tangential-sliding, and separating contact states."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


BLUE = "#2196F3"
BLUE_LIGHT = "#E3F2FD"
ORANGE = "#FF9800"
ORANGE_LIGHT = "#FFF3E0"
GREEN = "#4CAF50"
PURPLE = "#9C27B0"
RED = "#F44336"
GREY = "#607D8B"
GREY_LIGHT = "#ECEFF1"
INK = "#263238"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "contact_normal_velocity_cases.png"


def draw_badge(ax, text, color, face_color):
    badge = FancyBboxPatch(
        (-1.42, -0.68),
        2.84,
        0.25,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        facecolor=face_color,
        edgecolor=color,
        linewidth=1.4,
        zorder=4,
    )
    ax.add_patch(badge)
    ax.text(0, -0.555, text, ha="center", va="center", fontsize=13,
            color=color, fontweight="bold", zorder=5)


def draw_common_contact(ax, title, normal_value, relative_direction):
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-0.78, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(-2.02, 0.91, title, fontsize=18, color=INK,
            ha="left", va="center", fontweight="bold")

    ax.add_patch(Rectangle((-1.65, -0.32), 1.60, 0.72,
                           facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=2.2))
    ax.add_patch(Rectangle((0.05, -0.32), 1.60, 0.72,
                           facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=2.2))
    ax.text(-1.35, 0.03, r"$A$", fontsize=21, color=BLUE,
            ha="center", va="center", fontweight="bold")
    ax.text(1.35, 0.03, r"$B$", fontsize=21, color="#EF6C00",
            ha="center", va="center", fontweight="bold")

    ax.scatter([-0.05], [0.03], s=48, color=BLUE, zorder=6)
    ax.scatter([0.05], [0.03], s=48, color=ORANGE, zorder=6)
    ax.plot([-0.05, 0.05], [0.03, 0.03], color=GREY, linewidth=1.2,
            linestyle="--", zorder=5)
    ax.text(-0.12, -0.18, r"$p_A$", fontsize=16.5, color=BLUE,
            ha="right", fontweight="bold")
    ax.text(0.12, -0.18, r"$p_B$", fontsize=16.5, color="#EF6C00",
            ha="left", fontweight="bold")

    ax.annotate(
        "",
        xy=(1.48, 0.60),
        xytext=(0.10, 0.60),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.7, mutation_scale=16),
    )
    ax.text(0.82, 0.69, r"$\mathbf{n}$  (A to B)", fontsize=16,
            color="#2E7D32", ha="center", fontweight="bold")

    if relative_direction == "left":
        ax.annotate(
            "",
            xy=(-1.03, 0.60),
            xytext=(-0.08, 0.60),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.9, mutation_scale=17),
        )
        ax.text(-0.57, 0.69, r"$\mathbf{v}_{rel}$", fontsize=16,
                color=PURPLE, ha="center", fontweight="bold")
    elif relative_direction == "up":
        ax.annotate(
            "",
            xy=(0.0, 0.92),
            xytext=(0.0, 0.16),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.9, mutation_scale=17),
        )
        ax.text(-0.15, 0.72, r"$\mathbf{v}_{rel}=\mathbf{v}_t$", fontsize=15.5,
                color=PURPLE, ha="right", fontweight="bold")
    else:
        ax.annotate(
            "",
            xy=(1.02, 0.60),
            xytext=(0.08, 0.60),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.9, mutation_scale=17),
        )
        ax.text(0.55, 0.45, r"$\mathbf{v}_{rel}$", fontsize=16,
                color=PURPLE, ha="center", fontweight="bold")

    ax.text(2.10, -0.03, normal_value, fontsize=17, color=INK,
            ha="right", va="center", fontweight="bold")


def main():
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.6))
    fig.patch.set_facecolor("white")

    draw_common_contact(axes[0], "1  Approaching", r"$v_n<0$", "left")
    draw_badge(axes[0], "apply normal impact impulse", RED, "#FFEBEE")

    draw_common_contact(axes[1], "2  Tangential sliding", r"$v_n=0$", "up")
    draw_badge(axes[1], "no impact impulse; friction may act", GREY, GREY_LIGHT)

    draw_common_contact(axes[2], "3  Separating", r"$v_n>0$", "right")
    draw_badge(axes[2], "no normal impact impulse", GREY, GREY_LIGHT)

    fig.suptitle(
        r"$\mathbf{v}_{rel}=\mathbf{v}_B^p-\mathbf{v}_A^p$",
        fontsize=21,
        color=INK,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        r"$p_A$ and $p_B$ coincide at ideal contact (drawn apart for clarity)",
        ha="center",
        va="bottom",
        fontsize=12,
        color=GREY,
    )
    fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.975), h_pad=0.55)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
