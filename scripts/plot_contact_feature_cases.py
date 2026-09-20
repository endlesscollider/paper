#!/usr/bin/env python3
"""Draw representative contact-feature cases for the collision overview."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np


BLUE = "#2196F3"
BLUE_LIGHT = "#E3F2FD"
ORANGE = "#FF9800"
ORANGE_LIGHT = "#FFF3E0"
GREEN = "#4CAF50"
PURPLE = "#9C27B0"
GREY = "#607D8B"
GREY_LIGHT = "#ECEFF1"
INK = "#263238"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "contact_feature_cases.png"


def setup_axis(ax, title):
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.05, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, color=INK, pad=8, fontweight="bold")


def mark_contact(ax, point, label=None):
    ax.scatter(*point, s=42, color=PURPLE, zorder=8)
    if label:
        ax.text(point[0] + 0.08, point[1] + 0.08, label, color=PURPLE,
                fontsize=11, fontweight="bold")


def draw_vertex_face(ax):
    setup_axis(ax, "Vertex-face")
    ax.add_patch(Rectangle((-1.35, -0.05), 2.7, 0.38, facecolor=ORANGE_LIGHT,
                           edgecolor=ORANGE, linewidth=2.2))
    ax.add_patch(Polygon([(-1.1, 0.95), (0.2, 0.95), (0.0, -0.18)],
                         closed=True, facecolor=BLUE_LIGHT, edgecolor=BLUE,
                         linewidth=2.2))
    mark_contact(ax, (0.0, -0.18), "pA")
    mark_contact(ax, (0.0, -0.05), "pB")
    ax.annotate("", xy=(0.0, -0.05), xytext=(0.0, -0.28),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2))
    ax.text(0.12, -0.22, "n", color="#2E7D32", fontsize=12, fontweight="bold")
    ax.text(-1.28, -0.72, "one vertex against a face", fontsize=10, color=GREY)


def draw_edge_edge(ax):
    setup_axis(ax, "Edge-edge")
    ax.plot([-1.25, 1.1], [0.85, -0.75], color=BLUE, linewidth=9,
            solid_capstyle="round", alpha=0.85)
    ax.plot([-1.2, 1.2], [-0.72, 0.8], color=ORANGE, linewidth=9,
            solid_capstyle="round", alpha=0.85)
    mark_contact(ax, (0.0, 0.05), "p")
    ax.annotate("", xy=(0.42, 0.34), xytext=(0.05, 0.05),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2))
    ax.text(0.45, 0.36, "n", color="#2E7D32", fontsize=12, fontweight="bold")
    ax.text(-1.28, -0.94, "closest points of two edges", fontsize=10, color=GREY)


def draw_face_face(ax):
    setup_axis(ax, "Face-face")
    ax.add_patch(Rectangle((-1.28, 0.10), 2.1, 0.65, facecolor=BLUE_LIGHT,
                           edgecolor=BLUE, linewidth=2.2))
    ax.add_patch(Rectangle((-0.55, -0.35), 1.7, 0.65, facecolor=ORANGE_LIGHT,
                           edgecolor=ORANGE, linewidth=2.2))
    ax.add_patch(Rectangle((-0.55, 0.10), 1.05, 0.20, facecolor="#F3E5F5",
                           edgecolor=PURPLE, linewidth=1.2, linestyle="--"))
    for point in [(-0.42, 0.18), (0.0, 0.18), (0.38, 0.18)]:
        mark_contact(ax, point)
    ax.annotate("", xy=(1.02, 0.53), xytext=(1.02, 0.18),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2))
    ax.text(1.08, 0.37, "n", color="#2E7D32", fontsize=12, fontweight="bold")
    ax.text(-1.28, -0.72, "a patch becomes several contacts", fontsize=10, color=GREY)


def draw_vertex_vertex(ax):
    setup_axis(ax, "Vertex-vertex")
    ax.add_patch(Polygon([(-1.25, -0.65), (-0.2, 0.0), (-1.25, 0.7)],
                         closed=True, facecolor=BLUE_LIGHT, edgecolor=BLUE,
                         linewidth=2.2))
    ax.add_patch(Polygon([(1.25, -0.65), (0.2, 0.0), (1.25, 0.7)],
                         closed=True, facecolor=ORANGE_LIGHT, edgecolor=ORANGE,
                         linewidth=2.2))
    mark_contact(ax, (0.0, 0.0), "p")
    ax.annotate("", xy=(0.0, 0.0), xytext=(-0.45, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2))
    ax.text(-0.33, 0.12, "n", color="#2E7D32", fontsize=12, fontweight="bold")
    ax.text(-1.28, -0.94, "one point; normal may need a fallback", fontsize=10, color=GREY)


def draw_corner_multiple_faces(ax):
    setup_axis(ax, "Corner + multiple faces")
    ax.add_patch(Rectangle((-1.35, -0.85), 2.7, 0.25, facecolor=ORANGE_LIGHT,
                           edgecolor=ORANGE, linewidth=2.0))
    ax.add_patch(Rectangle((-1.35, -0.85), 0.25, 1.65, facecolor=ORANGE_LIGHT,
                           edgecolor=ORANGE, linewidth=2.0))
    ax.add_patch(Polygon([(-0.05, 0.9), (0.52, 0.32), (0.05, -0.25),
                          (-0.52, 0.32)], closed=True, facecolor=BLUE_LIGHT,
                         edgecolor=BLUE, linewidth=2.2))
    mark_contact(ax, (-0.05, -0.25), "c1")
    mark_contact(ax, (-0.52, 0.32), "c2")
    ax.annotate("", xy=(-0.05, -0.72), xytext=(-0.05, -0.25),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
    ax.annotate("", xy=(-1.03, 0.32), xytext=(-0.52, 0.32),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
    ax.text(0.02, -0.63, "n1", color="#2E7D32", fontsize=11, fontweight="bold")
    ax.text(-0.98, 0.42, "n2", color="#2E7D32", fontsize=11, fontweight="bold")
    ax.text(-1.28, -1.0, "keep each active normal as a constraint", fontsize=10, color=GREY)


def draw_plane_halfspace(ax):
    setup_axis(ax, "Plane / solid half-space")
    ax.add_patch(Rectangle((-1.45, -0.9), 2.9, 0.9, facecolor=GREY_LIGHT,
                           edgecolor=GREY, linewidth=2.0))
    ax.text(-1.35, -0.72, "static half-space", fontsize=10, color=GREY)
    angle = np.deg2rad(-13)
    center = np.array([0.0, 0.4])
    local = np.array([[-0.95, -0.42], [0.95, -0.42], [0.95, 0.42],
                      [-0.95, 0.42]])
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    vertices = local @ rotation.T + center
    ax.add_patch(Polygon(vertices, closed=True, facecolor=BLUE_LIGHT,
                         edgecolor=BLUE, linewidth=2.2))
    for point in [(-0.6, 0.0), (0.0, -0.04), (0.56, 0.03)]:
        mark_contact(ax, point)
        ax.annotate("", xy=(point[0], point[1] + 0.33), xytext=point,
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
    ax.text(0.7, 0.48, "one plane, many support contacts", fontsize=10, color=GREY,
            ha="center")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.7))
    fig.patch.set_facecolor("white")
    draw_vertex_face(axes[0, 0])
    draw_edge_edge(axes[0, 1])
    draw_face_face(axes[0, 2])
    draw_vertex_vertex(axes[1, 0])
    draw_corner_multiple_faces(axes[1, 1])
    draw_plane_halfspace(axes[1, 2])
    fig.suptitle("From geometric features to contact constraints", fontsize=19,
                 color=INK, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95], w_pad=1.0, h_pad=1.4)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
