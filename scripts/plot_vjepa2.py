"""Generate figures for the V-JEPA 2 paper reading article."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "public"

BLUE, GREEN, PURPLE, ORANGE = "#2196F3", "#4CAF50", "#9C27B0", "#FF9800"
RED, GREY = "#F44336", "#607D8B"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_scaling():
    steps = [
        ("Baseline\nViT-L/16\nVM2M", 84.2, 0.0, GREY),
        ("+ Data\n2M -> 22M", 85.2, 1.0, BLUE),
        ("+ Model\n300M -> 1B", 86.7, 1.5, GREEN),
        ("+ Training\n90K -> 252K it.", 87.5, 0.8, PURPLE),
        ("+ Resolution\n256 -> 384 / 16 -> 64f", 88.2, 0.7, ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    prev = 0.0
    for i, (_, cum, delta, color) in enumerate(steps):
        if i == 0:
            ax.bar(i, cum, color=color, width=0.6)
            ax.text(i, cum + 0.12, f"{cum:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
            prev = cum
        else:
            ax.bar(i, delta, bottom=prev, color=color, width=0.6)
            ax.text(i, prev + delta + 0.12, f"+{delta:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=color)
            prev += delta
    ax.axhline(84.2, color=GREY, ls="--", lw=1.0)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=8)
    ax.set_ylabel("Average accuracy over 6 tasks (%)")
    ax.set_title("Scaling ingredients of V-JEPA 2 pretraining")
    ax.set_ylim(80, 89.5)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "vjepa2_scaling_ingredients.png")


def fig_ek100():
    base = [("InAViT\n160M", 25.8), ("Video-LLaMA\n7B", 26.0), ("PlausiVL\n8B", 27.6)]
    ours = [("ViT-L\n300M", 32.7), ("ViT-H\n600M", 36.5),
            ("ViT-g\n1B", 38.0), ("ViT-g/384\n1B", 39.7)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xb = np.arange(len(base))
    xo = np.arange(len(ours)) + len(base) + 0.8
    ax.bar(xb, [v for _, v in base], color=GREY, width=0.6,
           label="Task-specific / LLM baselines")
    ax.bar(xo, [v for _, v in ours], color=BLUE, width=0.6,
           label="V-JEPA 2 (frozen + attentive probe)")
    for x, (_, v) in zip(xb, base):
        ax.text(x, v + 0.4, f"{v}", ha="center", fontsize=9)
    for x, (_, v) in zip(xo, ours):
        ax.text(x, v + 0.4, f"{v}", ha="center", fontsize=9, fontweight="bold", color=BLUE)
    ax.set_xticks(list(xb) + list(xo))
    ax.set_xticklabels([l for l, _ in base] + [l for l, _ in ours], fontsize=8)
    ax.set_ylabel("EK100 action recall@5 (%)")
    ax.set_title("Human action anticipation (Epic-Kitchens-100)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(20, 44)
    save(fig, "vjepa2_ek100_scaling.png")


def fig_planning():
    skills = ["Reach", "Grasp\nCup", "Grasp\nBox", "Pick&Place\nCup", "Pick&Place\nBox"]
    cosmos = [80, 0, 20, 0, 0]
    vjepa = [100, 60, 20, 80, 50]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    x = np.arange(len(skills))
    w = 0.38
    ax.bar(x - w / 2, cosmos, w, color=ORANGE, label="Cosmos (video generation)")
    ax.bar(x + w / 2, vjepa, w, color=BLUE, label="V-JEPA 2-AC (latent prediction)")
    ax.set_xticks(x)
    ax.set_xticklabels(skills, fontsize=8)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Zero-shot manipulation, Lab 2")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(["Cosmos\n80 samples", "V-JEPA 2-AC\n800 samples"], [240, 16],
           color=[ORANGE, BLUE], width=0.55)
    for i, v in enumerate([240, 16]):
        ax.text(i, v * 1.18, f"{v} s", ha="center", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Planning time per action (s, log scale)")
    ax.set_title("Cost of one planning step (RTX 4090)")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "vjepa2_planning_cost.png")


def fig_energy():
    sx, sy = 0.30, 0.45
    x = np.linspace(-0.6, 0.6, 400)
    y = np.linspace(-0.6, 0.6, 400)
    X, Y = np.meshgrid(x, y)
    E = 1.0 - np.exp(-((X ** 2) / (2 * sx ** 2) + ((Y + 0.05) ** 2) / (2 * sy ** 2)))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    cs = ax.contourf(X, Y, E, levels=20, cmap="viridis")
    ax.contour(X, Y, E, levels=12, colors="white", linewidths=0.5, alpha=0.5)
    fig.colorbar(cs, ax=ax, label="Energy  =  L1 distance to goal representation")
    ax.plot(0, -0.05, marker="*", ms=18, color=RED,
            label="Energy minimum  (0, -0.05)")
    ax.plot(0, -0.10, marker="X", ms=12, color="white", markeredgecolor="black",
            label="Ground-truth action  (0, -0.10)")
    ax.set_xlabel(r"$\Delta x$ (m)")
    ax.set_ylabel(r"$\Delta y$ (m)")
    ax.set_title("Schematic energy landscape (delta-y reaching)")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "vjepa2_energy_landscape.png")


if __name__ == "__main__":
    fig_scaling()
    fig_ek100()
    fig_planning()
    fig_energy()
