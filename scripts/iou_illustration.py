#!/usr/bin/env python3
"""Render an IoU illustration: two overlapping boxes, shaded intersection & union."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# ---- Left panel: the two regions + intersection ----
ax = axes[0]
ax.set_title("Two regions A (truth) and B (prediction)", fontsize=13, fontweight="bold")
# A: x in [1,5], y in [1.5,4.5]  -> 4 x 3 = 12
A = Rectangle((1, 1.5), 4, 3, linewidth=2.5, edgecolor="#2196F3", facecolor="#2196F3", alpha=0.25)
# B: x in [3.5,7.5], y in [2,5]  -> 4 x 3 = 12
B = Rectangle((3.5, 2.0), 4, 3, linewidth=2.5, edgecolor="#F44336", facecolor="#F44336", alpha=0.25)
ax.add_patch(A)
ax.add_patch(B)
# intersection: x in [3.5,5], y in [2,4.5] -> 1.5 x 2.5 = 3.75
inter = Rectangle((3.5, 2.0), 1.5, 2.5, linewidth=0, facecolor="#9C27B0", alpha=0.55)
ax.add_patch(inter)
ax.text(2.0, 2.6, "A", color="#1565C0", fontsize=18, fontweight="bold")
ax.text(6.6, 4.2, "B", color="#c62828", fontsize=18, fontweight="bold")
ax.text(4.25, 3.25, "A∩B", color="white", fontsize=11, fontweight="bold", ha="center")
ax.set_xlim(0, 9); ax.set_ylim(0, 6.2); ax.set_aspect("equal"); ax.axis("off")

# ---- Right panel: the ratio ----
ax = axes[1]
ax.set_title("IoU = area(A∩B) / area(A∪B)", fontsize=13, fontweight="bold")
# union outline (draw both boxes' union region as light)
U_A = Rectangle((1, 1.5), 4, 3, linewidth=0, facecolor="#607D8B", alpha=0.18)
U_B = Rectangle((3.5, 2.0), 4, 3, linewidth=0, facecolor="#607D8B", alpha=0.18)
ax.add_patch(U_A); ax.add_patch(U_B)
inter2 = Rectangle((3.5, 2.0), 1.5, 2.5, linewidth=0, facecolor="#9C27B0", alpha=0.65)
ax.add_patch(inter2)
ax.text(4.25, 3.25, "3.75", color="white", fontsize=11, fontweight="bold", ha="center")
ax.text(4.5, 0.7,
        "A∩B = 3.75,  A∪B = 12+12-3.75 = 20.25\nIoU = 3.75 / 20.25 ≈ 0.185",
        color="#37474F", fontsize=11, ha="center")
ax.set_xlim(0, 9); ax.set_ylim(0, 6.2); ax.set_aspect("equal"); ax.axis("off")

plt.tight_layout()
plt.savefig("public/iou_illustration.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("public/iou_illustration.png")
