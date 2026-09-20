"""Draw the local-linearization intuition used by the Jacobian article."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


output = Path(__file__).resolve().parents[1] / "public" / "jacobian_local_linearization.png"

x = np.linspace(-0.15, 2.25, 500)
y = 0.55 * x**2 + 0.35
x0 = 1.0
y0 = 0.55 * x0**2 + 0.35
slope = 1.1 * x0
tangent = y0 + slope * (x - x0)

fig, ax = plt.subplots(figsize=(7, 4.8), dpi=180)
ax.plot(x, y, color="#2196F3", linewidth=2.6, label="nonlinear mapping  y = f(x)")
ax.plot(x, tangent, color="#FF9800", linewidth=2.2, linestyle="--", label="local linear approximation")

dx = 0.38
x1 = x0 + dx
y1_curve = 0.55 * x1**2 + 0.35
y1_linear = y0 + slope * dx

ax.scatter([x0], [y0], s=55, color="#F44336", zorder=5)
ax.annotate("reference point", (x0, y0), xytext=(-55, 20), textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#607D8B"}, color="#37474F")

ax.annotate("", xy=(x1, y0 - 0.12), xytext=(x0, y0 - 0.12),
            arrowprops={"arrowstyle": "<->", "color": "#4CAF50", "linewidth": 2})
ax.text((x0 + x1) / 2, y0 - 0.25, "small input change  dx", ha="center", color="#2E7D32")

ax.annotate("", xy=(x1 + 0.04, y1_linear), xytext=(x1 + 0.04, y0),
            arrowprops={"arrowstyle": "<->", "color": "#9C27B0", "linewidth": 2})
ax.text(x1 + 0.10, (y0 + y1_linear) / 2, "predicted output\nchange  J dx",
        va="center", color="#7B1FA2")

ax.plot([x1, x1], [y1_linear, y1_curve], color="#F44336", linewidth=1.6)
ax.scatter([x1, x1], [y1_linear, y1_curve], s=28, color=["#FF9800", "#2196F3"], zorder=5)
ax.text(x1 + 0.08, y1_curve + 0.06, "curvature error", color="#C62828")

ax.set_title("Jacobian: replace a curve by a tangent near one point")
ax.set_xlabel("input x")
ax.set_ylabel("output y")
ax.set_xlim(-0.1, 2.15)
ax.set_ylim(0.05, 3.05)
ax.grid(alpha=0.22)
ax.legend(loc="upper left", frameon=True)
fig.tight_layout()
fig.savefig(output, bbox_inches="tight")
plt.close(fig)

print(output)
