"""Generate the worked-example figure used by the Unscented Transform article."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normal_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    scale = np.sqrt(variance)
    return np.exp(-0.5 * ((x - mean) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))


rng = np.random.default_rng(7)
x_samples = rng.normal(2.0, 1.0, 500_000)
y_samples = x_samples**2

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

x = np.linspace(-1.2, 5.2, 500)
axes[0].plot(x, x**2, color="#2196F3", linewidth=2.3, label=r"$g(x)=x^2$")
axes[0].scatter([2, 3, 1], [4, 9, 1], s=65, color=["#F44336", "#4CAF50", "#4CAF50"], zorder=3)
axes[0].annotate("mean point: (2, 4)", (2, 4), xytext=(0.35, 6.0), arrowprops={"arrowstyle": "->"})
axes[0].annotate("sigma point: (3, 9)", (3, 9), xytext=(3.3, 7.5), arrowprops={"arrowstyle": "->"})
axes[0].annotate("sigma point: (1, 1)", (1, 1), xytext=(-0.8, 2.2), arrowprops={"arrowstyle": "->"})
axes[0].set(xlabel="input x", ylabel="output y", title="Three points through a nonlinear transform", xlim=(-1.2, 5.2), ylim=(-0.5, 11))
axes[0].grid(alpha=0.25)
axes[0].legend(loc="upper left")

bins = np.linspace(0, 25, 180)
axes[1].hist(y_samples, bins=bins, density=True, alpha=0.34, color="#607D8B", label="Monte Carlo reference")
y = np.linspace(0, 25, 600)
axes[1].plot(y, normal_pdf(y, 4.0, 16.0), color="#FF9800", linewidth=2.2, label="EWA: mean=4, var=16")
axes[1].plot(y, normal_pdf(y, 5.0, 18.0), color="#9C27B0", linewidth=2.2, label="UT: mean=5, var=18")
axes[1].axvline(5.0, color="#4CAF50", linestyle="--", linewidth=1.3, label="exact mean=5")
axes[1].set(xlabel="transformed value y", ylabel="density", title="Moment approximation after transformation", xlim=(0, 25), ylim=(0, 0.32))
axes[1].grid(alpha=0.25)
axes[1].legend(loc="upper right", fontsize=8)

output = Path("public/unscented_transform_nonlinear_example.png")
fig.savefig(output, dpi=170)
plt.close(fig)
print(output)
