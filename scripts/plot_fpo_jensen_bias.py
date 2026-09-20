#!/usr/bin/env python3
"""
一次性脚本：为 FPO 精读生成"Jensen 不等式导致 FPO ratio 估计上偏"的直觉图。

exp(.) 是凸函数，所以 E[exp(X)] >= exp(E[X])。
用几个离散样本点 + 弦线，直观展示"凸函数的期望 >= 期望的函数值"。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

x = np.linspace(-2, 2, 400)
y = np.exp(x)
ax.plot(x, y, color="#2196F3", linewidth=2.2, label=r"$\exp(x)$ (convex function)")

# 几个离散样本点，代表不同 (tau, eps) 采样得到的 log-ratio 差值
samples_x = np.array([-1.2, -0.3, 0.5, 1.4])
samples_y = np.exp(samples_x)
mean_x = samples_x.mean()
mean_y_of_mean_x = np.exp(mean_x)
mean_of_y = samples_y.mean()

ax.scatter(samples_x, samples_y, color="#FF9800", s=55, zorder=5,
           label="Single (τ, ε) sample ratio")

# 连接采样点的弦（示意"平均值在弦上，恒不低于曲线"）
order = np.argsort(samples_x)
ax.plot(samples_x[order], samples_y[order], "--", color="#FF9800", alpha=0.5, linewidth=1.3)

# 标出 E[exp(X)] 和 exp(E[X])
ax.scatter([mean_x], [mean_of_y], color="#F44336", s=80, zorder=6, marker="D",
           label=r"$\mathbb{E}[\exp(X)]$  (FPO's estimator, avg over samples)")
ax.scatter([mean_x], [mean_y_of_mean_x], color="#4CAF50", s=80, zorder=6, marker="s",
           label=r"$\exp(\mathbb{E}[X])$  (true ratio)")

ax.annotate("", xy=(mean_x, mean_of_y), xytext=(mean_x, mean_y_of_mean_x),
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.3))
ax.text(mean_x + 0.08, (mean_of_y + mean_y_of_mean_x) / 2, "upward\nbias",
        fontsize=9, color="#666")

ax.set_title("Why FPO's Ratio Estimator is Upward-Biased", fontsize=12)
ax.set_xlabel(r"$X = \ell_{\theta_{old}} - \ell_\theta$ (single sample)")
ax.set_ylabel(r"$\exp(X)$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig("public/fpo_jensen_inequality_bias.png")
print("saved to public/fpo_jensen_inequality_bias.png")
