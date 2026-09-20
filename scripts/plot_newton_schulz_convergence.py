#!/usr/bin/env python3
"""
展示 Newton-Schulz 迭代如何把奇异值从初始范围 [0,1] 逐步推向 1（即把梯度矩阵
"拉"向正交矩阵的过程）。

多项式 phi(x) = a*x + b*x^3 + c*x^5，(a,b,c) = (3.4445, -4.7750, 2.0315) 是
Muon 使用的系数。图中展示 phi 迭代 1~5 次后，不同初始奇异值 x 被映射到的位置——
迭代次数越多，越多的奇异值被压缩到 1 附近（正交矩阵的所有奇异值都是 1）。
"""
import numpy as np
import matplotlib.pyplot as plt

a, b, c = 3.4445, -4.7750, 2.0315

def phi(x):
    return a * x + b * x**3 + c * x**5

x = np.linspace(0, 1, 500)

fig, ax = plt.subplots(figsize=(6, 4.5))

y = x.copy()
colors = ['#B0BEC5', '#FF9800', '#FFC107', '#4CAF50', '#2196F3', '#9C27B0']
ax.plot(x, x, '--', color='#607D8B', linewidth=1.2, label='identity (0 steps)')

n_steps = 5
for step in range(1, n_steps + 1):
    y = phi(y)
    ax.plot(x, y, '-', color=colors[step], linewidth=2.0, label=f'after {step} NS step(s)')

ax.axhline(y=1.0, color='#F44336', linestyle=':', linewidth=1.5, alpha=0.8)
ax.text(0.02, 1.03, 'target: all singular values -> 1 (orthogonal)', fontsize=8.5, color='#F44336')

ax.set_xlabel('Initial singular value $x \\in [0,1]$', fontsize=10)
ax.set_ylabel('$\\varphi^{(N)}(x)$ after N Newton-Schulz steps', fontsize=10)
ax.set_title('Newton-Schulz Iteration Pushes Singular Values Toward 1', fontsize=11, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.4)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('public/newton_schulz_convergence.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/newton_schulz_convergence.png")
