#!/usr/bin/env python3
"""
生成共轭梯度法(CG)求解稀疏线性系统的收敛可视化：
- 子图1：2D二次型 f(x) = 0.5 x^T A x - b^T x 的等高线图，
  对比最速下降(gradient descent)和共轭梯度法(CG)的迭代路径——
  最速下降走"之字形"，CG最多两步（2维情况下）精确收敛。
- 子图2：残差范数随迭代次数下降的曲线（对数坐标），
  展示CG在更高维度问题上的典型收敛速度。
"""
import numpy as np
import matplotlib.pyplot as plt

# 一个病态程度适中的2x2正定矩阵，方便可视化"之字形"效应
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x_star = np.linalg.solve(A, b)

def f(x):
    return 0.5 * x @ A @ x - b @ x

def gradient_descent(x0, n_iters, lr=0.15):
    path = [x0.copy()]
    x = x0.copy()
    for _ in range(n_iters):
        grad = A @ x - b
        x = x - lr * grad
        path.append(x.copy())
    return np.array(path)

def conjugate_gradient(x0, n_iters):
    path = [x0.copy()]
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    for _ in range(n_iters):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        path.append(x.copy())
        if np.linalg.norm(r) < 1e-10:
            break
    return np.array(path)

x0 = np.array([-1.5, -1.0])
gd_path = gradient_descent(x0, 25)
cg_path = conjugate_gradient(x0, 5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---------- 子图1：等高线 + 两种路径 ----------
xs = np.linspace(-2, 2.5, 200)
ys = np.linspace(-2, 2.5, 200)
X, Y = np.meshgrid(xs, ys)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

ax1.contour(X, Y, Z, levels=25, colors='#BDBDBD', linewidths=0.6)
ax1.plot(gd_path[:, 0], gd_path[:, 1], '-o', color='#F44336', markersize=3, linewidth=1.3, label=f'Gradient Descent ({len(gd_path)-1} steps)')
ax1.plot(cg_path[:, 0], cg_path[:, 1], '-o', color='#2196F3', markersize=6, linewidth=2.0, label=f'Conjugate Gradient ({len(cg_path)-1} steps)')
ax1.plot(x_star[0], x_star[1], '*', color='#4CAF50', markersize=18, label='Exact solution $x^*$', zorder=5)
ax1.plot(x0[0], x0[1], 's', color='#333', markersize=7, label='Start $x_0$', zorder=5)

ax1.set_xlabel('$x_1$', fontsize=11)
ax1.set_ylabel('$x_2$', fontsize=11)
ax1.set_title('CG Converges in 2 Steps,\nGradient Descent Zig-Zags', fontsize=11.5, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(True, alpha=0.2)
ax1.set_aspect('equal')

# ---------- 子图2：更高维度问题的残差收敛曲线 ----------
np.random.seed(0)
n = 100
M = np.random.randn(n, n)
A_big = M @ M.T + n * np.eye(n)  # 保证正定，且有一定病态程度
b_big = np.random.randn(n)

def cg_residuals(A, b, n_iters):
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    residuals = [np.linalg.norm(r)]
    for _ in range(n_iters):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        residuals.append(np.linalg.norm(r_new))
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    return residuals

residuals = cg_residuals(A_big, b_big, 60)
ax2.semilogy(residuals, '-o', color='#9C27B0', markersize=3, linewidth=1.8)
ax2.set_xlabel('CG iteration', fontsize=11)
ax2.set_ylabel('Residual norm $\\|r\\|$ (log scale)', fontsize=11)
ax2.set_title(f'CG Residual Decay on a {n}x{n} SPD System', fontsize=11.5, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('public/conjugate_gradient_convergence.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/conjugate_gradient_convergence.png")
