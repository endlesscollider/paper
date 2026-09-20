#!/usr/bin/env python3
"""
生成优化器下降轨迹对比图：
- 子图1：在一个椭圆形（各方向曲率差异很大）的二维损失面上，画出
  SGD、SGD+Momentum、Adam 三种优化器各自的下降轨迹（等高线 + 轨迹线）
- 子图2：三者的 loss 随迭代步数下降曲线（log 坐标），直观展示收敛速度差异

损失函数：f(x, y) = 0.5 * (a * x^2 + b * y^2)，a << b，
模拟"一个方向陡峭、一个方向平坦"的病态曲率（ill-conditioned）损失面——
这正是深度学习损失面的典型特征，也是 SGD 震荡、Momentum/Adam 改善的核心场景。
"""
import numpy as np
import matplotlib.pyplot as plt

# ============ 损失面定义：病态曲率的二维椭圆碗 ============
a, b = 1.0, 20.0  # x 方向曲率 1，y 方向曲率 20（20倍病态比）

def f(x, y):
    return 0.5 * (a * x**2 + b * y**2)

def grad(x, y):
    return np.array([a * x, b * y])

start = np.array([-4.0, 1.0])  # 起点，远离最优点 (0,0)
n_steps = 60

# ============ SGD（普通梯度下降） ============
def run_sgd(lr=0.09, steps=n_steps):
    x = start.copy()
    traj = [x.copy()]
    for _ in range(steps):
        g = grad(*x)
        x = x - lr * g
        traj.append(x.copy())
    return np.array(traj)

# ============ SGD + Momentum ============
def run_momentum(lr=0.09, beta=0.9, steps=n_steps):
    x = start.copy()
    v = np.zeros(2)
    traj = [x.copy()]
    for _ in range(steps):
        g = grad(*x)
        v = beta * v + (1 - beta) * g
        x = x - lr * v
        traj.append(x.copy())
    return np.array(traj)

# ============ Adam ============
def run_adam(lr=0.35, beta1=0.9, beta2=0.999, eps=1e-8, steps=n_steps):
    x = start.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    traj = [x.copy()]
    for t in range(1, steps + 1):
        g = grad(*x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(x.copy())
    return np.array(traj)

traj_sgd = run_sgd()
traj_mom = run_momentum()
traj_adam = run_adam()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ============ 子图1：等高线 + 下降轨迹 ============
xs = np.linspace(-4.5, 1.5, 300)
ys = np.linspace(-1.5, 1.5, 300)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

ax1.contour(X, Y, Z, levels=20, colors='#B0BEC5', linewidths=0.8, alpha=0.7)

ax1.plot(traj_sgd[:, 0], traj_sgd[:, 1], '-o', color='#F44336', markersize=2.5,
         linewidth=1.5, label='SGD (lr=0.09)', alpha=0.85)
ax1.plot(traj_mom[:, 0], traj_mom[:, 1], '-o', color='#FF9800', markersize=2.5,
         linewidth=1.5, label='SGD+Momentum (lr=0.09, $\\beta$=0.9)', alpha=0.85)
ax1.plot(traj_adam[:, 0], traj_adam[:, 1], '-o', color='#2196F3', markersize=2.5,
         linewidth=1.5, label='Adam (lr=0.35)', alpha=0.85)

ax1.plot(0, 0, '*', color='#4CAF50', markersize=16, zorder=5, label='Minimum')
ax1.plot(start[0], start[1], 's', color='#607D8B', markersize=8, zorder=5, label='Start')

ax1.set_xlabel('x (flat direction, curvature=1)', fontsize=10)
ax1.set_ylabel('y (steep direction, curvature=20)', fontsize=10)
ax1.set_title('Descent Trajectories on an Ill-Conditioned Loss Surface', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_xlim(-4.5, 1.5)
ax1.set_ylim(-1.5, 1.5)

# ============ 子图2：loss 随迭代步数下降（log 坐标） ============
loss_sgd = [f(*p) for p in traj_sgd]
loss_mom = [f(*p) for p in traj_mom]
loss_adam = [f(*p) for p in traj_adam]

ax2.semilogy(loss_sgd, '-', color='#F44336', linewidth=2.2, label='SGD')
ax2.semilogy(loss_mom, '-', color='#FF9800', linewidth=2.2, label='SGD+Momentum')
ax2.semilogy(loss_adam, '-', color='#2196F3', linewidth=2.2, label='Adam')

ax2.set_xlabel('Iteration step', fontsize=10)
ax2.set_ylabel('Loss $f(x,y)$ (log scale)', fontsize=10)
ax2.set_title('Convergence Speed Comparison', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('public/optimizer_trajectories_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/optimizer_trajectories_comparison.png")
