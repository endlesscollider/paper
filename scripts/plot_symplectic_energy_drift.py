#!/usr/bin/env python3
"""
生成辛积分 vs 显式 Euler 的能量稳定性对比图：
- 子图1：简谐振子（弹簧-质点）相空间轨迹对比——显式 Euler 螺旋外扩发散，
  半隐式 Euler（辛积分）轨迹保持在闭合椭圆附近
- 子图2：总机械能随时间变化曲线——显式 Euler 能量单调增长（数值发散），
  半隐式 Euler 能量在小范围内振荡但不发散
"""
import numpy as np
import matplotlib.pyplot as plt

# ============ 简谐振子参数：单位质量、单位弹簧系数 ============
k = 1.0   # 弹簧系数
m = 1.0   # 质量
omega = np.sqrt(k / m)

x0, v0 = 1.0, 0.0  # 初始位移、初始速度
dt = 0.3            # 故意用较大步长，放大数值误差方便观察
n_steps = 60

# ============ 显式 Euler（explicit Euler）============
def explicit_euler(x0, v0, dt, n_steps):
    x, v = x0, v0
    xs, vs = [x], [v]
    for _ in range(n_steps):
        a = -k / m * x            # 加速度：只由当前位置决定
        x_new = x + dt * v        # 用旧速度更新位置
        v_new = v + dt * a        # 用旧位置对应的加速度更新速度
        x, v = x_new, v_new
        xs.append(x)
        vs.append(v)
    return np.array(xs), np.array(vs)

# ============ 半隐式 Euler / Symplectic Euler ============
def symplectic_euler(x0, v0, dt, n_steps):
    x, v = x0, v0
    xs, vs = [x], [v]
    for _ in range(n_steps):
        a = -k / m * x
        v_new = v + dt * a        # 先用当前位置更新速度
        x_new = x + dt * v_new    # 再用"新"速度更新位置（这是关键区别）
        x, v = x_new, v_new
        xs.append(x)
        vs.append(v)
    return np.array(xs), np.array(vs)

xs_exp, vs_exp = explicit_euler(x0, v0, dt, n_steps)
xs_sym, vs_sym = symplectic_euler(x0, v0, dt, n_steps)

# 精确解（相空间是一个完美的圆，因为 m=k=1）
t_exact = np.linspace(0, 2 * np.pi / omega, 300)
x_exact = x0 * np.cos(omega * t_exact)
v_exact = -x0 * omega * np.sin(omega * t_exact)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ============ 子图1：相空间轨迹 ============
ax1.plot(x_exact, v_exact, '-', color='#9E9E9E', linewidth=1.5, alpha=0.6, label='Exact (closed ellipse)')
ax1.plot(xs_exp, vs_exp, '-o', color='#F44336', markersize=3, linewidth=1.5, label='Explicit Euler (spirals out)')
ax1.plot(xs_sym, vs_sym, '-o', color='#2196F3', markersize=3, linewidth=1.5, label='Symplectic Euler (stays bounded)')
ax1.plot(x0, v0, 's', color='#4CAF50', markersize=8, zorder=5, label='Start')

ax1.set_xlabel('Position $x$', fontsize=11)
ax1.set_ylabel('Velocity $v$', fontsize=11)
ax1.set_title('Phase Space: Explicit vs Symplectic Euler', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# ============ 子图2：总机械能随时间变化 ============
def total_energy(xs, vs, k=1.0, m=1.0):
    return 0.5 * k * xs**2 + 0.5 * m * vs**2

E_exp = total_energy(xs_exp, vs_exp)
E_sym = total_energy(xs_sym, vs_sym)
E0 = total_energy(np.array([x0]), np.array([v0]))[0]

steps = np.arange(n_steps + 1)
ax2.plot(steps, E_exp, '-o', color='#F44336', markersize=3, linewidth=1.8, label='Explicit Euler')
ax2.plot(steps, E_sym, '-o', color='#2196F3', markersize=3, linewidth=1.8, label='Symplectic Euler')
ax2.axhline(y=E0, color='#4CAF50', linestyle='--', linewidth=1.5, label='Exact energy (constant)')

ax2.set_xlabel('Time step', fontsize=11)
ax2.set_ylabel('Total mechanical energy $E$', fontsize=11)
ax2.set_title('Energy Drift Over Time', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('public/symplectic_vs_explicit_euler_energy.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/symplectic_vs_explicit_euler_energy.png")
