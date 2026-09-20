#!/usr/bin/env python3
"""
生成 Neo-Hookean 超弹性材料能量模型的可视化：
- 子图1：一维拉伸下,Neo-Hookean 能量 vs 简单线性(St. Venant-Kirchhoff)能量对比，
  展示 Neo-Hookean 在压缩到体积趋于0时能量趋于无穷大（自然阻止穿透/反转），
  而线性模型在这个区域反而能量下降，这是不合理的。
- 子图2：体积项 (J-1)^2 随 J 的曲线，展示"不可压缩惩罚"随体积偏离1增大的效果。
"""
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# ---------- 子图1：1D 拉伸下能量对比 ----------
lam = np.linspace(0.05, 3.0, 500)  # 1D 拉伸比

mu = 1.0
# Neo-Hookean 能量密度（1D 简化模拟）：mu/2 * (lam^2 - 1 - 2*ln(lam))
energy_nh = 0.5 * mu * (lam**2 - 1 - 2 * np.log(lam))
# St. Venant-Kirchhoff（线性材料对应的应变能量，用格林应变 E=(lam^2-1)/2）
E_green = 0.5 * (lam**2 - 1)
energy_svk = mu * E_green**2

ax1.plot(lam, energy_svk, color='#9E9E9E', linewidth=2.0, linestyle='--', label='St. Venant-Kirchhoff (linear material)')
ax1.plot(lam, energy_nh, color='#F44336', linewidth=2.4, label='Neo-Hookean')
ax1.axvline(x=1.0, color='#4CAF50', linestyle=':', linewidth=1.3, label='$\\lambda=1$ (rest)')
ax1.axvspan(0, 0.25, color='#FFCDD2', alpha=0.4)
ax1.text(0.08, 6.3, 'compressed\ntoward\nzero volume', fontsize=8, color='#B71C1C')

ax1.set_xlabel('Stretch ratio $\\lambda$', fontsize=11)
ax1.set_ylabel('Strain energy density $\\Psi$', fontsize=11)
ax1.set_title('Neo-Hookean Blows Up Near $\\lambda\\to0$,\nLinear Model Does Not', fontsize=11.5, fontweight='bold')
ax1.set_xlim(0, 3.0)
ax1.set_ylim(0, 7)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8.5, loc='upper right')

# ---------- 子图2：体积惩罚项 (J-1)^2 ----------
J = np.linspace(0.05, 2.5, 500)
kappa = 1.0
vol_energy = 0.5 * kappa * (J - 1) ** 2

ax2.plot(J, vol_energy, color='#2196F3', linewidth=2.4)
ax2.axvline(x=1.0, color='#4CAF50', linestyle=':', linewidth=1.3, label='$J=1$ (volume preserved)')
ax2.fill_between(J, vol_energy, alpha=0.08, color='#2196F3')
ax2.annotate('compressed\n($J<1$)', xy=(0.5, 0.5*kappa*(0.5-1)**2), xytext=(0.25, 1.2),
             fontsize=9, color='#F44336', arrowprops=dict(arrowstyle='->', color='#F44336'))
ax2.annotate('expanded\n($J>1$)', xy=(1.8, 0.5*kappa*(1.8-1)**2), xytext=(1.9, 0.9),
             fontsize=9, color='#FF9800', arrowprops=dict(arrowstyle='->', color='#FF9800'))

ax2.set_xlabel('Volume ratio $J=\\det F$', fontsize=11)
ax2.set_ylabel('Volumetric penalty energy', fontsize=11)
ax2.set_title('Volume-Preservation Penalty $\\frac{\\kappa}{2}(J-1)^2$', fontsize=11.5, fontweight='bold')
ax2.set_xlim(0, 2.5)
ax2.set_ylim(0, 2.0)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8.5, loc='upper center')

plt.tight_layout()
plt.savefig('public/neohookean_energy_vs_linear_and_volume_penalty.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/neohookean_energy_vs_linear_and_volume_penalty.png")
