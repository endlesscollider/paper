#!/usr/bin/env python3
"""
生成"应变能量随拉伸量变化"的曲线图，对比：
- 线性弹簧能量（胡克定律，对压缩和拉伸对称，且允许穿过0变成负体积）
- 更真实的超弹性能量形式（对压缩到0体积附近急剧增大，惩罚"压穿"）
用来直观说明为什么简单的线性弹簧模型不能直接套用到大变形的软体材料上。
"""
import numpy as np
import matplotlib.pyplot as plt

lam = np.linspace(0.05, 2.5, 500)  # 拉伸比 lambda = 当前长度 / 原长

k = 1.0
energy_linear = 0.5 * k * (lam - 1) ** 2          # 线性弹簧：只关心偏离1多少
energy_hyperelastic = 0.5 * k * ((lam - 1) ** 2 + 0.6 * (np.log(lam)) ** 2 / lam)  # 简化示意：压缩时急剧增大

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(lam, energy_linear, color='#9E9E9E', linewidth=2.0, linestyle='--',
        label='Linear spring energy $\\frac{1}{2}k(\\lambda-1)^2$')
ax.plot(lam, energy_hyperelastic, color='#F44336', linewidth=2.4,
        label='Hyperelastic-like energy (penalizes $\\lambda\\to0$)')

ax.axvline(x=1.0, color='#4CAF50', linestyle=':', linewidth=1.5, label='$\\lambda=1$ (rest length)')
ax.axvspan(0, 0.15, color='#FFCDD2', alpha=0.4)
ax.text(0.06, ax.get_ylim()[1]*0.75, 'inverted /\ncrushed\nvolume', fontsize=8.5, color='#B71C1C')

ax.set_xlabel('Stretch ratio $\\lambda$ = current length / rest length', fontsize=11)
ax.set_ylabel('Strain energy density', fontsize=11)
ax.set_title('Why Linear Springs Are Not Enough for Large Deformation', fontsize=12, fontweight='bold')
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 3.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc='upper center')

plt.tight_layout()
plt.savefig('public/strain_energy_linear_vs_hyperelastic.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ public/strain_energy_linear_vs_hyperelastic.png")
