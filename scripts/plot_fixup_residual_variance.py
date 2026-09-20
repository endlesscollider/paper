#!/usr/bin/env python3
"""
生成 Fixup 论文核心理论图：残差网络输出方差随深度的增长曲线。

依据 Fixup 论文 Section 2 的分析：
  Var[x_{l+1}] = Var[x_l] + E[Var[F_l(x_l)|x_l]]
在标准初始化（如 He/Kaiming）下，每个残差分支的输出方差 ≈ 输入方差，
所以 Var[x_{l+1}] ≈ 2 * Var[x_l]，方差随深度指数增长（每个残差块翻倍）。

Fixup 的做法：把残差分支最后一层权重初始化为 0（分支起始贡献为零），
同时对分支内其余层按 L^{-1/(2m-2)} 缩放。这样在训练开始时分支贡献被压到
极小，方差不会随深度爆炸性增长，而是保持接近输入方差的稳定水平。

本图模拟 L=50 个残差块堆叠时，"标准初始化"和"Fixup 初始化"下网络中间
激活方差的变化，直观展示 Fixup 解决的问题。
"""
import numpy as np
import matplotlib.pyplot as plt

L = 50  # 残差块数
blocks = np.arange(0, L + 1)

# 标准初始化（如 He/Kaiming）：每个残差块方差近似翻倍：Var[x_{l+1}] = 2 * Var[x_l]
var_standard = np.array([2.0 ** l for l in blocks], dtype=float)

# Fixup 初始化：每个残差分支的输出在训练开始时被压缩为接近 0
# （最后一层权重清零），所以 Var[x_{l+1}] ≈ Var[x_l] + epsilon，近似保持稳定
# 这里用一个很小的残差贡献 epsilon=0.02 模拟"分支存在但贡献极小"
epsilon = 0.02
var_fixup = np.array([1.0 + epsilon * l for l in blocks], dtype=float)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(blocks, var_standard, '-o', color='#F44336', linewidth=2.3, markersize=3,
        label='Standard init (He/Kaiming), no normalization\nVar doubles per residual block')
ax.plot(blocks, var_fixup, '-s', color='#4CAF50', linewidth=2.3, markersize=3,
        label='Fixup init\nlast layer of each branch zero-initialized')

ax.set_yscale('log')
ax.set_xlim(0, L)
ax.set_xlabel('Residual block index (depth)', fontsize=11)
ax.set_ylabel('Activation variance Var[x_l] (log scale)', fontsize=11)
ax.set_title('Residual Network Activation Variance vs. Depth', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

ax.annotate(f'{var_standard[-1]:.2e}\n(explodes)', xy=(L, var_standard[-1]),
            xytext=(-80, -5), textcoords='offset points', fontsize=8.5, color='#F44336')
ax.annotate(f'{var_fixup[-1]:.1f}\n(stays bounded)', xy=(L, var_fixup[-1]),
            xytext=(-90, 8), textcoords='offset points', fontsize=8.5, color='#4CAF50')

plt.tight_layout()
plt.savefig('public/fixup_residual_variance_growth.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("saved public/fixup_residual_variance_growth.png")
print(f"standard final var (L={L}): {var_standard[-1]:.4e}")
print(f"fixup final var (L={L}): {var_fixup[-1]:.4f}")
