#!/usr/bin/env python3
"""
生成"不同初始化方法下，网络各层激活值方差随层数变化"对比图。

模拟设置：一个 20 层的全连接网络，每层宽度 n=512，权重从零均值高斯分布采样，
偏置为 0。对比四种权重方差设置：
  1. "too large"：Var(W) = 2/n * 5 （人为放大 5 倍，模拟初始化过大）
  2. "too small"：Var(W) = 2/n / 5 （人为缩小 5 倍，模拟初始化过小）
  3. Xavier：Var(W) = 1/n，激活函数为 tanh（近似线性区）
  4. Kaiming：Var(W) = 2/n，激活函数为 ReLU

每种设置下用解析递推公式计算方差随层数的变化（而不是随机模拟单次网络，
避免样本噪声），公式依据：
  - 线性/tanh 近似：Var(x_{l+1}) = n * Var(W) * Var(x_l)
  - ReLU：Var(x_{l+1}) = n * Var(W) * Var(x_l) / 2   （ReLU 砍掉一半方差）
"""
import numpy as np
import matplotlib.pyplot as plt

n = 512  # 每层宽度（输入维度=输出维度，简化为方阵）
num_layers = 20
layers = np.arange(0, num_layers + 1)

var0 = 1.0  # 输入方差归一化为 1

def propagate(var_w_scale_factor, relu):
    """按解析递推公式算方差随层数变化。
    var_w_scale_factor: 权重方差 = var_w_scale_factor / n
    relu: 是否过 ReLU（砍一半方差）
    """
    variances = [var0]
    v = var0
    for _ in range(num_layers):
        v = n * (var_w_scale_factor / n) * v  # = var_w_scale_factor * v
        if relu:
            v = v / 2.0
        variances.append(v)
    return np.array(variances)

# 1. 初始化过大：Var(W) = 10/n，配合 tanh（近似线性），方差每层放大 10 倍
var_too_large = propagate(var_w_scale_factor=10.0, relu=False)

# 2. 初始化过小：Var(W) = 0.2/n，配合 tanh，方差每层缩小到 0.2 倍
var_too_small = propagate(var_w_scale_factor=0.2, relu=False)

# 3. Xavier：Var(W) = 1/n，配合 tanh（近似线性），方差理论上每层保持 1 倍
var_xavier = propagate(var_w_scale_factor=1.0, relu=False)

# 4. Kaiming：Var(W) = 2/n，配合 ReLU，方差每层先乘 2 再砍半，保持 1 倍
var_kaiming = propagate(var_w_scale_factor=2.0, relu=True)

# 5. 对比：如果 ReLU 网络错误地使用 Xavier（Var(W)=1/n）而不是 Kaiming，
#    方差每层乘 1 再砍半 = 每层缩小到 0.5 倍，逐层衰减
var_xavier_with_relu = propagate(var_w_scale_factor=1.0, relu=True)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(layers, var_too_large, '-o', color='#F44336', linewidth=2.2, markersize=3,
        label='Too-large init (Var(W)=10/n, tanh)')
ax.plot(layers, var_too_small, '-o', color='#FF9800', linewidth=2.2, markersize=3,
        label='Too-small init (Var(W)=0.2/n, tanh)')
ax.plot(layers, var_xavier, '-s', color='#2196F3', linewidth=2.4, markersize=4,
        label='Xavier init (Var(W)=1/n, tanh)')
ax.plot(layers, var_kaiming, '-^', color='#4CAF50', linewidth=2.4, markersize=4,
        label='Kaiming init (Var(W)=2/n, ReLU)')
ax.plot(layers, var_xavier_with_relu, '--d', color='#9C27B0', linewidth=2.0, markersize=3,
        label='Xavier init misused with ReLU (Var(W)=1/n, ReLU)')

ax.axhline(y=1.0, color='#607D8B', linestyle=':', linewidth=1.2, alpha=0.7)
ax.set_yscale('log')
ax.set_xlim(0, num_layers)
ax.set_xlabel('Layer depth', fontsize=11)
ax.set_ylabel('Activation variance (log scale)', fontsize=11)
ax.set_title('Activation Variance vs. Depth under Different Initializations', fontsize=12, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper left' if var_too_large[-1] > 1 else 'lower left')
ax.grid(True, alpha=0.3, which='both')

# 标注关键数值点
ax.annotate(f'{var_too_large[-1]:.1e}', xy=(num_layers, var_too_large[-1]),
            xytext=(-55, 8), textcoords='offset points', fontsize=8, color='#F44336')
ax.annotate(f'{var_too_small[-1]:.1e}', xy=(num_layers, var_too_small[-1]),
            xytext=(-55, -12), textcoords='offset points', fontsize=8, color='#FF9800')
ax.annotate(f'{var_xavier[-1]:.2f} (stable)', xy=(num_layers, var_xavier[-1]),
            xytext=(-95, 6), textcoords='offset points', fontsize=8, color='#2196F3')
ax.annotate(f'{var_kaiming[-1]:.2f} (stable)', xy=(num_layers, var_kaiming[-1]),
            xytext=(-95, -14), textcoords='offset points', fontsize=8, color='#4CAF50')

plt.tight_layout()
plt.savefig('public/init_variance_propagation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("saved public/init_variance_propagation.png")
print(f"too_large final var: {var_too_large[-1]:.4e}")
print(f"too_small final var: {var_too_small[-1]:.4e}")
print(f"xavier final var: {var_xavier[-1]:.4f}")
print(f"kaiming final var: {var_kaiming[-1]:.4f}")
print(f"xavier_misused_with_relu final var: {var_xavier_with_relu[-1]:.4e}")
