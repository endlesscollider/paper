#!/usr/bin/env python3
"""
可视化 TRELLIS 第一阶段输出的二值体素化占位掩码。

SparseStructureDecoder 输出 (B, 1, R, R, R) 的连续张量，
decoder(z_s) > 0 二值化后得到占位掩码，torch.argwhere 取出
非零坐标 coords: (N, 4)，形如 [batch_idx, x, y, z]。

这里用合成数据（带噪声的球体）模拟一个典型生成结果，
展示：3D 散点视图 + XY / XZ / YZ 三向最大值投影。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 合成占位掩码（模拟 decoder(z_s) > 0 的输出）─────────────────────────────
R = 32
cx = cy = cz = R / 2

xi, yi, zi = np.meshgrid(np.arange(R), np.arange(R), np.arange(R), indexing='ij')
dist = np.sqrt((xi - cx)**2 + (yi - cy)**2 + (zi - cz)**2)

# 实心球体，截掉底部模拟物体放在支撑面上
mask = (dist <= 13.0) & (zi >= 3)

# 少量表面噪声，让轮廓更接近真实生成结果
np.random.seed(42)
surface_band = (dist > 11.5) & (dist < 14.5)
noise = (np.random.rand(R, R, R) < 0.25) & surface_band
mask = mask | noise

coords = np.argwhere(mask)   # (N, 3)，对应 coords[:, 1:] 去掉 batch 维
N = coords.shape[0]
sparsity = N / R**3 * 100

# ── 图形布局 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.6))
fig.patch.set_facecolor('white')

ax3d  = fig.add_subplot(1, 4, 1, projection='3d')
ax_xy = fig.add_subplot(1, 4, 2)
ax_xz = fig.add_subplot(1, 4, 3)
ax_yz = fig.add_subplot(1, 4, 4)

# ── 3D 散点：z 坐标着色给出深度感 ─────────────────────────────────────────
c = coords[:, 2].astype(float)
ax3d.scatter(
    coords[:, 0], coords[:, 1], coords[:, 2],
    c=c, cmap='plasma', s=1.0, alpha=0.30, linewidths=0,
)
ax3d.set_xlim(0, R); ax3d.set_ylim(0, R); ax3d.set_zlim(0, R)
ax3d.set_xlabel('X', fontsize=8, labelpad=1)
ax3d.set_ylabel('Y', fontsize=8, labelpad=1)
ax3d.set_zlabel('Z', fontsize=8, labelpad=1)
ax3d.tick_params(labelsize=6)
ax3d.set_title(
    f'3D 体素分布\n{N} 个占位 / {R}³ 总格 ({sparsity:.1f}%)',
    fontsize=9.5, fontweight='bold', pad=4,
)
ax3d.view_init(elev=24, azim=-50)
for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor('#CCCCCC')
ax3d.grid(False)

# ── 三向最大值投影 ────────────────────────────────────────────────────────────
proj_configs = [
    (ax_xy, mask.max(axis=2).T, 'XY 投影 (沿 Z 轴)',   'X →', 'Y →'),
    (ax_xz, mask.max(axis=1).T, 'XZ 投影 (沿 Y 轴)',   'X →', 'Z →'),
    (ax_yz, mask.max(axis=0).T, 'YZ 投影 (沿 X 轴)',   'Y →', 'Z →'),
]

for ax, proj, title, xlabel, ylabel in proj_configs:
    ax.imshow(
        proj, origin='lower', cmap='Blues',
        vmin=0, vmax=1, interpolation='nearest',
        extent=[0, R, 0, R],
    )
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_edgecolor('#AAAAAA')

# 右侧三图共用颜色说明
fig.text(
    0.99, 0.5,
    '蓝色=有占位\n白色=空',
    va='center', ha='right', fontsize=8, color='#444444',
    style='italic',
)

plt.suptitle(
    'TRELLIS 第一阶段：体素化占位掩码 (SparseStructureDecoder 输出)',
    fontsize=11, fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(
    'public/trellis_occupancy_mask.png',
    dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none',
)
print("public/trellis_occupancy_mask.png")
