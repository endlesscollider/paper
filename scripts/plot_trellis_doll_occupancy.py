#!/usr/bin/env python3
"""
从 doll.ply 的高斯中心坐标反推第一阶段体素化占位掩码并可视化。

TRELLIS 输出的高斯点云坐标在 [-0.5, 0.5]³ 归一化空间，
对应 SparseStructureDecoder 输出的稀疏体素位置。
把 xyz 量化回 R³ 网格即可近似还原中间 mask。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import struct, os

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 读取 PLY ──────────────────────────────────────────────────────────────────
PLY = '/home/flageval/projects/TRELLIS/outputs/doll/doll.ply'

with open(PLY, 'rb') as f:
    while f.readline().strip() != b'end_header':
        pass
    n = 748480
    dtype = np.dtype([
        ('x','<f4'),('y','<f4'),('z','<f4'),
        ('nx','<f4'),('ny','<f4'),('nz','<f4'),
        ('f_dc_0','<f4'),('f_dc_1','<f4'),('f_dc_2','<f4'),
        ('opacity','<f4'),
        ('scale_0','<f4'),('scale_1','<f4'),('scale_2','<f4'),
        ('rot_0','<f4'),('rot_1','<f4'),('rot_2','<f4'),('rot_3','<f4'),
    ])
    verts = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)

xyz     = np.stack([verts['x'], verts['y'], verts['z']], axis=1)   # (N,3)
opacity = verts['opacity']

# ── 用 opacity 过滤（sigmoid(opacity) > 0.5  ↔  opacity > 0）────────────────
# 低 opacity 的高斯对渲染几乎无贡献，过滤掉减少噪点
active = opacity > 0.0
xyz_active = xyz[active]
print(f"全部高斯: {n}，激活 (opacity>0): {active.sum()}")

# ── 体素化回 R³ 网格 ──────────────────────────────────────────────────────────
R = 64       # TRELLIS 第二阶段默认 64³ 空间
coords_norm = np.clip((xyz_active + 0.5) * R, 0, R - 1).astype(int)  # [0, R-1]

mask = np.zeros((R, R, R), dtype=bool)
mask[coords_norm[:, 0], coords_norm[:, 1], coords_norm[:, 2]] = True

occupied = mask.sum()
sparsity = occupied / R**3 * 100
print(f"占位体素: {occupied} / {R}³={R**3}  ({sparsity:.1f}%)")

coords_vis = np.argwhere(mask)   # (M, 3)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.8), facecolor='white')

ax3d  = fig.add_subplot(1, 4, 1, projection='3d')
ax_xy = fig.add_subplot(1, 4, 2)
ax_xz = fig.add_subplot(1, 4, 3)
ax_yz = fig.add_subplot(1, 4, 4)

# 3D 散点：z 着色
c = coords_vis[:, 2].astype(float)
ax3d.scatter(
    coords_vis[:, 0], coords_vis[:, 1], coords_vis[:, 2],
    c=c, cmap='plasma', s=0.6, alpha=0.25, linewidths=0,
)
ax3d.set_xlim(0, R); ax3d.set_ylim(0, R); ax3d.set_zlim(0, R)
ax3d.set_xlabel('X', fontsize=8, labelpad=1)
ax3d.set_ylabel('Y', fontsize=8, labelpad=1)
ax3d.set_zlabel('Z', fontsize=8, labelpad=1)
ax3d.tick_params(labelsize=6)
ax3d.set_title(
    f'doll — 3D 占位分布\n{occupied} 体素 / {R}³  ({sparsity:.1f}%)',
    fontsize=9.5, fontweight='bold', pad=4,
)
ax3d.view_init(elev=22, azim=-55)
for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor('#CCCCCC')
ax3d.grid(False)

# 三向最大值投影
proj_configs = [
    (ax_xy, mask.max(axis=2).T, 'XY 投影 (沿 Z)',  'X →', 'Y →'),
    (ax_xz, mask.max(axis=1).T, 'XZ 投影 (沿 Y)',  'X →', 'Z →'),
    (ax_yz, mask.max(axis=0).T, 'YZ 投影 (沿 X)',  'Y →', 'Z →'),
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

plt.suptitle(
    'TRELLIS doll — 体素化占位掩码（从高斯中心反推，opacity > 0）',
    fontsize=11, fontweight='bold', y=1.02,
)
plt.tight_layout()
out = 'public/trellis_doll_occupancy.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(out)
