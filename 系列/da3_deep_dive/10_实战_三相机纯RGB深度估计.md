---
title: "实战：三相机纯 RGB 深度估计"
order: 3130
tags: [深度估计, 推理, API, 三相机, 点云, 3DGS, 实战]
category: 系列
series:
  id: da3_deep_dive
  chapterIndex: 10
  dir: /系列/da3_deep_dive
---

# 第十章：实战——三相机纯 RGB 深度估计

> 本系列用三台水平排列的 RGB 相机作为贯穿全文的运行示例。本章把这个示例从概念落地到代码：从安装、加载模型，到输入图像和位姿，再到读取深度、外参和高斯球输出。每个 API 细节都对应前九章的某个设计决定。

---

## 场景设定

三台相机水平排列，正前方相机为参考视角，左右相机分别旋转 ±30°。相机参数如下：

| 相机 | 位置（世界坐标） | 焦距 | 图像尺寸 |
|------|----------------|------|---------|
| 左（cam0） | $(-0.2, 0, 0)$ | $f = 500$ px | $518 \times 518$ |
| 中（cam1，参考） | $(0, 0, 0)$ | $f = 500$ px | $518 \times 518$ |
| 右（cam2） | $(+0.2, 0, 0)$ | $f = 500$ px | $518 \times 518$ |

整篇文章里出现的"三相机"、"$S=3$"、"$f=500$"都指这个配置。

---

## 安装

```bash
# 克隆仓库并安装
git clone https://github.com/DepthAnything/Depth-Anything-V3
cd Depth-Anything-V3
pip install -e .

# 下载权重（以 DA3NESTED-GIANT-LARGE 为例）
# 模型文件会放到 ~/.cache/huggingface/hub/
```

DA3 的权重通过 HuggingFace Hub 分发，模型 ID 在 README 里列出。推荐用 DA3NESTED-GIANT-LARGE-1.1 做生产推理（多视角 + 米制深度），用 DA3-GIANT 做纯几何任务。

---

## 加载模型

```python
from depth_anything_3 import DepthAnything3

# 加载 DA3-Nested（Giant 主系列 + Large Metric，米制深度输出）
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
model = model.cuda().eval()
```

`DepthAnything3.from_pretrained` 根据 config 里的 `_target_` 字段自动选择 `DepthAnything3Net` 或 `NestedDepthAnything3Net`——不需要手动选择类。加载后的模型在 `eval()` 模式下运行，BN 和 Dropout 固定。

---

## 准备输入

DA3 的输入接口接受：
- **图像张量** `[B, S, 3, H, W]`，值域 `[0, 1]`，RGB 顺序
- **可选外参** `[B, S, 4, 4]`，w2c 格式（世界到相机变换）
- **可选内参** `[B, S, 3, 3]`，像素单位

### 图像

```python
import torch
from torchvision import transforms

to_tensor = transforms.ToTensor()  # HxWxC uint8 → CxHxW float [0,1]

# images: List[PIL.Image]，长度 S=3，已 resize 到 518x518
images = [load_and_resize(path, 518) for path in image_paths]
imgs = torch.stack([to_tensor(img) for img in images])  # [3, 3, 518, 518]
imgs = imgs.unsqueeze(0).cuda()                          # [1, 3, 3, 518, 518]
```

DA3 的默认训练分辨率是 $518 \times 518$，输入其他分辨率也能运行（backbone 是 ViT，支持任意尺寸），但偏离太多会影响质量。

### 已知位姿（机器人或标定相机）

```python
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def make_c2w(tx, ty, tz, rot_y_deg):
    """构建 c2w 矩阵：相机在世界坐标中的位置和朝向"""
    r = R_scipy.from_euler('y', rot_y_deg, degrees=True).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = r
    c2w[:3,  3] = [tx, ty, tz]
    return c2w

def c2w_to_w2c(c2w):
    w2c = np.linalg.inv(c2w)
    return w2c

# 三台相机的 w2c 外参
extrinsics_np = np.stack([
    c2w_to_w2c(make_c2w(-0.2, 0, 0, +30)),  # 左相机
    c2w_to_w2c(make_c2w( 0.0, 0, 0,   0)),  # 中相机（参考）
    c2w_to_w2c(make_c2w(+0.2, 0, 0, -30)),  # 右相机
])  # [3, 4, 4]

# 三台相机的内参（焦距 f=500，主点在图像中心）
H, W = 518, 518
fx = fy = 500.0
K = np.array([[fx, 0, W/2],
              [0, fy, H/2],
              [0,  0,   1]])
intrinsics_np = np.stack([K, K, K])  # [3, 3, 3]

extrinsics = torch.from_numpy(extrinsics_np).float().unsqueeze(0).cuda()  # [1, 3, 4, 4]
intrinsics = torch.from_numpy(intrinsics_np).float().unsqueeze(0).cuda()  # [1, 3, 3, 3]
```

外参传 `w2c` 格式（世界到相机），这和大多数计算机视觉库的惯例一致，也和 DA3 的 `CameraEnc` 接口（第五章）保持一致——`CameraEnc` 第一步就把 w2c 转成 c2w。

### 未知位姿（普通照片）

```python
# 不传外参时，模型自动估计位姿
extrinsics = None
intrinsics = None
```

这时模型走 camray → RANSAC 路径（第二章），从预测的深度光线还原相机位姿。

---

## 运行推理

```python
with torch.inference_mode():
    output = model(
        images=imgs,
        extrinsics=extrinsics,       # [1, 3, 4, 4] w2c，或 None
        intrinsics=intrinsics,       # [1, 3, 3, 3]，或 None
        infer_gs=True,               # 是否同时输出 3D 高斯球
        ref_view_index=1,            # 参考视角索引（中相机）
    )
```

`inference_mode()` 禁用梯度计算，比 `no_grad()` 更彻底，推理速度更快、显存更低。

`infer_gs=False` 时 GS 头（第八章的 GSDPT + GaussianAdapter）完全不运行，节省约 20% 的显存和时间。如果只需要深度和位姿，关掉它。

---

## 读取输出

`model()` 返回一个 `Prediction` 数据类（`specs.py`），字段如下：

```python
output.depth          # [1, 3, 518, 518]   — 每像素深度，米（DA3-Nested）或相对单位
output.is_metric      # bool               — True 表示深度单位是米
output.conf           # [1, 3, 518, 518]   — 深度置信度（expp1 激活，≥1）
output.sky            # [1, 3, 518, 518]   — 天空 mask（概率，越大越可能是天空）
output.extrinsics     # [1, 3, 4, 4]       — w2c 外参（输入已知则直接复用，未知则估计）
output.intrinsics     # [1, 3, 3, 3]       — 内参矩阵（像素单位）
output.gaussians      # Gaussians          — 仅 infer_gs=True 时有效
output.scale_factor   # float              — 深度从相对单位到米的缩放系数（已应用）
```

### 深度图

```python
depth = output.depth[0]        # [3, 518, 518]，三张深度图
depth_center = depth[1]        # [518, 518]，中相机深度图

# 可视化（伪彩色）
import matplotlib.pyplot as plt
plt.imshow(depth_center.cpu().numpy(), cmap='plasma')
plt.colorbar(label='depth (m)' if output.is_metric else 'depth (relative)')
plt.savefig('depth_center.png')
```

`output.is_metric` 为 `True` 时（DA3-Nested 路径），值域通常在 0 到几十米，可以直接用于机器人导航和三维重建。

### 置信度引导采样

置信度 `conf` 可以用来过滤低质量深度点：

```python
conf = output.conf[0]           # [3, 518, 518]，≥ 1.0
threshold = conf.median()       # 或者用固定阈值如 1.5
mask = conf > threshold         # [3, 518, 518]

# 只保留高置信度区域的深度
depth_filtered = depth.clone()
depth_filtered[~mask] = float('nan')
```

这和 `compute_alignment_mask`（第七章）的第一个条件完全一致——评测时也是用置信度中位数作为基准。

### 位姿

```python
extrinsics_out = output.extrinsics[0]   # [3, 4, 4]，w2c
intrinsics_out = output.intrinsics[0]   # [3, 3, 3]

# 取中相机的内参
K_center = intrinsics_out[1]            # [3, 3]

# 取左相机的旋转和平移
w2c_left = extrinsics_out[0]
c2w_left = torch.linalg.inv(w2c_left)
R_left = c2w_left[:3, :3]             # 旋转矩阵
t_left = c2w_left[:3, 3]              # 相机原点在世界坐标系中的位置（米）
```

如果输入时提供了已知外参，`output.extrinsics` 和输入完全一致（DA3 不会修改已知位姿，只在 `extrinsics=None` 时估计）。

---

## 生成点云

把三张深度图反投影到世界坐标系，拼成一个统一的点云：

```python
import torch
from depth_anything_3.utils.geometry import get_world_rays, sample_image_grid

B, S, H, W = 1, 3, 518, 518

# 像素坐标（归一化）
xy_ray, _ = sample_image_grid((H, W), device='cuda')  # [H, W, 2]

# 归一化内参：K / [W, H]
intr_normed = intrinsics_out.clone()
intr_normed[:, 0] /= W
intr_normed[:, 1] /= H

# c2w 外参
c2w = torch.linalg.inv(extrinsics_out)  # [3, 4, 4]

# 获取世界空间射线
origins, directions = get_world_rays(
    xy_ray.unsqueeze(0).expand(S, -1, -1, -1),  # [3, H, W, 2]
    c2w,
    intr_normed,
)
# origins: [3, H, W, 3]   相机原点（三台相机各自的原点）
# directions: [3, H, W, 3]  归一化射线方向

# 反投影：三维坐标 = 原点 + 深度 × 方向
depth = output.depth[0]                            # [3, H, W]
points = origins + directions * depth.unsqueeze(-1)  # [3, H, W, 3]

# 展平成点云
xyz = points.reshape(S * H * W, 3)                 # [N, 3]，N = 805,932

# 对应的 RGB 颜色（从输入图像取）
rgb = imgs[0].permute(0, 2, 3, 1)                  # [3, H, W, 3]
colors = rgb.reshape(S * H * W, 3)                 # [N, 3]，值域 [0,1]

# 导出为 PLY
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
o3d.io.write_point_cloud("output.ply", pcd)
```

80 万个点的点云约 30MB（以 float32 存储 XYZ + RGB）。

---

## 使用 3D 高斯球输出

`output.gaussians` 是 `Gaussians` 数据类，包含世界空间的 3DGS 参数（第八章）：

```python
gaussians = output.gaussians

means      = gaussians.means[0]       # [805932, 3]   世界空间坐标（米）
scales     = gaussians.scales[0]      # [805932, 3]   三轴尺度
rotations  = gaussians.rotations[0]   # [805932, 4]   四元数 WXYZ
harmonics  = gaussians.harmonics[0]   # [805932, 3, 1] DC 球谐系数（sh_degree=0）
opacities  = gaussians.opacities[0]   # [805932]       透明度 ∈ (0,1)
```

### 用 gsplat 渲染新视角

```python
import gsplat

# 定义一个新视角（比如从上方俯视）
camera_novel = {
    'c2w': novel_c2w,          # [4, 4]
    'K': K_center,             # [3, 3]
    'width': W,
    'height': H,
}

# 渲染
rendered_rgb, rendered_alpha, meta = gsplat.rasterization(
    means=means,
    quats=rotations,           # gsplat 用 WXYZ 四元数
    scales=scales,
    opacities=opacities,
    colors=harmonics[..., 0],  # DC 分量作为颜色（视角无关）
    viewmats=novel_c2w.inverse().unsqueeze(0),  # [1, 4, 4]
    Ks=K_center.unsqueeze(0),  # [1, 3, 3]
    width=W,
    height=H,
)
# rendered_rgb: [1, H, W, 3]
```

DA3 的 `rotations` 已经是世界空间 WXYZ 四元数，满足 gsplat 接口要求，不需要额外转换。

### 导出为 3DGS 标准格式

```python
# 导出为 .ply（兼容 3DGS 查看器如 SuperSplat、Luma AI）
from plyfile import PlyData, PlyElement
import numpy as np

# 3DGS 存储格式：f_dc_0/1/2 是 SH DC 系数，rot_0/1/2/3 是四元数 WXYZ
vertex_data = np.zeros(len(means.cpu()),
    dtype=[('x','f4'),('y','f4'),('z','f4'),
           ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
           ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
           ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4'),
           ('opacity','f4')])

m = means.cpu().numpy()
vertex_data['x'], vertex_data['y'], vertex_data['z'] = m[:,0], m[:,1], m[:,2]
h = harmonics[:,0,:].cpu().numpy()  # DC 分量
vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2'] = h[:,0], h[:,1], h[:,2]
s = scales.cpu().numpy()
vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2'] = s[:,0], s[:,1], s[:,2]
r = rotations.cpu().numpy()  # WXYZ
vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3'] = r[:,0],r[:,1],r[:,2],r[:,3]
vertex_data['opacity'] = opacities.cpu().numpy()

el = PlyElement.describe(vertex_data, 'vertex')
PlyData([el]).write('output_gs.ply')
```

---

## TSDF 融合（生成三角网格）

对应第九章的评测流程——把三张深度图融合成三角网格：

```python
import open3d as o3d
import numpy as np

# TSDF 体积（参数参考 7Scenes：体素 0.0078m，截断 3×体素 ≈ 0.024m）
voxel_size = 0.01
tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size,
    sdf_trunc=3 * voxel_size,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

depth_np = output.depth[0].cpu().numpy()    # [3, H, W]
images_np = (imgs[0].permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)  # [3, H, W, 3]
extr_np = extrinsics_out.cpu().numpy()       # [3, 4, 4] w2c
intr_np = intrinsics_out.cpu().numpy()       # [3, 3, 3]

for i in range(3):
    depth_o3d = o3d.geometry.Image(depth_np[i].astype(np.float32))
    color_o3d = o3d.geometry.Image(images_np[i])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,           # 已经是米
        depth_trunc=5.0,           # 截断 5 米以外
        convert_rgb_to_intensity=False,
    )
    extr_o3d = o3d.camera.PinholeCameraIntrinsic(
        W, H,
        intr_np[i][0,0], intr_np[i][1,1],  # fx, fy
        intr_np[i][0,2], intr_np[i][1,2],  # cx, cy
    )
    tsdf.integrate(rgbd, extr_o3d, extr_np[i])  # 传入 w2c

mesh = tsdf.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("output_mesh.ply", mesh)
```

---

## 参数调整指南

### 选择模型

| 需求 | 推荐模型 | 备注 |
|------|---------|------|
| 米制深度 + 多视角一致 | DA3NESTED-GIANT-LARGE | 最高质量，需双模型显存 |
| 纯多视角相对深度 | DA3-GIANT | 最大单模型，不含 Metric |
| 快速推理 / 低显存 | DA3-LARGE | Giant 的 1/3 参数量 |
| 单图深度 | DA3-MONO-LARGE | 无跨视角注意力 |

### `ref_view_index` 的选择

参考视角是 camray RANSAC 的基准视角（第二章），决定了世界坐标系的原点和朝向：

```python
output = model(images=imgs, extrinsics=extrinsics, ref_view_index=1)
# 0 = 左相机为参考，1 = 中相机（推荐），2 = 右相机
```

对称配置用中间视角（`ref_view_index=1`），让左右相机均等地参与多视角注意力。评测时固定用 `first`（第 0 帧），和第九章的 `EVAL_REF_VIEW_STRATEGY` 一致。

### `infer_gs` 开关

```python
# 只需要深度和位姿（更快）
output = model(images=imgs, extrinsics=extrinsics, infer_gs=False)

# 同时需要可渲染场景
output = model(images=imgs, extrinsics=extrinsics, infer_gs=True)
```

GS 头（GSDPT + GaussianAdapter，第八章）在 `infer_gs=False` 时完全跳过，不影响深度和位姿的精度。

### 批量推理

```python
# 同时处理多个场景（B > 1）
imgs_batch   = torch.stack([imgs_scene0, imgs_scene1])   # [2, 3, 3, H, W]
extr_batch   = torch.stack([extrinsics0, extrinsics1])   # [2, 3, 4, 4]
intr_batch   = torch.stack([intrinsics0, intrinsics1])   # [2, 3, 3, 3]

with torch.inference_mode():
    output = model(images=imgs_batch, extrinsics=extr_batch, intrinsics=intr_batch)
# output.depth: [2, 3, H, W]
```

批量推理是 GPU 利用率最高的方式，但显存随 B 线性增长。ViT backbone（第三章）支持可变的 S 维度（视角数），但批内的 S 必须一致。

---

## 常见问题

**深度图有大面积 nan 或 0 怎么办**

检查输入图像的值域是否在 `[0, 1]`，以及分辨率是否与内参匹配。DA3 训练时输入是 RGB `[0, 1]`，如果意外传入了 `[0, 255]`，深度预测会严重退化。

**`is_metric=False`，尺度对齐失败**

DA3-Nested 的 `compute_alignment_mask` 没有找到足够多的可靠像素——可能是：（1）场景几乎全是天空；（2）所有像素的深度置信度都低于中位数（极端纹理缺乏场景）；（3）内参传入了错误值导致焦距缩放异常。检查 `output.sky` 和 `output.conf` 的分布。

**点云有明显断裂或跳变**

通常是外参坐标系不一致——DA3 期望 `extrinsics` 是 w2c 格式的 `[4, 4]` 齐次矩阵，如果传入了 c2w 或者只有 `[3, 4]`，反投影结果会出错。用 `np.linalg.inv` 显式转换。

**高斯球渲染颜色偏暗或偏灰**

DA3 的球谐系数是 DC 分量（零阶，视角无关），没有经过 3DGS 优化里的 view-dependent 训练。渲染质量天然低于充分优化的 3DGS，可以把 DA3 的输出作为初始化，再用 gsplat 做 500~1000 步微调。

---

## 完整示例：端到端三相机推理

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from depth_anything_3 import DepthAnything3

# 1. 加载模型
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
model = model.cuda().eval()

# 2. 准备图像
to_tensor = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
])
image_paths = ["cam0.jpg", "cam1.jpg", "cam2.jpg"]
imgs = torch.stack([to_tensor(Image.open(p)) for p in image_paths])
imgs = imgs.unsqueeze(0).cuda()  # [1, 3, 3, 518, 518]

# 3. 准备已知位姿（标定相机）
def build_w2c(tx, tz, rot_y_deg):
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler('y', rot_y_deg, degrees=True).as_matrix()
    c2w = np.eye(4); c2w[:3,:3] = r; c2w[:3,3] = [tx, 0, tz]
    return np.linalg.inv(c2w)

extr = np.stack([build_w2c(-0.2, 0, 30),
                 build_w2c( 0.0, 0,  0),
                 build_w2c(+0.2, 0,-30)])
extrinsics = torch.from_numpy(extr).float().unsqueeze(0).cuda()

H, W, f = 518, 518, 500.0
K = np.array([[f,0,W/2],[0,f,H/2],[0,0,1]])
intrinsics = torch.from_numpy(np.stack([K,K,K])).float().unsqueeze(0).cuda()

# 4. 推理
with torch.inference_mode():
    output = model(
        images=imgs,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        infer_gs=True,
        ref_view_index=1,
    )

# 5. 读取结果
print(f"米制深度: {output.is_metric}")
print(f"深度范围: {output.depth.min():.2f} ~ {output.depth.max():.2f} m")
print(f"高斯球数量: {output.gaussians.means.shape[1]}")
```

---

## 本章小结

从安装到输出，三相机 DA3 推理的完整流程：

输入端，图像张量 `[B, S, 3, H, W]` 和可选的 w2c 外参与内参；已知位姿时走 CameraEnc 注入（第五章），未知位姿时模型自动走 RANSAC 估计（第二章）。

输出端，`output.depth` 是米制深度图（DA3-Nested），`output.extrinsics` 是 w2c 外参，`output.gaussians` 是可直接送入 gsplat 的 3DGS 参数。深度反投影得到彩色点云，Open3D TSDF 融合得到三角网格，gsplat 渲染得到任意新视角图像。

前九章覆盖了每个组件的设计原理；本章的代码只是把这些设计连在一起，用三台相机走了一遍完整的推理路径。

::: details 📎 知识链接

- [第二章：深度光线表示](./02_深度光线表示_统一单目与多视角) — 未知位姿时的 camray → RANSAC 路径
- [第五章：相机编解码器](./05_相机编解码器_位姿怎么进出模型) — CameraEnc 把 w2c 编码为 cam_token 注入 backbone
- [第七章：DA3-Nested 从相对深度到米制深度](./07_DA3Nested_从相对深度到米制深度) — `is_metric=True` 的来源；`output.depth` 的米制单位
- [第八章：3D 高斯头](./08_3D高斯头_从深度直接到可渲染场景) — `output.gaussians` 的结构和 gsplat 兼容格式
- [第九章：评测体系 DA3-BENCH](./09_评测体系_DA3-BENCH六大数据集) — TSDF 融合参数的来源；`ref_view_strategy="first"` 和本章 `ref_view_index=1` 的关系

:::
