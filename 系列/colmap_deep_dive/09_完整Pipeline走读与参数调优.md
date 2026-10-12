---
title: "完整 Pipeline 走读与参数调优"
order: 9
tags: [COLMAP, Pipeline, 命令行, 参数调优, 故障排查, 实战, SfM, MVS]
category: 系列
series:
  id: colmap_deep_dive
  chapter: 9
---

# 完整 Pipeline 走读与参数调优

前八章逐个讲清了 COLMAP 每个阶段的原理。这一章是实战手册：从一批照片出发，用命令行跑完整个流程，理解每个命令在做什么，遇到问题知道从哪里下手。

## 目录结构约定

COLMAP 要求一个固定的工作目录结构：

```
project/
├── images/          # 输入图像（COLMAP 按文件名排序）
├── database.db      # 特征提取和匹配结果（自动创建）
├── sparse/          # SfM 输出
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── dense/           # MVS 输出（可选）
    └── 0/
        ├── images/      # 去畸变图像
        ├── sparse/      # 去畸变后的稀疏重建
        ├── stereo/      # 深度图和法向量图
        └── fused.ply    # 稠密点云
```

把照片放进 `images/`，其他目录由 COLMAP 自动创建。

## 完整命令流程

### 第一步：特征提取

```bash
colmap feature_extractor \
  --database_path project/database.db \
  --image_path project/images \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1
```

**关键参数**：
- `--ImageReader.camera_model SIMPLE_RADIAL`：所有图像共享同一相机模型，适合手机或同一台相机拍摄的照片。混用多台相机时改用 `PER_IMAGE`。
- `--ImageReader.single_camera 1`：所有图像共享同一组内参（焦距/畸变）。如果用了变焦镜头，设为 0 让每张图独立内参。
- `--SiftExtraction.use_gpu 1`：用 GPU。没有 NVIDIA GPU 时设为 0（慢 10-50 倍）。

运行后检查：`database.db` 文件出现，大小随图像数量增大。

### 第二步：特征匹配

**方案 A：穷举匹配**（≤ 100 张图）

```bash
colmap exhaustive_matcher \
  --database_path project/database.db \
  --SiftMatching.use_gpu 1
```

**方案 B：词袋匹配**（> 100 张图）

```bash
colmap vocab_tree_matcher \
  --database_path project/database.db \
  --VocabTreeMatching.vocab_tree_path /path/to/vocab_tree_flickr100K_words32K.bin \
  --VocabTreeMatching.num_nearest_neighbors 10 \
  --SiftMatching.use_gpu 1
```

词汇树文件从 [COLMAP 官网](https://colmap.github.io/faq.html) 下载，32K 版本约 100 MB，256K 版本约 1 GB，通常 32K 已经够用。

**方案 C：序列匹配**（视频帧或有序照片）

```bash
colmap sequential_matcher \
  --database_path project/database.db \
  --SequentialMatching.overlap 15 \
  --SiftMatching.use_gpu 1
```

### 第三步：增量式 SfM

```bash
colmap mapper \
  --database_path project/database.db \
  --image_path project/images \
  --output_path project/sparse
```

这是最耗时的步骤。100 张图通常需要几分钟到几十分钟，取决于图像分辨率和特征数量。进度输出示例：

```
...
Registering image #42 (image_042.jpg)
  => Image sees 312 / 3241 points3D
  => Registered 42 images with 28431 points3D
...
Bundle adjustment report:
  Residuals : 134231
  Initial cost: 0.621 [px]
  Final cost  : 0.183 [px]
  Termination : Convergence
```

`Final cost` < 0.5 像素说明重建质量好。

### 第四步（可选）：稠密重建

需要 NVIDIA GPU。

**去畸变**：

```bash
colmap image_undistorter \
  --image_path project/images \
  --input_path project/sparse/0 \
  --output_path project/dense/0 \
  --output_type COLMAP
```

**深度估计**：

```bash
colmap patch_match_stereo \
  --workspace_path project/dense/0 \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true
```

**深度融合**：

```bash
colmap stereo_fusion \
  --workspace_path project/dense/0 \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path project/dense/0/fused.ply
```

**（可选）泊松网格重建**：

```bash
colmap poisson_mesher \
  --input_path project/dense/0/fused.ply \
  --output_path project/dense/0/meshed-poisson.ply
```

## 用 COLMAP GUI 替代命令行

如果不熟悉命令行，COLMAP 提供了完整的 GUI：

```bash
colmap gui
```

GUI 里 `Reconstruction → Automatic Reconstruction` 会弹出对话框，填入工作目录和图像目录，勾选是否需要稠密重建，点击运行。适合快速验证，调参不如命令行方便。

## 故障排查手册

### 问题：mapper 找不到有效初始图像对

**症状**：`No good initial image pair found.` 或重建结果只有几张图。

**排查顺序**：
1. 检查匹配数量：打开 COLMAP GUI → Database → 查看 `two_view_geometries` 表，内点数少于 100 的图像对很难成为初始对
2. 图像重叠是否足够？相邻图像需要 60–80% 区域重叠
3. 场景是否纹理稀少？降低 `SiftExtraction.peak_threshold` 到 0.004
4. 降低初始化阈值：`--Mapper.init_min_num_inliers 50 --Mapper.init_min_tri_angle 8`（代价是初始位姿精度降低）

### 问题：部分图像未注册

**症状**：`num_reg_images` 远小于总图像数，`sparse/0/` 里 `images.bin` 里只有部分图像有位姿。

**排查顺序**：
1. 未注册图像是否有足够匹配？GUI 里查数据库
2. 是否因为相机位置太极端（几乎只看到已重建区域的背面）？
3. 降低 `--Mapper.min_num_inliers 15`，允许用更少内点注册
4. 如果是序列场景，考虑把 `sequential_matcher` 的 `overlap` 加大

### 问题：重建分裂成多个分量

**症状**：`sparse/` 目录下有 `0/`、`1/`、`2/` 多个子目录，各自只有部分图像。

**原因**：场景中有些区域和其他区域没有重叠图像，无法连通。

**解决**：
- 补拍过渡图像连接两个区域
- 检查匹配策略：穷举匹配保证不遗漏，词袋匹配可能漏掉跨区域的相似图像

### 问题：重投影误差偏高（> 1.5 像素）

**症状**：BA 报告 `Final cost > 1.5 [px]`，重建出的点云肉眼可见噪声。

**排查顺序**：
1. 相机模型选择是否合适？广角镜头用 `OPENCV` 或 `FULL_OPENCV`
2. 是否有移动物体（行人、车辆）出现在多张图中制造错误匹配？尽量用没有运动物体的照片
3. 图像是否有严重运动模糊？模糊图像的 SIFT 特征不稳定
4. 对焦是否准确？离焦图像产生不稳定特征

### 问题：稠密重建有大量空洞

**症状**：`fused.ply` 在白墙、天花板、地板等区域完全没有点。

**这是正常现象**，PatchMatch 无法从无纹理区域估计深度。

**缓解方案**：
- 在这些区域补贴或投影纹理（实验室环境下常用方法）
- 用泊松重建（`poisson_mesher`）填补孔洞，代价是这些区域的几何只是插值
- 接受空洞，下游用 NeRF/3DGS 这类方法会自行处理

## 性能优化

| 瓶颈 | 优化方法 |
|------|----------|
| 特征提取慢 | `use_gpu 1`，降低 `max_image_size` |
| 匹配慢 | 换词袋匹配，降低 `num_nearest_neighbors` |
| mapper 慢 | 降低 `max_num_features`（每张图特征点上限） |
| 深度估计显存不足 | 降低 `max_image_size`，减小 `window_radius` |
| 深度融合内存不足 | 降低 `check_num_images` |

## COLMAP 输出的文件格式

`sparse/0/` 下的三个文件是二进制格式（`.bin`），也可以导出为文本（`.txt`）：

```bash
colmap model_converter \
  --input_path project/sparse/0 \
  --output_path project/sparse/0_txt \
  --output_type TXT
```

文本格式方便调试，每行含义见下一章。

## 小结

COLMAP 的完整 Pipeline 是五个命令的串联：特征提取 → 特征匹配 → SfM → 去畸变 → 深度估计 → 深度融合。每个命令背后是前几章讲过的算法。故障排查的关键是定位是哪个步骤出了问题：匹配数量不足、初始化失败、注册率低、重投影误差高，每个症状对应不同的调参方向。

**知识链接**：
- → [COLMAP 输出与下游对接](/系列/colmap_deep_dive/10_COLMAP输出与下游对接) — 如何读取输出文件、对接 NeRF/3DGS

上一章：[深度图融合与稠密点云生成](./08_深度图融合与点云生成) | 下一章：[COLMAP 输出与下游对接](./10_COLMAP输出与下游对接)
