---
title: "场景采集与重建：COLMAP + 3DGUT"
series:
  id: nvidia_real2sim2real_deep_dive
  chapter: 2
order: 2
---

# 第二章：场景采集与重建——COLMAP + 3DGUT

> **前情提要**：第一章讲清楚了 NuRec 由四个子组件组成——重建引擎、Asset Harvester、渲染引擎、Harmonizer。本章聚焦第一个、也是最基础的一环：怎么把一段拍摄数据，变成一个可以导入 Isaac Sim 的 USDZ 文件。本章所有命令都来自 NVIDIA 官方文档 [Reconstruct Scenes from Mono Camera Data](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html)，可以直接照着跑。

**知识链接**：
- [3D Gaussian Splatting：用一堆椭球把真实场景搬进电脑](/前置知识/007a_前置知识_3D_Gaussian_Splatting三维场景重建) — 3DGUT 是 3DGS 的直接技术扩展，本章会讲清楚具体扩展了什么

---

## 一、为什么是"3DGUT"而不是普通的 3DGS

### 1.1 普通 3DGS 的一个技术限制

[前置知识里讲过的 3D Gaussian Splatting](/前置知识/007a_前置知识_3D_Gaussian_Splatting三维场景重建#三渲染怎么把几十万个椭球变成一张2d图片)，把 3D 椭球投影到 2D 屏幕的方式叫 EWA splatting（椭圆加权平均投影）——这个方法假设相机是理想的针孔模型,并且用线性近似（雅可比矩阵）来计算投影。这个假设在大多数场景下没问题，但遇到两类情况会出问题：

1. **畸变镜头**（比如鱼眼相机、有明显桶形畸变的手机广角镜头）：EWA 的线性近似会产生明显误差,且需要针对每一种畸变模型单独推导对应的雅可比矩阵。
2. **滚动快门效应**（Rolling Shutter，相机传感器逐行曝光导致的运动畸变，常见于自动驾驶和机器人场景里高速移动的相机）：EWA 的静态投影假设无法处理这种随时间变化的效应。

### 1.2 3DGUT 的解法：用 Unscented Transform 替代线性近似

[3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/)（3D Gaussian Unscented Transform，CVPR 2025 Oral）用**Unscented Transform**（一种在卡尔曼滤波等领域常用的、不需要线性化就能近似非线性变换的数值方法）替代 EWA 的线性近似——用几个精心选取的"sigma 点"来代表整个高斯分布，让这些点分别经过任意非线性的相机投影函数（不管是针孔、鱼眼还是带滚动快门的模型），再用变换后的点重新估计出 2D 投影分布。

**这样做的好处**：不需要为每一种相机模型单独推导雅可比矩阵，同一套方法直接适配任意非线性投影,同时保持和原始 3DGS 相当的渲染速度（不像基于光线追踪的方法那样显著变慢）。这正是 NuRec 选择 3DGUT 而不是普通 3DGS 作为重建引擎的原因——机器人和自动驾驶场景里，用手机拍摄（有畸变）、用高速移动的传感器采集（有滚动快门）都是常态。

---

## 二、Step 1：拍摄真实场景

### 2.1 硬件要求

官方文档明确说明——单目重建工作流适配任意能拍照片的相机：智能手机（iPhone、Android）、DSLR、微单、运动相机，不需要专门的立体相机设备（这是给专业场景用的另一套 Stereo 工作流）。

### 2.2 拍摄规范（官方给出的具体参数）

| 项目 | 要求 |
|------|------|
| 光照与对焦 | 稳定光照、锁定对焦，避免快速运动导致的模糊 |
| 快门速度 | 建议 1/100 秒或更快 |
| 相机设置 | 锁定白平衡，避免不同照片间颜色偏移 |
| 拍摄路径 | 绕场景缓慢转一圈，覆盖多个高度和角度 |
| 重叠率 | 相邻照片之间保持约 60% 的重叠 |
| 数量 | 宁多勿少——COLMAP 能自动处理冗余照片 |

**一个容易踩坑的细节**：如果用 iPhone 拍摄，默认保存格式是 HEIC，COLMAP 不能直接处理，需要在设置里改成"最兼容"格式，或者拍完后转换成 JPG。

---

## 三、Step 2：用 COLMAP 生成稀疏重建

### 3.1 COLMAP 在做什么

COLMAP 是一套通用的 Structure-from-Motion（SfM）+ Multi-View Stereo（MVS）工具，这一步的任务是：从一堆照片里，反推出每张照片对应的精确相机位姿，以及场景的一个稀疏 3D 点云——这正是[前置知识里](/前置知识/007a_前置知识_3D_Gaussian_Splatting三维场景重建#41-训练的完整循环) 3DGS 训练需要的初始化数据。

### 3.2 命令行完整流程（官方文档原始命令）

```bash
# 第一步：特征提取
colmap feature_extractor \
    --database_path ./colmap/database.db \
    --image_path    ./images/ \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model   PINHOLE \
    --SiftExtraction.max_image_size 2000 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

# 第二步：特征匹配
colmap exhaustive_matcher \
    --database_path ./colmap/database.db \
    --SiftMatching.use_gpu 1

# 第三步：全局 SfM 重建
colmap mapper \
    --database_path ./colmap/database.db \
    --image_path    ./images/ \
    --output_path   ./colmap/sparse

# 第四步：可视化验证
colmap gui --import_path ./colmap/sparse/0 \
    --database_path ./colmap/database.db \
    --image_path    ./images/
```

**关键参数说明**：
- `--ImageReader.camera_model PINHOLE`：**必须**选择 `PINHOLE` 或 `SIMPLE_PINHOLE` 相机模型——这是为了和后面的 3DGUT 兼容（3DGUT 虽然能处理非线性投影，但 COLMAP 这一步的相机模型设置仍然要求这个标准格式）。
- `--SiftMatching.use_gpu 1`：用 GPU 加速特征匹配，大幅提升处理速度。

### 3.3 输出内容

跑完之后，会得到：
- 场景的稀疏点云
- 所有输入照片对应的相机位姿
- 一个项目文件夹，包含 `database.db`（COLMAP 数据库）、`images/`（原始照片）、`sparse/`（重建数据：`cameras.txt`、`images.txt`、`points3D.txt`）

这份数据就是下一步 3DGUT 训练需要的全部输入。

---

## 四、Step 3：用 3DGUT 训练稠密重建

### 4.1 环境配置

```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut
chmod +x install_env.sh
./install_env.sh 3dgrut
conda activate 3dgrut
```

3DGUT 要求 Linux 系统、CUDA 11.8、GCC ≤ 11。如果系统自带 GCC 12+（比如 Ubuntu 24.04），需要在 conda 环境里单独装一个 GCC 11：

```bash
conda install -c conda-forge gcc=11 gxx=11
```

### 4.2 训练命令：COLMAP + 3DGUT + MCMC

```bash
conda activate 3dgrut
python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
    path=/path/to/colmap/ \
    out_dir=/path/to/out/ \
    experiment_name=3dgut_mcmc \
    export_usdz.enabled=true \
    export_usdz.apply_normalizing_transform=true
```

**参数说明**：
- `config-name`：选择 `colmap_3dgut_mcmc.yaml` 配置，启用 **MCMC**（Markov Chain Monte Carlo）密度化策略——这个策略会自适应地在"不确定区域"多采样，让细结构（比如物体边缘）重建得更清晰。
- `export_usdz.enabled=true`：训练完成后直接导出 USDZ 文件，给 Isaac Sim 用。
- `export_usdz.apply_normalizing_transform=true`：让场景居中、缩放到合理范围（但**不保证**地面正好在 z=0，后面第四章导入 Isaac Sim 时可能还需要手动微调）。

### 4.3 训练过程与输出

训练时长从几分钟（小场景）到几小时（复杂场景）不等，取决于场景复杂度。完成后输出目录包含：

| 文件 | 内容 |
|------|------|
| `ckpt_last.pt` | 最终模型检查点 |
| `export_last.usdz` | **给 Isaac Sim 用的最终 USDZ 文件** |
| `parsed.yaml` | 训练配置记录 |
| `ours_xxxxx/` | 每个训练 checkpoint 的中间输出 |

---

## 五、常见问题排查（官方文档提供的排查清单）

| 阶段 | 问题 | 解决方案 |
|------|------|---------|
| COLMAP | 重建失败 | 确认重叠率≥60%、照片无运动模糊、相机设置一致；尝试切换 `pinhole` / `simple pinhole` 模型 |
| COLMAP | 稀疏点云不完整 | 补拍缺失角度的照片；增大特征提取的 `max_num_features` |
| 3DGUT | GPU 显存不足 | 降低 COLMAP 重建前的图片分辨率；换更大显存的 GPU |
| 3DGUT | 重建质量差 | 确认 COLMAP 重建质量本身够高；用 MCMC 配置；增加训练迭代数 |
| Isaac Sim | 场景比例不对 | 确认已启用 `apply_normalizing_transform`；在 Isaac Sim 里手动缩放场景根节点 |

---

## 六、本章小结

| 步骤 | 工具 | 输入 | 输出 |
|------|------|------|------|
| 拍摄 | 手机/相机 | — | 一批照片（60%重叠、稳定光照） |
| 稀疏重建 | COLMAP | 照片 | 相机位姿 + 稀疏点云 |
| 稠密重建 | 3DGUT | COLMAP 输出 | 3D 高斯表示 + USDZ 文件 |

## 下章预告

第三章要讲的是 NuRec 流水线里另外两个子组件——**Asset Harvester**（怎么把场景里的具体物体单独提取成可编辑的 3D 资产）和**Harmonizer**（怎么用扩散模型让渲染画面更逼真、更时序一致）。

---

## 延伸阅读

- [Reconstruct Scenes from Mono Camera Data（官方文档，本章命令来源）](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html)
- [3DGUT 论文与项目页](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
- [3dgrut 代码仓库](https://github.com/nv-tlabs/3dgrut)
- [3D Gaussian Splatting：用一堆椭球把真实场景搬进电脑](/前置知识/007a_前置知识_3D_Gaussian_Splatting三维场景重建)
