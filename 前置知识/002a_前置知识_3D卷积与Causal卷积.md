---
title: "3D 卷积与 Causal 卷积：视频模型的基础算子"
order: 280
tags: [卷积, 3D卷积, Causal卷积, 视频生成, VAE]
category: 深度学习基础
---

# 3D 卷积与 Causal 卷积：视频模型的基础算子

> 为什么视频 VAE 不能用普通 2D 卷积？3D 卷积解决了什么问题？Causal 又是什么意思？本文从最基础的卷积出发，一步步讲到 3D Causal Convolution 的设计。

**知识链接**：
- 本文是 [fast-WAM 人类数据 World Model 实验全记录](/系列/fastwam_human_pretrain) 系列第 2 章的前置知识
- 相关：[Causal Attention 因果注意力掩码](/前置知识/001g_前置知识_Causal_Attention因果注意力掩码)（注意力层面的因果性）

---

## 1. 从 1D 卷积开始

### 1.1 最基本的概念

卷积的本质是**滑动窗口 + 加权求和**。

给一个 1D 序列 `[a, b, c, d, e]` 和一个 kernel_size=3 的卷积核 `[w1, w2, w3]`：

```
位置 0: w1*a + w2*b + w3*c = 输出[0]
位置 1: w1*b + w2*c + w3*d = 输出[1]
位置 2: w1*c + w2*d + w3*e = 输出[2]
```

卷积核在输入序列上滑动，每到一个位置就和局部窗口做加权求和，产生一个输出值。

### 1.2 Padding 的作用

上面的例子中，输入长度 5，输出长度 3——输出变短了。如果想让输出和输入一样长，需要在输入两端补零（padding）：

```
padding=1: [0, a, b, c, d, e, 0]
位置 0: w1*0 + w2*a + w3*b = 输出[0]  ← 能看到位置 0 的前后
位置 1: w1*a + w2*b + w3*c = 输出[1]
...
位置 4: w1*d + w2*e + w3*0 = 输出[4]
```

这里 padding=1 就是左右各补 1 个零。这种**对称 padding** 意味着每个位置的输出同时利用了"过去"和"未来"的信息。

---

## 2. 什么是 Causal 卷积？

### 2.1 问题：对称卷积"看到了未来"

在语音、时间序列、视频等有**时间顺序**的数据中，有时候我们要求：**计算当前时刻的输出时，不能使用未来时刻的信息。**

为什么？举个例子：
- 视频生成的 Image-to-Video 模式：给定第 1 帧，预测后续帧。此时第 1 帧的编码不应该依赖第 2、3、4 帧的像素——因为那些帧还没被生成！
- 自回归语言模型（GPT）：预测第 5 个词时不能看到第 6、7、8 个词

### 2.2 解决方案：只在左边 padding

**Causal（因果）卷积**的做法非常简单：**把 padding 全部放到左边（过去方向），右边（未来方向）不 pad。**

```
普通卷积 (kernel=3, padding=1):   pad=[1, 1]  → 每个位置看 [t-1, t, t+1]
Causal 卷积 (kernel=3, padding=2): pad=[2, 0]  → 每个位置看 [t-2, t-1, t]
```

对比一下：

```
普通卷积:
  输出[2] = w1*输入[1] + w2*输入[2] + w3*输入[3]  ← 用到了"未来"的输入[3]

Causal 卷积:
  输出[2] = w1*输入[0] + w2*输入[1] + w3*输入[2]  ← 只用"过去"和"当前"
```

### 2.3 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """1D 因果卷积：输出只依赖当前和过去的输入"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # 左侧 pad = kernel_size - 1，右侧 pad = 0
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=0  # 不使用 Conv1d 自带的 padding
        )
    
    def forward(self, x):
        # x: [B, C, T]
        # 只在左侧（时间方向的"过去"）填充零
        x = F.pad(x, (self.pad, 0))  # [B, C, T + pad]
        return self.conv(x)           # [B, C_out, T]  ← 输出长度不变
```

**验证因果性**：

```python
# kernel_size=3 的 causal conv
# 输出[t] = w1*输入[t-2] + w2*输入[t-1] + w3*输入[t]
# 
# 特别地，输出[0] = w1*pad + w2*pad + w3*输入[0]
# → 第一个时间步的输出只依赖第一个输入（pad 是零）
```

---

## 3. 从 2D 卷积到 3D 卷积

### 3.1 2D 卷积：处理图像

2D 卷积对 `[C, H, W]` 的图像操作——kernel 在 H 和 W 两个方向滑动：

```python
# 输入: [B, C_in, H, W]
# kernel_size = (3, 3)
# 每个输出位置看一个 3×3 的空间邻域
conv2d = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)
```

### 3.2 3D 卷积：处理视频

视频多了一个**时间维度 T**，数据 shape 是 `[C, T, H, W]`。

3D 卷积的 kernel 在**三个方向**同时滑动——时间 T、高度 H、宽度 W：

```python
# 输入: [B, C_in, T, H, W]
# kernel_size = (3, 3, 3) → 同时看时间上的 3 帧、空间上的 3×3
conv3d = nn.Conv3d(C_in, C_out, kernel_size=3, padding=1)
```

**为什么视频需要 3D 卷积而不是逐帧 2D 卷积？**

逐帧 2D 卷积只能提取每一帧内的空间特征，**无法捕捉帧间的运动关系**。比如"一只手从左移到右"这种运动信息，必须同时看多帧才能理解。3D 卷积的时间维 kernel（如 kernel_t=3）让模型能同时看相邻帧，从而学到时间连续性。

### 3.3 数值例子

假设一个 5 帧 8×8 的灰度视频，用 `Conv3d(1, 16, kernel_size=3, padding=1)`：

```
输入: [1, 1, 5, 8, 8]   ← 1通道, 5帧, 8×8
输出: [1, 16, 5, 8, 8]  ← 16通道, 5帧, 8×8

每个输出位置 output[:, :, t, h, w] 的感受野是：
  输入的 [t-1:t+2, h-1:h+2, w-1:w+2]（3×3×3 的时空立方体）
```

---

## 4. 3D Causal Convolution：核心概念

### 4.1 把 Causal 性加到 3D 卷积的时间维

3D Causal Conv = **时间方向是 causal 的 + 空间方向是普通对称 padding 的**

```
时间方向: 只 pad 左边（过去），不 pad 右边（未来）→ Causal
空间方向: 左右上下对称 padding → 普通（空间没有因果性）
```

为什么空间方向不需要 causal？因为空间维度（H, W）没有"过去-未来"的概念——图像的上边和下边是平等的。只有时间维度有因果关系。

### 4.2 完整实现

```python
class CausalConv3d(nn.Module):
    """
    3D Causal Convolution:
    - 时间维度：只看过去和当前（左 pad, 不右 pad）
    - 空间维度：对称 padding（正常 2D 卷积行为）
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # 如果 kernel_size 是 int，展开为 (kt, kh, kw)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        kt, kh, kw = kernel_size
        
        # 时间维度的 causal padding 量 = kt - 1
        # （保证输出 t 只依赖输入的 t, t-1, ..., t-(kt-1)）
        self.temporal_pad = kt - 1
        
        # 空间维度的对称 padding
        self.spatial_pad_h = kh // 2
        self.spatial_pad_w = kw // 2
        
        # Conv3d 本身不做 padding，手动控制
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0  # 所有 padding 手动做
        )
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        
        # 手动 padding: F.pad 的参数顺序是 (W_left, W_right, H_left, H_right, T_left, T_right)
        x = F.pad(x, (
            self.spatial_pad_w, self.spatial_pad_w,   # W 方向对称
            self.spatial_pad_h, self.spatial_pad_h,   # H 方向对称
            self.temporal_pad, 0                       # T 方向只 pad 左边！
        ))
        
        return self.conv(x)
```

### 4.3 验证：首帧只依赖自己

```python
# 假设 kernel_size=(3, 3, 3), stride=1
# temporal_pad = 3 - 1 = 2
#
# 输入 x: [B, C, T, H, W]
# pad 后:  [B, C, T+2, H+2, W+2]  (时间左边+2, 空间上下左右各+1)
#
# 输出的 t=0 位置:
#   conv kernel 覆盖时间范围 [0, 1, 2]（pad 后的坐标）
#   对应原始输入的时间 [-2, -1, 0]
#   其中 -2 和 -1 是 pad 的零 → 不携带任何信息
#   只有原始的 t=0（第一帧）的像素参与计算
#
# ✓ 首帧的输出只依赖首帧本身！
```

---

## 5. 为什么视频 VAE 需要 3D Causal Conv？

### 5.1 应用场景：Image-to-Video 生成

在 Wan2.2 的 TI2V（Text+Image-to-Video）模式中：

1. 用户提供一张**首帧图像**和一段文本指令
2. 模型需要**先编码首帧**到 latent 空间
3. 然后生成后续帧的 latent
4. 最后所有 latent 一起解码回像素

关键约束：**编码首帧时，后续帧还没有**。如果 VAE Encoder 用普通 3D Conv，计算第一帧的 latent 就会需要第 2、3 帧的像素（因为 kernel 覆盖了未来帧）——这在 I2V 模式下是不可能的。

Causal Conv 解决了这个问题：**首帧的 latent 只依赖首帧像素**，可以独立编码。

### 5.2 对比：如果不用 Causal 会怎样？

```
普通 3D Conv (kernel_t=3, padding_t=1):
  latent[t=0] = f(pixel[t=-1], pixel[t=0], pixel[t=1])
                         ↑ pad的零          ↑ 需要第2帧！ ← 在I2V模式下不存在

Causal 3D Conv (kernel_t=3, padding_t=2 左):
  latent[t=0] = f(pixel[t=-2], pixel[t=-1], pixel[t=0])
                         ↑ pad的零      ↑ pad的零    ↑ 只用第1帧 ✓
```

### 5.3 第二个好处：支持流式编码

Causal 性还使得 VAE 可以**流式编码**任意长度的视频——每当新的一帧到来，只需要和之前的帧做卷积，不需要等后续帧。这对实时应用（如机器人控制中的在线视频编码）非常有用。

---

## 6. Stride 与下采样

3D Conv 还有一个重要用途：**通过 stride 实现下采样**。

```python
# 时间 4× 下采样 + 空间不变
CausalConv3d(C_in, C_out, kernel_size=3, stride=(4, 1, 1))

# 空间 2× 下采样 + 时间不变  
CausalConv3d(C_in, C_out, kernel_size=3, stride=(1, 2, 2))
```

在 Wan-VAE 中：
- 时间 4× 下采样：通过一个 stride=(4,1,1) 的 CausalConv3d 实现
- 空间 8× 下采样：通过三个 stride=(1,2,2) 的 CausalConv3d 逐步实现（2×2×2=8）

---

## 7. 和 Causal Attention 的区别

本文讲的 Causal Conv 和 [Causal Attention](/前置知识/001g_前置知识_Causal_Attention因果注意力掩码) 的关系：

| | Causal Convolution | Causal Attention |
|--|---|---|
| 作用层 | 卷积层（VAE 中使用） | 注意力层（Transformer 中使用） |
| 因果性实现 | 通过不对称 padding | 通过 attention mask（下三角矩阵） |
| 感受野 | 固定大小（kernel_size 决定） | 理论上无限（能看到所有过去位置） |
| 典型应用 | 视频 VAE 编解码器 | GPT、自回归生成模型 |
| 共同点 | **都保证当前位置的输出不依赖未来位置的输入** |

---

## 8. 本文小结

| 概念 | 一句话 |
|------|--------|
| 3D 卷积 | 在时间+空间三个方向同时滑动 kernel，捕捉视频的时空局部模式 |
| Causal | 时间维度只看过去和当前，不看未来 |
| 3D Causal Conv | 空间正常对称 pad + 时间只左 pad = 既能建模时空关系，又保持因果性 |
| 为什么需要 | Image-to-Video 模式要求首帧能独立编码 |
| 实现核心 | `F.pad(x, (..., temporal_pad, 0))` — 时间维度只 pad 左边 |

---

## 延伸阅读

- [Wan2.2 的 3D-VAE 与视频 Token 化](/系列/fastwam_human_pretrain/02_Wan2.2的3D-VAE与视频Token化)（本文知识的实际应用）
- [Causal Attention 因果注意力掩码](/前置知识/001g_前置知识_Causal_Attention因果注意力掩码)（注意力层面的因果性设计）
