---
title: "Eagle vs Qwen3：两代骨干的工程差异"
series:
  id: groot_n1d7_deep_dive
  chapter: 8
order: 8
---

# Eagle vs Qwen3：两代骨干的工程差异

> 上一章已经逐行走读过 `Qwen3Backbone` 的完整实现。本章不重复那些代码，而是站在"如果我要把骨干换成另一个 VLM，需要改哪些地方"的视角，把 Eagle 和 Qwen3 的工程差异按维度梳理清楚。

## 相关阅读

- [Qwen3Backbone 实现详解](./07_Qwen3Backbone实现详解)（上一章，Qwen3 侧的完整代码在这里）
- [Flow Matching 数学基础](./09_Flow_Matching数学基础)（下一章）
- [从 N1.5 到 N1.7 架构升级](./03_从N1d5到N1d7_架构升级)（升级动机总览）

---

## 前情提要

上一章我们完整走读了 `Qwen3Backbone`——模型加载、层截断、冻结控制、前向传播的每一行代码。
本章的目标不一样：不是再讲一遍 Qwen3 怎么实现，而是回答一个更实际的问题——
**如果未来要把 GR00T 的骨干换成别的 VLM（比如 InternVL3、LLaVA-Next），需要检查哪些具体的工程细节？**

`EagleBackbone`（N1.5 用）和 `Qwen3Backbone`（N1.7 用）刚好是一次真实发生过的骨干替换，
把它们放在一起对比，就是最好的"替换清单"参考。全文按七个工程维度展开，每个维度先说
"这里为什么会有差异、差异会带来什么影响"，再给出对照。

---

## 1. 加载方式：要不要信任外部代码？

Eagle 用的是 `AutoModel.from_config()`——只加载模型架构，不加载权重，而且必须传
`trust_remote_code=True` 才能执行仓库自带的自定义模型代码。这意味着部署 Eagle 时，
仓库里必须带着 `nvidia/Eagle-Block2A-2B-v2/` 这个本地目录，权重还需要在 Pipeline 层面
单独加载进去。

Qwen3-VL 已经是 transformers 的原生模型，`Qwen3Backbone` 直接用标准的 `from_pretrained()`
把架构和权重一次性加载完，不需要执行任何非官方代码，也不需要仓库里带着额外的模型定义文件——
可以直接从 HuggingFace Hub 下载，或者指向一个本地路径。

| 维度 | Eagle (N1.5) | Qwen3 (N1.7) |
|------|-------------|-------------|
| 加载方法 | `AutoModel.from_config()`（只架构） | `Qwen3VLForConditionalGeneration.from_pretrained()`（架构+权重） |
| 自定义代码 | 需要 `trust_remote_code=True` | 不需要，原生 transformers 模型 |
| 本地文件依赖 | 必须带 `nvidia/Eagle-Block2A-2B-v2/` 目录 | 不需要，可从 Hub 下载 |
| 版本守卫 | 无 | 有——`transformers < 4.57.0` 时导入即报错，提示升级 |

**换骨干时要检查什么**：新骨干是否已经是 transformers 官方支持的模型类。如果不是，
就要像 Eagle 一样接受 `trust_remote_code` 的安全代价，并把自定义代码文件一起放进仓库。

---

## 2. 注意力后端：能不能优雅降级？

Eagle 对硬件的要求是**硬性**的——初始化时直接 `assert use_flash_attention` 和
`assert load_bf16`，缺一个都无法运行。这在 Apple Silicon 或不支持 Flash Attention 的
旧款 GPU（如 V100）上会直接报错，调试时也没法退回 FP32。

Qwen3 走的是逐级降级的路线：优先尝试 `flash_attn`，如果导入失败就自动回退到 PyTorch
原生的 SDPA，并打印警告而不是报错。此外 `dit.py` 里还专门处理了 NVIDIA Blackwell（SM 12.1）
架构下 memory-efficient SDPA 的已知问题，强制切到纯数学实现：

```python
def _is_spark_sm121() -> bool:
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) == (12, 1)
```

| 维度 | Eagle (N1.5) | Qwen3 (N1.7) |
|------|-------------|-------------|
| Attention 后端 | 仅 Flash Attention 2，无回退 | Flash-2 → SDPA → Math，逐级降级 |
| 精度要求 | 仅 BF16 | BF16 / FP32 均可 |
| 硬件兼容性 | 依赖 NVIDIA GPU + Flash Attention | 几乎所有现代 NVIDIA GPU |

**换骨干时要检查什么**：新骨干对 Flash Attention 的依赖是不是硬性的。如果是，最好按
Qwen3 的模式包一层 try/except 降级逻辑，而不是让用户在不支持的硬件上直接崩溃。

---

## 3. 内部结构路径：LLM 的层列表藏在哪一层？

这是换骨干时**最容易踩坑**的一类差异——两个 VLM 的内部模块命名和嵌套方式完全不同，
同一个操作（截断层数、冻结视觉编码器）在代码里要走不同的属性路径。

Eagle 基于 InternVL 架构，它的语言模型多包了一层 `model` 容器，视觉部分叫
`vision_model`，并且有一个独立的 `mlp1` 投影层把视觉特征维度对齐到 LLM 维度。
Qwen3-VL 架构更"扁平"，语言模型的层列表直接挂在 `language_model.layers` 下，
视觉部分叫 `visual`，而且视觉到语言的维度对齐是在架构内部完成的，没有独立的投影层需要管理。

| 维度 | Eagle (N1.5) | Qwen3 (N1.7) |
|------|-------------|-------------|
| LLM 层列表路径 | `language_model.model.layers` | `language_model.layers` |
| 视觉模块路径 | `vision_model` | `visual` |
| 视觉→语言投影层 | 有，独立的 `mlp1` | 无，架构内部处理 |
| image token 字段名 | `config.image_token_index` | `config.image_token_id` |
| hidden_states 访问方式 | `outputs["hidden_states"]`（dict） | `outputs.hidden_states`（属性，新版 transformers 风格） |

有了这张表，"截断层数"和"冻结视觉编码器"这两个操作的代码差异其实只是路径字符串不同，
逻辑完全一样（详见第 7 章讲过的完整实现）；唯一需要额外注意的一行是 Eagle 冻结视觉时
必须**多冻结一次** `mlp1`：

```python
if not tune_visual:
    self.model.vision_model.requires_grad_(False)
    self.model.mlp1.requires_grad_(False)  # Eagle 特有，Qwen3 没有这一行
```

如果照抄 Eagle 的冻结逻辑去接一个新骨干，却忘了这个新骨干可能有自己的投影层，
就会出现"看起来冻住了，其实投影层还在偷偷更新梯度"的隐蔽 bug。

**换骨干时要检查什么**：打印 `print(model)` 找到 LLM 层列表和视觉模块的真实路径，
并确认新骨干有没有类似 `mlp1` 这种需要单独管理的中间模块。

---

## 4. 前向传播的输入输出：接口一致才能不改下游代码

Eagle 的前向传播只需要 3 个输入键（`input_ids`、`attention_mask`、`pixel_values`），
因为它只支持固定分辨率输入。Qwen3-VL 支持动态分辨率，多了一个 `image_grid_thw`——
记录每张图像被切成多少个 patch（时间×高×宽三个维度），ViT 靠这个信息施加正确的位置编码。

尽管两者内部实现差异很大，`forward()` 的**输出**接口被设计成完全一致——都返回同样
三个键的 `BatchFeature`：

```python
BatchFeature(data={
    "backbone_features": outputs,               # [B, seq_len, hidden_dim]
    "backbone_attention_mask": attention_mask,   # [B, seq_len]
    "image_mask": image_mask,                    # [B, seq_len]
})
```

这一点是整个骨干替换能够"无痛"的关键——下游的 DiT 和动作头完全不关心骨干内部是
Eagle 还是 Qwen3，只要这三个张量的形状和语义对得上就行。

| 维度 | Eagle (N1.5) | Qwen3 (N1.7) |
|------|-------------|-------------|
| 输入键数量 | 3 个 | 4 个（多了 `image_grid_thw`） |
| 分辨率支持 | 固定分辨率 | 动态分辨率 |
| 输出接口 | `BatchFeature{backbone_features, backbone_attention_mask, image_mask}` | 完全相同 |

**换骨干时要检查什么**：新骨干需要哪些额外的输入键（比如动态分辨率、多帧视频等特殊机制），
但无论输入多复杂，输出必须严格对齐成这三个键，下游模块才不需要任何改动。

---

## 5. 数据预处理：Processor 与 padding 方向

骨干换了，配套的 Processor 也要换。Eagle 时代用的是自定义 Processor（固定 resize +
normalize，配自定义的 InternLM2 词表）。Qwen3 时代直接用 transformers 原生的
`Qwen3VLProcessor`，能处理动态分辨率图像。

这里有一个容易被忽略但会直接影响正确性的设置——`padding_side`：

```python
processor.tokenizer.padding_side = "left"
```

Flash Attention 对 padding 的位置有隐含假设，同时 GR00T 在提取骨干最终特征时希望
"每个样本序列的最后一个有效 token 都在相同的位置"——如果 padding 在右边，不同样本的
最后一个有效 token 会落在不同位置，需要额外用 attention_mask 去定位；改成左 padding
后，所有样本的最后一个位置天然就是有效 token，取特征的代码可以写得更简单。

**换骨干时要检查什么**：新骨干配套的 Processor 是否支持左 padding，以及它对 attention
实现的 padding 方向有没有隐含假设。

---

## 6. 参数量与延迟：升级是否划算

| 组件 | Eagle-2B | Cosmos-Reason2-2B |
|------|----------|-------------------|
| 视觉编码器 | ~300M（InternViT） | ~300M（Qwen3-VL ViT） |
| 视觉→语言投影层 | ~50M（`mlp1`） | 0（无需投影） |
| LLM 完整参数量 | ~1.6B（InternLM2） | ~1.5B（Qwen3） |
| 截断到 16 层后 | ~800M | ~750M |
| 截断后总计 | ~1.15B | ~1.05B |

N1.7 截断后的总参数量比 N1.5 略少（省掉了 `mlp1` 这部分），但视觉理解能力更强——
这正是"架构进步"的直接体现：用更少的参数做更好的事，而不是单纯堆参数量。

---

## 7. 如果你要换一个新的骨干网络

把前面六节的检查项汇总成一份清单——将来接入新的 VLM（InternVL3、LLaVA-Next 等）时，
按这个顺序确认：

1. 新骨干是不是 transformers 原生支持？如果不是，要接受 `trust_remote_code` 的代价
2. 新骨干对 Flash Attention 的依赖是不是硬性的？需不需要包一层降级逻辑
3. `print(model)` 找到 LLM 层列表和视觉模块的真实路径，以及有没有需要单独管理的投影层
4. image token 的字段名是什么？（`image_token_index` 还是 `image_token_id`，或者别的名字）
5. 新骨干需要哪些特殊的输入键（动态分辨率、多帧视频等）
6. 无论内部多复杂，`forward()` 的输出必须严格对齐成 `backbone_features` /
   `backbone_attention_mask` / `image_mask` 这三个键
7. 配套的 Processor 是否支持左 padding
8. 在 `get_backbone_cls()` 中注册新的 `model_name` 匹配规则：

```python
def get_backbone_cls(config: Gr00tN1d7Config):
    if "Cosmos-Reason2" in config.model_name or "Qwen3-VL" in config.model_name:
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")
```

只要新骨干返回相同格式的 `BatchFeature`，整个动作头（DiT + 编解码器）完全不需要修改——
这就是第 4 节讲的"接口一致性"设计带来的实际收益：模块化封装把一次骨干替换的影响范围
限制在了骨干类内部。

---

## 下一章预告

从下一章开始，我们进入本系列的核心部分——动作生成。第 9 章将从数学基础讲起：
Flow Matching 是什么？为什么它能用 4 步就从噪声生成精确动作？
ODE、速度场、直线插值——这些概念如何在 GR00T 中落地？
