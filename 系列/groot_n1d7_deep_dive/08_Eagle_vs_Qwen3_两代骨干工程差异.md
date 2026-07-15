---
title: "Eagle vs Qwen3：两代骨干的工程差异"
series:
  id: groot_n1d7_deep_dive
  chapter: 8
order: 8
---

# Eagle vs Qwen3：两代骨干的工程差异

> 逐行并排对比 `EagleBackbone` 和 `Qwen3Backbone` 的实现差异——从输入格式、token 结构到 attention mask 处理，理解升级时需要做的每一项适配。

## 相关阅读

- [Qwen3Backbone 实现详解](./07_Qwen3Backbone实现详解)（上一章）
- [Flow Matching 数学基础](./09_Flow_Matching数学基础)（下一章）
- [从 N1.5 到 N1.7 架构升级](./03_从N1d5到N1d7_架构升级)

---

## 前情提要

上一章我们深入走读了 Qwen3Backbone 的完整实现。本章我们把 EagleBackbone
（N1.5 使用）放在旁边做逐行对比，帮你理解两代骨干在**工程层面**的所有差异。
这对于理解代码演进、以及将来可能的骨干替换工作非常有价值。

---

## 1. 文件位置与导入对比

| | Eagle (N1.5) | Qwen3 (N1.7) |
|-|-------------|-------------|
| 文件 | `model/modules/eagle_backbone.py` | `model/modules/qwen3_backbone.py` |
| 模型类 | `AutoModel` (自定义) | `Qwen3VLForConditionalGeneration` |
| 依赖 | `AutoConfig`, `AutoModel` | `Qwen3VLForConditionalGeneration` |
| 版本要求 | 无特殊要求 | `transformers >= 4.57.0` |

```python
# Eagle
from transformers import AutoConfig, AutoModel

# Qwen3
try:
    from transformers import Qwen3VLForConditionalGeneration
    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False
```

Qwen3 有**显式的版本守卫**——如果 transformers 版本不支持 Qwen3-VL，
导入时就设标志位，初始化时报清晰错误。Eagle 没有这种保护。

---

## 2. 模型加载方式对比

### Eagle：本地配置 + `from_config`

```python
# Eagle 的加载方式
eagle_path = os.path.join(os.path.dirname(__file__), "nvidia", "Eagle-Block2A-2B-v2")
config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
self.model = AutoModel.from_config(config, trust_remote_code=True)
```

Eagle 使用 `from_config`（只加载架构，**不加载权重**），权重在外部单独加载。
这意味着：
- 仓库中必须包含 `nvidia/Eagle-Block2A-2B-v2/` 目录
- 目录中有 `config.json` 和自定义模型代码
- `trust_remote_code=True` 是**必须的**——没有它就无法加载自定义架构
- 模型权重需要从别处加载（通常在 Pipeline 层面通过 `from_pretrained` 统一处理）

### Qwen3：直接 `from_pretrained`

```python
# Qwen3 的加载方式
self.model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,  # "nvidia/Cosmos-Reason2-2B" 或本地路径
    **extra_kwargs,
    **transformers_loading_kwargs,
).eval()
```

Qwen3 直接用 `from_pretrained`（架构 + 权重一起加载）。
- 无需 `trust_remote_code`——Qwen3-VL 已经是 transformers 的原生模型
- 支持 HuggingFace Hub 自动下载和缓存
- `.eval()` 直接在加载后调用

### 对比总结

| 维度 | Eagle | Qwen3 |
|------|-------|-------|
| 加载方法 | `from_config` (只架构) | `from_pretrained` (架构+权重) |
| 自定义代码 | 需要 trust_remote_code | 不需要 |
| 本地文件依赖 | 必须有 nvidia/目录 | 不需要（可从 Hub 下载） |
| 安全性 | 低（执行不可信代码） | 高（原生 transformers 代码） |
| 可复现性 | 需要完整仓库 | 只需模型名字符串 |

---

## 3. 注意力实现对比

### Eagle：无回退

```python
if use_flash_attention:
    extra_kwargs["attn_implementation"] = "flash_attention_2"
# 紧接着有 assert：
assert use_flash_attention, "nvidia/Eagle-Block2A-2B-v2 requires flash attention by default"
assert load_bf16, "nvidia/Eagle-Block2A-2B-v2 requires bfloat16 by default"
```

Eagle **强制**要求 Flash Attention 2 + BF16。没有就直接报错。
这意味着：
- 没有 NVIDIA GPU（如 Apple Silicon）→ 无法运行
- GPU 不支持 Flash Attention（如旧款 V100）→ 无法运行
- 想用 FP32 调试 → 无法运行

### Qwen3：优雅降级

```python
if use_flash_attention:
    try:
        import flash_attn
        extra_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        logger.warning("flash_attn is not installed. Falling back to sdpa attention.")
        extra_kwargs["attn_implementation"] = "sdpa"
```

Qwen3 的降级链：`Flash Attention 2 → SDPA → Math (via _sdpa_context)`

此外，DiT 模块中还有针对特定硬件的兼容处理：
```python
# 在 dit.py 中
def _is_spark_sm121() -> bool:
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) == (12, 1)  # Blackwell 架构
```

这使得 N1.7 可以在几乎**所有**现代 NVIDIA GPU 上运行。

---

## 4. 层截断路径对比

```python
# Eagle：嵌套两层
while len(self.model.language_model.model.layers) > select_layer:
    self.model.language_model.model.layers.pop(-1)

# Qwen3：只嵌套一层
while len(self.model.language_model.layers) > select_layer:
    self.model.language_model.layers.pop(-1)
```

**路径差异**：`language_model.model.layers` vs `language_model.layers`

这反映了两种 VLM 架构内部结构的差异：
- Eagle (InternVL) 的 LLM 包裹在一个额外的 `model` 容器中
- Qwen3-VL 的 LLM 直接暴露 `layers`

如果你未来要适配新的 VLM，这是第一个要检查的事：
**LLM layers 的实际路径是什么？** 可以通过 `print(model)` 查看模型结构。

---

## 5. 冻结参数对比

### Eagle

```python
if not tune_llm:
    self.model.language_model.requires_grad_(False)
if not tune_visual:
    self.model.vision_model.requires_grad_(False)
    self.model.mlp1.requires_grad_(False)  # Eagle 特有：投影层也冻结
```

### Qwen3

```python
if not tune_llm:
    self.model.language_model.requires_grad_(False)
if not tune_visual:
    self.model.visual.requires_grad_(False)
```

**关键差异**：
1. 视觉模型路径：`vision_model` vs `visual`
2. Eagle 有额外的 `mlp1` 投影层需要单独冻结，Qwen3 没有

### set_frozen_modules_to_eval_mode 对比

```python
# Eagle
if self.model.language_model and not self.tune_llm:
    self.model.language_model.eval()
if self.model.vision_model and not self.tune_visual:
    self.model.vision_model.eval()
    self.model.mlp1.eval()  # 多一行

# Qwen3
if self.model.language_model and not self.tune_llm:
    self.model.language_model.eval()
if self.model.visual and not self.tune_visual:
    self.model.visual.eval()
```

Eagle 在冻结视觉时必须同时把 `mlp1` 设为 eval——这是一个容易遗漏的点。
Qwen3 没有这个问题，因为它没有显式的 MLP 投影层。

---

## 6. 前向传播对比

### 6.1 输入键的差异

```python
# Eagle
keys_to_use = ["input_ids", "attention_mask", "pixel_values"]

# Qwen3
keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
```

Qwen3 多了 `image_grid_thw`——这是动态分辨率机制的核心。
每张输入图像的 patch grid 结构通过这个张量传递给 ViT，
使得 ViT 能正确施加 3D 位置编码。

### 6.2 hidden_states 的访问方式

```python
# Eagle
outputs = self.model(**vl_input, output_hidden_states=True)
outputs = outputs["hidden_states"][-1]  # dict 索引

# Qwen3
outputs = self.model(**vl_input, output_hidden_states=True)
outputs = outputs.hidden_states[-1]  # 属性访问
```

这是 transformers 版本差异导致的：
- 旧版（Eagle 时代）：模型输出是 dict，用 `["key"]` 访问
- 新版（Qwen3 时代）：模型输出是 dataclass/namedtuple，用 `.attribute` 访问

### 6.3 image token ID 的获取

```python
# Eagle
image_mask = vl_input["input_ids"] == self.model.config.image_token_index

# Qwen3
image_mask = vl_input["input_ids"] == self.model.config.image_token_id
```

字段名不同：`image_token_index` vs `image_token_id`。
功能完全相同——都是用来标记 input_ids 中哪些位置是图像占位符。

### 6.4 输出完全相同

尽管内部实现差异很大，两个骨干的**输出接口完全一致**：

```python
# 两者都返回
BatchFeature(data={
    "backbone_features": outputs,           # [B, seq_len, hidden_dim]
    "backbone_attention_mask": attention_mask,  # [B, seq_len]
    "image_mask": image_mask,               # [B, seq_len]
})
```

这就是良好封装的价值——下游的 DiT 和 Action Head 完全不需要修改，
只要骨干返回这三个张量就行。

---

## 7. 参数量与计算量对比

| 组件 | Eagle-2B | Cosmos-Reason2-2B |
|------|----------|-------------------|
| 视觉编码器 | ~300M (InternViT) | ~300M (Qwen3-VL ViT) |
| MLP 投影层 | ~50M (mlp1) | 0 (无需投影) |
| LLM (完整) | ~1.6B (InternLM2) | ~1.5B (Qwen3) |
| LLM (截断到16层) | ~800M | ~750M |
| 总计 (截断后) | ~1.15B | ~1.05B |

N1.7 在截断后的总参数量略少（省掉了 mlp1），但视觉理解能力更强。
这是"架构进步"的体现——用更少的参数做更好的事。

---

## 8. Processor 对比

骨干网络的差异也体现在数据预处理的 Processor 上：

### Eagle Processor（推测）

```python
# 需要自定义 Processor
# 图像：固定 resize → normalize → to_tensor
# 文本：自定义 tokenizer（InternLM2 词表）
# 特殊 token：image_token_index
```

### Qwen3VLProcessor

```python
# 直接用 transformers 的原生 Processor
from transformers import Qwen3VLProcessor

processor = Qwen3VLProcessor.from_pretrained(model_name)
processor.tokenizer.padding_side = "left"  # Flash Attention 需要左 padding
```

**`padding_side = "left"` 的重要性**：

Flash Attention 的实现假设序列的有效部分是**连续的、右对齐的**。
如果 padding 在右边，短序列的有效 token 后面跟着一堆 padding，
Flash Attention 仍然能处理（通过 `cu_seqlens`）。
但在 GR00T 的 batch 处理中，左 padding 可以简化后续操作——
所有序列的最后一个 token 都在相同位置，方便取最终特征。

---

## 9. 完整差异总结表

| 维度 | Eagle (N1.5) | Qwen3 (N1.7) | 影响 |
|------|-------------|-------------|------|
| 加载方式 | from_config (只架构) | from_pretrained (架构+权重) | 部署更简单 |
| 自定义代码 | 必须 trust_remote_code | 不需要 | 更安全 |
| 注意力后端 | 仅 Flash-2 | Flash-2/SDPA/Math | 更多硬件兼容 |
| 精度要求 | 仅 BF16 | BF16/FP32/混合 | 调试更灵活 |
| 层路径 | language_model.model.layers | language_model.layers | 代码适配 |
| 视觉模型路径 | model.vision_model | model.visual | 冻结逻辑适配 |
| 投影层 | 有 (mlp1) | 无 | 少一个需要管理的模块 |
| 输入键 | 3个 | 4个 (+image_grid_thw) | 支持动态分辨率 |
| hidden_states | dict["hidden_states"] | .hidden_states | API 风格差异 |
| image token | config.image_token_index | config.image_token_id | 字段名差异 |
| 输出接口 | 完全相同 | 完全相同 | 下游无需修改 |

---

## 10. 如果你要换一个新的骨干网络

基于以上分析，如果将来要将 GR00T 的骨干换成另一个 VLM（如 InternVL3、LLaVA-Next 等），
你需要：

1. **新建一个 `XxxBackbone` 类**，实现相同的 4 个方法接口
2. **确定 LLM layers 的路径**：`print(model)` 找到 layers 列表
3. **确定 image token ID 的字段名**：检查 model.config
4. **确定视觉模型的路径**：用于冻结控制
5. **确认输出维度**：更新 `backbone_embedding_dim` 配置
6. **适配 Processor**：确保输出 input_ids, attention_mask, pixel_values 等
7. **在 `get_backbone_cls()` 中注册**：添加新的 model_name 匹配逻辑

```python
# gr00t_n1d7.py 中的骨干选择逻辑
def get_backbone_cls(config: Gr00tN1d7Config):
    if "Cosmos-Reason2" in config.model_name or "Qwen3-VL" in config.model_name:
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")
```

只要新骨干返回相同格式的 `BatchFeature`，整个动作头（DiT + 编解码器）
完全不需要修改。这就是模块化设计的力量。

---

## 下一章预告

从下一章开始，我们进入本系列的核心部分——动作生成。第 9 章将从数学基础讲起：
Flow Matching 是什么？为什么它能用 4 步就从噪声生成精确动作？
ODE、速度场、直线插值——这些概念如何在 GR00T 中落地？
