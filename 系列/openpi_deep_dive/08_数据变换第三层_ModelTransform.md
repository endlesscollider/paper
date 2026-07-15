---
title: "数据变换第三层：ModelTransform 与 Tokenization"
series:
  id: openpi_deep_dive
  chapter: 8
order: 8
---

# 第八章：数据变换第三层 —— ModelTransform 与 Tokenization

> 本章目标：理解模型变换层的四个操作——图像缩放、语言分词、状态离散化、以及 FAST 动作 token 化——是如何将"人类可读的数据"转变为"模型可消费的 token"的。

**前情提要**：上一章完成了归一化，state 和 actions 已经在 $[-3, 3]$ 的标准范围内。现在需要把所有数据转为模型能直接处理的格式——token 序列。

**知识链接**：
- [第七章：归一化](./07_数据变换第二层_归一化)
- [第二章：π₀ 一句话做了什么？](./02_pi0一句话做了什么)

---

## 8.1 ModelTransform 的四个操作

ModelTransform 是进入模型前的最后一步，它完成四件事：

| 操作 | 类 | 作用 |
|------|-----|------|
| 注入默认 prompt | `InjectDefaultPrompt` | 如果没有 prompt，填入默认值 |
| 图像缩放 | `ResizeImages` | 所有图像统一到 224×224 |
| 文本分词 | `TokenizePrompt` | 语言指令 → token id 序列 |
| 维度填充 | `PadStatesAndActions` | state/actions 填充到 model_action_dim |

按顺序执行后，数据就可以直接构造为 `Observation` 对象送入模型。

---

## 8.2 InjectDefaultPrompt：缺省指令注入

有些推理场景中用户可能不传 prompt（例如某个微调任务只有一个固定指令）。`InjectDefaultPrompt` 会在 prompt 缺失时自动注入默认值：

```python
@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None
    
    def __call__(self, data):
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data
```

例如 `pi0_aloha_towel` 配置设置了 `default_prompt="fold the towel"`——每次推理都默认执行"叠毛巾"。

---

## 8.3 ResizeImages：统一图像尺寸

SigLIP 视觉编码器要求输入图像为 224×224 像素。但各机器人的相机分辨率各不相同（256×256、480×640、720×1280 等）。

```python
@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int  # 224
    width: int   # 224
    
    def __call__(self, data):
        data["image"] = {
            k: image_tools.resize_with_pad(v, self.height, self.width)
            for k, v in data["image"].items()
        }
        return data
```

`resize_with_pad` 的策略是：
1. 先按比例缩放到目标尺寸内（保持宽高比）
2. 如果有余边，用黑色（0）填充

例如 480×640 的图像：
- 先缩放到 168×224（宽度填满 224，高度按比例为 168）
- 上下各填充 28 行黑色像素 → 最终 224×224

---

## 8.4 TokenizePrompt：语言指令分词

这是 ModelTransform 中最复杂的一步。它把自然语言字符串转为模型能理解的 token id 序列。

### 8.4.1 PaligemmaTokenizer（π₀ 和 π₀.₅ 共用）

OpenPI 使用 Google 的 PaliGemma 分词器（基于 SentencePiece）：

```python
class PaligemmaTokenizer:
    def __init__(self, max_len=48):
        self._max_len = max_len
        self._tokenizer = sentencepiece.SentencePieceProcessor(...)
    
    def tokenize(self, prompt, state=None):
        if state is not None:
            # π₀.₅ 格式：状态也编码为文本
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {prompt}, State: {state_str};\nAction: "
        else:
            # π₀ 格式：只编码文本
            full_prompt = prompt + "\n"
        
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        # 填充或截断到 max_len
        ...
        return tokens, mask
```

### 8.4.2 π₀ 与 π₀.₅ 的 prompt 格式差异

**π₀ 的 prompt 格式**（`discrete_state_input=False`）：

```
<BOS>pick up the fork\n
```

简单直接——文本指令加上换行符作为"回答开始"的标记。

**π₀.₅ 的 prompt 格式**（`discrete_state_input=True`）：

```
<BOS>Task: pick up the fork, State: 145 89 201 128 110 156 90 130;\nAction: 
```

状态被量化为 256 个离散 bin 的索引（0-255），然后作为文本编码进 prompt。这样做的好处是：

1. 状态信息通过文本 token 进入主 Gemma 2B——主模型可以"看到"状态
2. 不再需要 `state_proj` 把状态注入动作专家——减少了设计复杂度
3. 离散化后的状态是 token 级别的，与文本 token 同等处理

**量化细节**：`np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1`
- 将归一化后的 state（约 $[-1, 1]$ 范围）均匀切分为 256 个 bin
- 输出 0-255 的整数索引
- 转为字符串后由 SentencePiece 编码为 token

### 8.4.3 输出格式

```python
tokens = np.array([2, 1413, 502, 89, ...])  # token id 序列
mask = np.array([True, True, True, ..., False, False])  # 哪些位置是真实 token
```

- `tokens` 形状：`(max_token_len,)`，不足的位置用 0 填充
- `mask` 形状：`(max_token_len,)`，填充位置为 False

---

## 8.5 FASTTokenizer：动作 Token 化（π₀-FAST 专用）

π₀-FAST 模型不使用 Flow Matching，而是用自回归方式生成动作。为此需要将连续动作序列离散化为 token。

### 8.5.1 FAST 分词的核心思想

FAST（Fourier Action Sequence Tokenization）的工作方式：

1. 对动作序列做 DCT（离散余弦变换）→ 频域表示
2. 保留低频分量（丢弃高频细节）→ 压缩
3. 对压缩后的系数做 FSQ（有限标量量化）→ 离散 token

这样一个 (50, 7) 的连续动作序列就变成了约 50-100 个离散 token。

### 8.5.2 FASTTokenizer 的 prompt 格式

```
<BOS>Task: pick up the fork, State: 145 89 201 128;\nAction: [FAST_TOKEN_1][FAST_TOKEN_2]...|<EOS>
```

- 前缀：任务描述 + 离散化状态
- 后缀：`Action: ` + FAST token 序列 + `|` 终止符 + `<EOS>`

### 8.5.3 注意力掩码设计

FAST 的 tokenize 同时返回 `ar_mask` 和 `loss_mask`：

```python
ar_mask = [0, 0, 0, ..., 0,  # 前缀：双向注意力（mask=0 → 可互相看）
           1, 1, 1, ..., 1]  # 后缀：因果注意力（mask=1 → 只看前面）

loss_mask = [False, False, ...,  # 前缀：不计算损失
             True, True, ...]    # 后缀：计算 next-token 损失
```

- `ar_mask=0`：该 token 与前面的 token 共享注意力（prefix-LM 行为）
- `ar_mask=1`：该 token 只能看到之前的所有 token（causal 行为）
- `loss_mask=True`：该位置参与损失计算

---

## 8.6 PadStatesAndActions：维度填充

最后一步，确保 state 和 actions 的最后一维对齐到模型的 `action_dim`：

```python
@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    model_action_dim: int  # 如 24
    
    def __call__(self, data):
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data
```

`pad_to_dim` 在末尾补零：

- DROID state (8,) + 16 个零 → (24,)
- LIBERO actions (50, 7) + 17 列零 → (50, 24)

**为什么需要填充到统一维度？** 因为模型的 `action_in_proj` 和 `action_out_proj` 是固定维度的线性层。所有输入都必须是相同的 `action_dim`。多出的零维度在推理输出时由 `XxxOutputs` 截断丢弃。

---

## 8.7 完整流程示例

以 DROID + π₀.₅ 为例，从 DataTransform 的输出到进入模型：

```python
# DataTransform 输出（归一化后）
data = {
    "state": np.array([0.1, -0.8, 0.3, 0.5, -0.1, 0.2, 0.0, 0.4]),  # (8,)
    "image": {"base_0_rgb": array(480,640,3), "left_wrist_0_rgb": array(480,640,3), ...},
    "actions": array(50, 8),  # 训练时才有
    "prompt": "pick up the fork",
}

# 1. InjectDefaultPrompt → 无变化（已有 prompt）
# 2. ResizeImages → 图像变为 (224, 224, 3)
# 3. TokenizePrompt(discrete_state_input=True)
#    → 构造 "Task: pick up the fork, State: 145 89 201 ..."
#    → SentencePiece 编码为 token ids
#    → data["tokenized_prompt"] = array(48,)
#    → data["tokenized_prompt_mask"] = array(48,)
# 4. PadStatesAndActions(24)
#    → data["state"] = array(24,)  # 后16维为0
#    → data["actions"] = array(50, 24,)  # 后16列为0
```

---

## 8.8 本章小结

| 操作 | 输入 | 输出 | 关键参数 |
|------|------|------|----------|
| InjectDefaultPrompt | 无 prompt 的字典 | 有默认 prompt | 配置中的 default_prompt |
| ResizeImages | 任意尺寸图像 | 224×224 | resize_with_pad |
| TokenizePrompt (π₀) | "pick up fork" | token ids (48,) | max_token_len=48 |
| TokenizePrompt (π₀.₅) | "pick up fork" + state | token ids (48,) | discrete_state_input=True |
| FASTTokenizer (π₀-FAST) | prompt + state + actions | tokens + ar_mask + loss_mask | max_len=256 |
| PadStatesAndActions | state (8,), actions (50,8) | state (24,), actions (50,24) | model_action_dim=24 |

---

## 下一章预告

数据变换管线到此完成。下一章我们正式进入模型内部——首先是 SigLIP 视觉编码器。我们会理解 Vision Transformer 如何将一张 224×224 的图像切割为 16×16 个 patch，经过 27 层 Transformer 后输出 256 个语义丰富的视觉 token。
