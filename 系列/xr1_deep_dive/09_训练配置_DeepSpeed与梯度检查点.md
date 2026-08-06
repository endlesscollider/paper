---
title: "训练配置：DeepSpeed + FusedAdam + 梯度检查点实战"
series:
  id: xr1_deep_dive
  chapter: 10
order: 10
---

# 训练配置：DeepSpeed + FusedAdam + 梯度检查点实战

> **前情提要**：上一章详细拆解了 60 维动作空间的数据格式和归一化。本章进入训练的工程细节——分布式策略、优化器配置、显存优化。

**知识链接**：
- 前代对照：[XR-0 训练配置](/系列/xr0_deep_dive/11_训练配置_DeepSpeed与优化器)

---

## 1. 分布式训练策略：DeepSpeed ZeRO

XR-1 使用 DeepSpeed 做分布式训练：

```yaml
trainer:
  strategy:
    type: deepspeed
    params:
      allgather_bucket_size: 5e8   # 500MB
      reduce_bucket_size: 5e8      # 500MB
```

从配置看使用的是 **ZeRO Stage 2**（优化器状态 + 梯度分片，不分片参数）。

### 1.1 为什么用 ZeRO-2 而不是 ZeRO-3

| ZeRO Stage | 分片内容 | 优缺点 |
|------------|---------|--------|
| Stage 1 | 优化器状态 | 节省最少，但通信最少 |
| **Stage 2** | 优化器 + 梯度 | **好平衡：显存节省 ~3x，通信开销可控** |
| Stage 3 | 优化器 + 梯度 + 参数 | 节省最多，但每次前向需要 allgather 参数 |

XR-1 模型约 5B 参数。用 bf16 精度：
- 参数占用：5B × 2 bytes = 10 GB
- 优化器状态（Adam fp32）：5B × 8 bytes = 40 GB
- 梯度：5B × 2 bytes = 10 GB

ZeRO-2 把优化器和梯度分片到多 GPU：
- 8 GPU 时每张卡只存 1/8 的优化器状态 = 5 GB
- 参数仍完整存在每张卡上 = 10 GB
- 总每卡占用 ≈ 10 + 5 + 1.25 ≈ 16 GB（不含激活值）

ZeRO-3 能进一步把参数也分片，但每次 forward 需要通信聚合参数，对 DiT 这种频繁 forward 的模块（training_repeat=4）来说通信开销太大。

## 2. 优化器：FusedAdam

```yaml
optimizer:
  type: "deepspeed.ops.adam.FusedAdam"
  params:
    lr: 1.0           # 注意：实际 lr 由 scheduler 控制
    betas: [0.9, 0.95]
    weight_decay: 0.1
    eps: 1.0e-08
```

**FusedAdam** 是 DeepSpeed 提供的 CUDA kernel 融合版 Adam——把 Adam 的多步运算（exp_avg 更新、exp_avg_sq 更新、参数更新）合并成一个 kernel，减少 GPU 内核启动开销。

关键参数选择：
- `betas=[0.9, 0.95]`：β₂=0.95 比默认的 0.999 更小，让二阶矩估计更快适应新梯度。这是大模型训练的常见做法（GPT-3、LLaMA 等都用类似设置）
- `weight_decay=0.1`：较强的权重衰减，防止过拟合（数据量大时可以承受）
- `lr=1.0`：配合 scheduler 使用，scheduler 直接输出目标学习率

## 3. 学习率调度：Cosine Warmup

```yaml
scheduler:
  type: mibot.utils.cosine_warmup.get_cosine_schedule_with_warmup
  params:
    num_training_steps: 10000
    num_warmup_steps: 500
    warmup_lr_start: 5e-7
    max_lr: 0.00002
    min_lr: 0.000005
```

学习率变化曲线：

```
lr
2e-5 ─────────╮
              │ ╲
              │   ╲
              │     ╲ cosine decay
5e-7 ─╱      │       ╲
     ╱        │         ╲_____ 5e-6
    ╱ warmup  │
───┼──────────┼──────────────── steps
   0   500         10000
```

- 前 500 步：从 5e-7 线性升到 2e-5
- 500~10000 步：从 2e-5 余弦衰减到 5e-6

max_lr = 2e-5 在 5B 模型上是保守的选择——VLM 骨干已经预训练好，太高的学习率会破坏已有知识。

## 4. 梯度检查点（Gradient Checkpointing）

XR-1 对 VLM 骨干做了两层梯度检查点：

### 4.1 ViT 视觉编码器

```python
self.vlm.model.visual.gradient_checkpointing_enable()
```

视觉编码器处理高分辨率图像，激活值占用巨大。梯度检查点让前向时不缓存中间激活，反向时重新计算——用时间换显存。

### 4.2 VLM 语言模型的 MLP 层

```python
if self.ffn_gradient_checkpointing:
    for layer in self.vlm.model.language_model.layers:
        mlp = layer.mlp
        original_forward = mlp.forward

        def checkpointed_forward(x, forward=original_forward, module=mlp):
            if module.training and torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(forward, x, use_reentrant=False)
            return forward(x)

        mlp.forward = checkpointed_forward
```

这段代码只对 MLP 做检查点（不对 Attention 做）。原因：
- MLP 的激活值占用大（intermediate_size = 4×hidden_size = 10240）
- Attention 的 Flash Attention 本身已经做了显存优化
- 只 checkpoint MLP 是性价比最好的选择

### 4.3 显存节省估算

36 层 VLM，每层 MLP 中间激活 ≈ batch_size × seq_len × 10240 × 2 bytes：
- batch=2, seq_len=800: 每层 ≈ 33 MB → 36 层 ≈ 1.2 GB
- 用检查点后这部分激活降为 0（反向时重算）

ViT 检查点节省更多（图像 token 数通常 > 1000）。

## 5. VLM Embedding 层冻结

```python
self.vlm.model.get_input_embeddings().requires_grad_(False)
```

Word embedding 层被冻结（不训练）。原因：
- Embedding 层参数量大（vocab_size × hidden_size ≈ 150K × 2560 ≈ 384M）
- 但后训练中不需要修改 token embedding（词汇表不变）
- 冻结后节省 384M 参数的梯度和优化器状态

## 6. 精度：bf16-mixed

```yaml
precision: "bf16-mixed"
```

- **模型权重**：bf16 存储和计算
- **优化器状态**：fp32（Adam 的一阶/二阶矩需要高精度）
- **梯度**：bf16 计算后在通信前转 fp32（DeepSpeed 处理）
- **Loss 计算**：`compute_flow_loss` 中显式 `.float()` 转 fp32

```python
def compute_flow_loss(self, pred, target, action_mask, weight):
    pred, target, weight = pred.float(), target.float(), weight.float()
    ...
```

这是因为 MSE loss 对精度敏感——bf16 的精度在小数值差异时可能导致数值不稳定。

## 7. 训练启动命令解析

```bash
RESOURCE_GPU=1 bash scripts/train.sh \
  trainer.project="xiaomi-robotics-1" \
  trainer.exp_name="posttrain" \
  trainer.default_root_dir="outputs" \
  data=load_washer \
  model=posttrain \
  model.params.pretrained="pretrained_ckpt/model_states.pt"
```

| 参数 | 含义 |
|------|------|
| `RESOURCE_GPU=1` | 使用 1 张 GPU |
| `trainer.project` | WandB 项目名 |
| `trainer.exp_name` | 实验名，也是 checkpoint 子目录名 |
| `data=load_washer` | 加载 `configs/data/load_washer.yaml` |
| `model=posttrain` | 加载 `configs/model/posttrain.yaml` |
| `model.params.pretrained` | 预训练权重路径 |

多 GPU 训练只需改 `RESOURCE_GPU=8`。多节点需要设置 `WORLD_SIZE`、`RANK`、`MASTER_ADDR`、`MASTER_PORT`。

## 8. 其他训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| seed | 42 | 随机种子 |
| max_steps | 10000 | 总训练步数 |
| save_interval | 10000 | 每 N 步保存 checkpoint |
| accumulate_grad_batches | 1 | 不做梯度累积 |
| gradient_clip_val | 1.0 | 梯度裁剪阈值 |
| gradient_clip_algorithm | "norm" | 按梯度范数裁剪 |

梯度裁剪 = 1.0 是标准设置——防止偶尔的大梯度破坏训练稳定性。

## 9. Checkpoint 结构

训练输出的 checkpoint 目录结构：

```
outputs/project_xiaomi-robotics-1/posttrain/
├── config.py              # 解析后的完整配置
├── last.ckpt/
│   └── checkpoint/
│       └── mp_rank_00_model_states.pt   # DeepSpeed 格式权重
└── wandb/                 # WandB 日志
```

部署时 `deploy.py` 从 checkpoint 目录加载 config 和权重。

## 10. 本章小结

XR-1 训练配置的关键选择：

| 维度 | 选择 | 原因 |
|------|------|------|
| 分布式 | DeepSpeed ZeRO-2 | 平衡显存和通信 |
| 优化器 | FusedAdam | CUDA 融合加速 |
| lr | 2e-5 (max) | 保守，保护 VLM 预训练知识 |
| 调度 | Cosine + 500步warmup | 平滑过渡 |
| 精度 | bf16-mixed | 训练效率 + 数值稳定的折中 |
| 梯度检查点 | ViT + VLM MLP | 主要显存瓶颈 |
| Embedding | 冻结 | 不需要修改 |
| 梯度裁剪 | norm=1.0 | 训练稳定性 |

---

**下一章预告**：[Ch10 推理部署](./10_推理部署_异步执行与Server架构) 将介绍部署流程——Server/Client 架构、异步执行流水线、以及真机集成的具体接口。
