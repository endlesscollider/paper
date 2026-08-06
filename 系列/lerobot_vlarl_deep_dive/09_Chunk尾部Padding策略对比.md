---
title: "09 Chunk 尾部 Padding：零填充 vs 重复填充 vs 无需填充"
series:
  id: lerobot_vlarl_deep_dive
  chapter: 9
order: 9
---

# Chunk 尾部 Padding 策略：RLinf 零填充 vs LeRobot 重复填充

## 相关阅读

- [AWR-Flow Handoff 与 ConRFT 的 Actor 更新机制对比](./AWR_Flow_Handoff与ConRFT的Actor更新机制对比)
- [Chunk 长度与时间尺度对比](./Chunk长度与时间尺度_Handoff40步vsConRFT16步vsSACQC5步)
- [LeRobot-VLARL 全面拆解](/系列/lerobot_vlarl_deep_dive/)

---

## 一、问题场景

Episode 快结束了，剩余 3 个真实 action step，但 chunk_size = 40（或 16 或 5）。问题来了：**chunk 中"超出 episode 尾部"的那些位置该放什么？**

这不是一个无关紧要的工程细节——它直接影响 Critic 学到什么、Actor loss 的梯度方向、以及物理语义的正确性。

```
Episode: [a₁, a₂, a₃, DONE]
Chunk:   [a₁, a₂, a₃, ???, ???, ..., ???]  ← 后面放什么？
                        ↑ 这里的选择很重要
```

---

## 二、RLinf 的做法：零填充 + 严格 Valid Mask

### 2.1 数据构造

对应 `rlinf/data/chunk_sac_bc.py:244-284`：

```python
# 初始化：全部为 0
chunk_actions = torch.zeros(num_chunks, chunk_length, 62, dtype=torch.float32)
chunk_rewards = torch.zeros(num_chunks, chunk_length, dtype=torch.float32)
valid = torch.zeros(num_chunks, chunk_length, dtype=torch.bool)

for chunk_index, start in enumerate(starts):
    length = min(chunk_length, episode_length - start)
    
    # 只填充有效部分
    chunk_actions[chunk_index, :length] = torch.from_numpy(actions[start:start+length])
    chunk_rewards[chunk_index, :length] = rewards[start:start+length]
    valid[chunk_index, :length] = True  # 只有前 length 步标记为有效
```

结果：

```
action:  [真实a₁, 真实a₂, 真实a₃, 0, 0, ..., 0]
reward:  [r₁,     r₂,     r₃,     0, 0, ..., 0]
valid:   [True,   True,   True, False, False, ..., False]
```

### 2.2 Loss 中的使用

Actor BC loss 严格使用 valid mask：

```python
@staticmethod
def _action_bc_loss(policy_actions, data):
    valid = data["chunk_sac_valid"].to(policy_actions.dtype)  # (B, T)
    squared_error = (policy_actions - data["chunk_sac_action"]).square()  # (B, T, D)
    
    # 只对有效位置计算 MSE
    per_sample = (squared_error * valid.unsqueeze(-1)).sum(dim=(1, 2)) / (
        valid.sum(dim=1).clamp_min(1.0) * squared_error.shape[-1]
    )
    return per_sample
```

Critic 的 TD return 计算中同样使用 valid mask——无效步的 reward 不参与折扣累加。

### 2.3 语义分析

- **填充值 = 0**：对大部分机器人动作来说，0 代表"不动"（关节增量为零）
- **valid mask 严格屏蔽**：确保 0 填充值**永远不会**影响任何 loss 的梯度
- **episode 尾部清晰**：模型不会误以为"最后 37 步机器人静止不动"

---

## 三、LeRobot (dev_wq_temp_temp) 的做法：重复最后动作

### 3.1 数据构造

对应 `actor_online_training.py:291-338`：

```python
for offset in range(self.chunk_size):
    src = idx + offset
    if src < len(transitions):
        # 有效范围内：用真实动作
        actions.append(_action_as_2d(transitions[src]["action"]))
        rewards_chunk.append(_reward_as_2d(transitions[src]["reward"]))
        action_is_pad.append(False)
    else:
        # 超出范围：重复最后一个真实动作
        actions.append(_action_as_2d(transitions[-1]["action"]))
        rewards_chunk.append(torch.zeros((1, 1), dtype=torch.float32))
        action_is_pad.append(True)

item["complementary_info"]["action_is_pad"] = torch.tensor(action_is_pad, dtype=torch.bool)
```

结果：

```
action:       [真实a₁, 真实a₂, 真实a₃, a₃重复, a₃重复, ..., a₃重复]
reward:       [r₁,     r₂,     r₃,     0,      0,      ..., 0]
action_is_pad:[False,  False,  False,  True,   True,   ..., True]
```

### 3.2 Mask 使用情况：⚠️ 关键问题

ConRFT 的 `preprocess_batch()` 中，将 `action_is_pad` 合并到 `action_mask` 的代码**目前是注释状态**：

```python
# conrft_policy_groot.py:579-590 (注释掉的代码)
# action_mask = curr_proc["action_mask"]
# action_is_pad = batch.get("complementary_info", {}).get("action_is_pad", None)
# if action_is_pad is not None:
#     action_is_pad = action_is_pad.to(device=action_mask.device, dtype=torch.bool)
#     ...
#     action_mask = action_mask & (~action_is_pad).unsqueeze(-1)
```

这意味着：**`action_is_pad` 信息被保存了，但可能没有真正进入所有 loss 计算。**

---

## 四、两种策略的后果对比

### 4.1 如果 mask 正确使用

| 维度 | RLinf 零填充 + valid mask | LeRobot 重复填充 + is_pad mask |
|------|--------------------------|-------------------------------|
| **对 loss 的影响** | 完全无影响（被 mask 屏蔽） | 完全无影响（被 mask 屏蔽） |
| **显存占用** | 相同 | 相同 |
| **计算浪费** | 填充位置仍经过网络（但梯度被 mask 归零） | 同左 |
| **语义正确性** | ✅ | ✅ |

如果 mask 正确使用，两种方案**数学上等价**——填充值是什么都不影响训练结果。

### 4.2 如果 mask 未正确使用（LeRobot 的风险）

由于 ConRFT 中 `action_is_pad` 合并逻辑被注释掉，以下 loss 计算可能受到影响：

#### 4.2.1 Critic Loss

如果 Critic 接收的 action chunk 包含"重复的最后动作"但没有 mask：

```
Critic 看到: Q(s, [a₁, a₂, a₃, a₃, a₃, ..., a₃])
实际含义:     "机器人做了 a₁→a₂→a₃ 然后停在 a₃ 的位置 13 步"
```

Critic 会学到一个**错误的 Q 估计**——它以为这个"静止等待"的序列是有意义的，但实际上这些步骤从未发生过。

#### 4.2.2 BC Loss

如果 Actor 被训练去模仿包含重复 padding 的 chunk：

```
Actor 目标: 学会输出 [a₁, a₂, a₃, a₃, a₃, ..., a₃]
             ← 前 3 步是有意义的 →  ← 后 13 步是人为重复 →
```

Actor 可能学到一种"尾部保持"的假行为——在 chunk 后半段持续输出最后一个动作。

#### 4.2.3 具体的物理语义偏差

| 场景 | 重复填充的问题 |
|------|-------------|
| 夹爪正在关闭 | 最后动作 = "关闭中"→ 填充 13 步 = "持续施压" → Critic 认为这是好动作 |
| 手臂正在加速 | 最后动作 = "大增量"→ 填充 13 步 = "持续加速" → 物理上会撞墙 |
| 精确对齐中 | 最后动作 = "微调"→ 填充 13 步 = "反复微调" → 可能振荡 |

**零填充**（RLinf）没有这个问题——0 增量对应"静止"，这至少是一个物理上合理的默认行为。

---

## 五、为什么 LeRobot 选择"重复最后动作"？

可能的设计动机：

1. **Consistency Policy 的训练需要连续性**：如果填充为 0，chunk 内会出现"突然跳到零"的断裂，可能影响 CP 的训练稳定性
2. **对 GR00T VLA 的兼容性**：VLA 的 action head 被训练为输出"连续变化"的轨迹，突然跳零可能让 loss landscape 更复杂
3. **"最后动作重复"是 action chunking 文献中的常见做法**：ACT 原论文也用 last-action repeat

但这些都**前提是 mask 能正确进入 loss**。

---

## 六、修复方案

### 6.1 取消注释 action_is_pad 合并逻辑

最直接的修复是把 `conrft_policy_groot.py:579-590` 的注释取消：

```python
action_mask = curr_proc["action_mask"]
action_is_pad = batch.get("complementary_info", {}).get("action_is_pad", None)
if action_is_pad is not None:
    action_is_pad = action_is_pad.to(device=action_mask.device, dtype=torch.bool)
    if action_is_pad.dim() == 1:
        action_is_pad = action_is_pad.unsqueeze(0)
    if action_is_pad.dim() == 3 and action_is_pad.shape[-1] == 1:
        action_is_pad = action_is_pad.squeeze(-1)
    # 合并：真实数据 padding 和 episode 尾部 padding 都被 mask
    action_mask = action_mask & (~action_is_pad).unsqueeze(-1)
```

### 6.2 确保所有 loss 路径都使用合并后的 mask

需要检查以下位置是否都使用了正确的 `action_mask`：
- BC loss（Actor 监督学习部分）
- CQL loss（Critic 的 OOD 惩罚项）
- TD target 计算（chunk 内的 reward 累加）
- Advantage 计算（如果使用梯度精修）

### 6.3 RLinf 方案（更保守）

如果不想改代码，也可以改数据构造：把"重复最后动作"改为"填零"：

```python
else:
    actions.append(torch.zeros_like(_action_as_2d(transitions[-1]["action"])))  # 改为零
    action_is_pad.append(True)
```

---

## 七、检测方法：如何确认 Padding 是否进入了 Loss

### 7.1 梯度检查

在训练循环中加一个断言：

```python
# 如果 padding 位置对 loss 有非零梯度贡献，说明 mask 不工作
action_chunk.retain_grad()
loss.backward()

is_pad = batch["complementary_info"]["action_is_pad"]
pad_grad = action_chunk.grad[is_pad]
assert pad_grad.abs().max() < 1e-7, "Padding is leaking into loss!"
```

### 7.2 数值实验

构造一个对照实验：
1. 把所有 padding 值改为极端大值（如 100.0）
2. 如果训练行为完全不变 → mask 工作正常
3. 如果 loss 爆炸或行为改变 → mask 有泄漏

---

## 八、ReinFlow / ScoRe-Flow：无 Padding 问题的设计

[ReinFlow](/前置知识/001u_前置知识_ReinFlow_Flow策略的噪声注入RL微调) 和 [ScoRe-Flow](/论文综述/080_ScoRe_Flow_Score引导的Flow策略RL微调) **完全不存在 chunk 尾部 padding 问题**。原因很简单：它们不使用 action chunk。

它们的 MDP 是标准的 step-level：每步独立决策，每步独立 reward，每步独立 log-prob。Episode 结束就是 `done=True`，不需要对"未来还没发生的步骤"做任何填充。

这是 step-level MDP 的一个**隐含优势**——在 chunk-level RL 中，padding 是一个必须处理但容易出 bug 的工程细节（如本文第三/四节分析的那样）。选择 step-level MDP 从根本上避免了这类问题。

代价是：ReinFlow/ScoRe-Flow 每步都要跑完整的 Flow SDE（4 步 denoising），推理频率和延迟都是 chunk 方案的瓶颈。

---

## 九、总结

| 维度 | RLinf 零填充 | LeRobot 重复填充 | ReinFlow / ScoRe-Flow |
|------|------------|-----------------|----------------------|
| **填充值** | 0（代表"静止"） | 最后一个真实动作 | **N/A（无 chunk，无 padding）** |
| **Mask 机制** | `valid` 严格进入所有 loss | `action_is_pad` 合并逻辑**被注释** | **N/A** |
| **如果 mask 失效** | 影响小（0 合理） | **影响大**（重复动作有错误语义） | **不可能发生** |
| **风险等级** | 低 | ⚠️ 中-高 | **零风险** |

**核心 takeaway**：

1. Padding 策略本身不是问题——**mask 是否真正生效才是问题**
2. RLinf 的设计更安全：即使 mask 出 bug，零值不会引入严重的物理偏差
3. LeRobot dev_wq_temp_temp 分支存在一个**已知的隐患**：`action_is_pad` 合并逻辑被注释，可能导致 padding 动作泄漏到 loss 中
4. 对于接触操作、夹爪控制等"最后一步动作有特殊语义"的场景，这个泄漏的影响尤其严重
