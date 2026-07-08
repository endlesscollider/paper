# 如何学习这套教程

## 三个原则

1. **先画图，再看公式。** 只要你不能画出 token 如何流动，就说明还没理解模型。
2. **先问输入输出，再问网络细节。** 每篇机器人论文都先回答：输入是什么？输出是什么？损失是什么？数据是什么？
3. **先手算小例子，再跑大模型。** Attention 的核心可以用 3 个 token 手算出来。

## 每章学习模板

读每章时都回答：

- 这一章解决的痛点是什么？
- 它把什么东西表示成 token？
- 哪些信息彼此 attention？
- 输出是类别、文本、动作，还是一段动作？
- 如果换成机器人任务，会卡在哪里？

## 阶段验收

### 阶段 1：Attention
你能不用公式解释：
> 每个 token 会根据自己当前的 Query，和所有 token 的 Key 做匹配，得到权重，再把所有 Value 按权重加权求和。

### 阶段 2：Transformer
你能画出：
> embedding → self-attention → residual/norm → MLP → residual/norm → output head。

### 阶段 3：ACT
你能解释：
> ACT 不是每一帧只预测一个动作，而是预测未来 k 步动作块，并用 temporal ensembling 平滑执行。

### 阶段 4：VLA
你能解释：
> VLA 把视觉、语言、状态变成条件信息，再输出机器人动作；Transformer 常作为跨模态信息融合器或 VLM backbone。
