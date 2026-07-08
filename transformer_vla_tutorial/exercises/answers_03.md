# 03 章练习答案

## 1. Attention 和 MLP 分工

Attention 负责 token 之间的信息交换；MLP 负责每个 token 内部的非线性变换。

## 2. 位置编码的重要性

语言中顺序改变会改变语义；动作块中第 1 步、第 10 步、第 50 步也不是同一个时间点。没有位置编码，模型很难知道 token 的顺序或身份。

## 3. Encoder-only vs decoder-only

Encoder-only 通常允许所有 token 互相看，适合理解。Decoder-only 使用 causal mask，只能看过去，适合逐 token 生成。

## 4. VLA + diffusion 是否 Transformer-based

可以说部分是 Transformer-based。如果它用 Transformer/VLM 作为视觉语言条件编码器，那么核心表征来自 Transformer；但最终动作生成机制可能是 diffusion，而不是纯自回归 Transformer。
