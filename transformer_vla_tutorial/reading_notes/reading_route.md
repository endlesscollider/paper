# 推荐论文阅读路线

## 第 1 组：Transformer 基础

1. Attention Is All You Need  
   重点：Q/K/V、multi-head、positional encoding、encoder-decoder。

2. Vision Transformer  
   重点：图像 patch 如何变成 token。

## 第 2 组：机器人模仿学习与动作块

1. ACT / ALOHA: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware  
   重点：action chunk、temporal ensembling、CVAE、低成本双臂数据。

2. Diffusion Policy  
   重点：为什么连续多模态动作适合 diffusion。

## 第 3 组：VLA

1. RT-1  
   重点：大规模机器人数据 + Transformer policy + 离散动作。

2. RT-2  
   重点：VLM 知识如何迁移到动作。

3. OpenVLA  
   重点：开源 VLA 的模型接口和动作表示。

4. Octo  
   重点：generalist robot policy、数据混合、任务 conditioning。

## 第 4 组：更新的 action expert 路线

1. π0 / openpi  
   重点：VLM + flow matching action expert。

2. GR00T  
   重点：humanoid/generalist robot foundation model 的系统栈。
