---
title: GR00T 与 π 系列对比 ACT
order: 2
tags: [机器人, VLA, ACT]
category: 工程实践
star: 3
---

# GR00T 和 π 相比 ACT 的区别

本文面向已经理解 ACT 的读者，目标是把 NVIDIA GR00T 和 Physical Intelligence 的 π 系列从“听起来像机器人大模型”拆成可以工程判断的概念。阅读完以后，你应该能回答三个问题：

1. GR00T / π 到底比 ACT 多了什么。
2. 这些“大模型”的能力来自模型结构、数据规模、语言语义，还是来自工程数据闭环。
3. 在 MiGenRL / embodied-arena 这类训练闭环里，什么时候应该继续用 ACT，什么时候应该考虑 VLA foundation model。

本文信息截至 2026-06-23。GR00T 和 π 都在快速迭代，尤其 GR00T N1.7 仍是 Early Access，π0.7 也主要以论文和公开视频形式呈现。因此本文更强调稳定的技术脉络，而不是把某个版本的 benchmark 当成永久结论。

## 一句话结论

如果你熟悉 ACT，可以先这样建立直觉：

ACT 是一个很强的“任务级模仿学习 policy”。它通常在一个机器人、一个或一组接近的任务、固定观测和动作接口上训练，核心技巧是 action chunking：每次预测未来一段动作，降低有效决策 horizon，让模仿学习更稳定。

GR00T 和 π 是“VLA robot foundation model”。VLA 指 Vision-Language-Action，也就是把图像、语言和机器人状态映射到动作。它们不是只想解决一个具体任务，而是希望通过大量跨任务、跨场景、跨机器人数据预训练，获得类似大语言模型的通用先验，然后再通过少量 post-training / fine-tuning 适配到某个机器人和任务。

更具体地说：

| 维度 | ACT | GR00T | π 系列 |
| --- | --- | --- | --- |
| 基本定位 | 小到中等规模的专用模仿学习策略 | 面向 humanoid / generalist robot skills 的开放 VLA foundation model | 面向通用物理智能的 VLA foundation model 系列 |
| 典型输入 | 多相机 RGB、qpos，有时无语言 | RGB、语言、机器人状态、embodiment 配置 | RGB、语言、机器人状态，后续版本加入更丰富上下文 |
| 典型输出 | 固定维度 action chunk | 连续动作，N1/N1.5/N1.7 使用 diffusion / DiT 风格动作头 | π0 使用 flow matching action expert；π0-FAST 用离散动作 token 自回归；π0.7 强调上下文 steerable |
| 数据规模 | 十几到几百条 demo 也能起步 | 大规模机器人数据、合成数据、人类视频、post-training 数据 | 多机器人、多任务、开放数据、自有机器人数据、web / multimodal / semantic 数据 |
| 泛化方式 | 主要靠任务内数据覆盖和视觉闭环 | 预训练先验 + 语言 grounding + post-training | VLM 先验 + flow/action expert + heterogeneous co-training + context conditioning |
| 工程开放度 | 代码简单，容易完全自控 | 官方 GitHub / Hugging Face 权重，Apache-2.0，N1.7 EA | π0 / openpi 开源；π0.5、π0.7 不等价于完整可下载产品 |
| 最适合 | 明确任务、明确机器人、数据量可控、需要快速闭环 | humanoid / 双臂 / 多任务，需要借助开放基础模型和 NVIDIA Isaac 生态 | 研究 VLA 泛化、复杂家居长程任务、跨机器人技能迁移 |

一句更工程化的话：ACT 解决的是“我有这个机器人和这个任务，怎么学得稳”；GR00T / π 解决的是“我希望同一个基础模型已经懂很多物体、语言、场景和动作模式，然后我只做适配”。

## 先复盘 ACT：它强在哪里，也边界在哪里

ACT 的全称是 Action Chunking with Transformers，来自 ALOHA / low-cost bimanual manipulation 工作。原论文的核心场景不是“机器人通用智能”，而是一个更具体的问题：低成本、存在误差的双臂硬件，能不能通过端到端模仿学习完成细粒度操作任务，例如开杯盖、插电池、穿扎带这类需要接触、视觉闭环和双手配合的任务。

ACT 的关键不是“用了 Transformer”这么简单，而是它把行为克隆里最难的两个问题做了一个很实用的折中。

第一个问题是 compounding error。标准 behavior cloning 每一步预测一个动作，执行一点点，下一步再根据新的 observation 预测。只要某一步略微偏离专家轨迹，后续 observation 就开始偏离训练分布，模型会越走越偏。对于接触丰富、精细操作任务，这个问题特别明显，因为毫米级误差就可能导致物体滑走、夹爪碰错边、插入失败。

第二个问题是人类示教的非平稳性。人示教时并不是一个严格的反馈控制器；同一个任务中，人有时快、有时慢，有时会犹豫，有时会微调。单步 BC 会把这些局部动作看成独立标签，容易学出抖动或者卡顿。

ACT 的处理方式是预测 action chunk，也就是一次预测未来若干步动作。训练时，模型看到当前多相机图像和关节状态，同时学习一个未来动作序列的生成模型。原始 ACT 用 CVAE 风格结构：训练时用 encoder 把未来动作序列压成 latent style，decoder 再根据 observation 和 latent 预测 action chunk；测试时丢掉 encoder，把 latent 设成 prior mean，直接生成动作序列。实践里还会用 temporal aggregation，把不同时间步预测出的重叠动作 chunk 做平滑聚合。

这个设计带来几个好处：

1. 有效 horizon 变短。模型不是每 20ms 重新决定下一小步，而是学一个短时技能片段，例如“伸过去、闭合夹爪、略微上提”。这更接近人类示教中的运动基元。
2. 输出更平滑。chunk 内动作天然连续，temporal aggregation 又能减少不同预测之间的跳变。
3. 数据效率高。ALOHA 论文里一些真实任务只用约 10 分钟示教就能达到可观成功率。对工程迭代来说，这很重要。
4. 系统简单。ACT 不要求互联网规模预训练，也不要求复杂语言理解。只要观测、动作、归一化和控制器契约稳定，就能快速闭环。

但 ACT 的边界也很清楚。

首先，它通常是 task-specialist，不是 foundation model。你可以训练一个 ACT 做 stack can、open drawer、fold cloth，也可以用多任务数据训练一个共享 ACT，但它并不会天然理解“把桌面收拾干净”这种开放指令，也不会因为在网页上见过水杯、餐盘、毛巾就知道这些物体该怎么处理。ACT 的语义来自示教数据本身。

其次，它对 embodiment 和 action representation 很敏感。一个 ACT checkpoint 的 action dim、qpos dim、相机名、控制频率、坐标系、夹爪定义基本都被训练数据固定住了。如果换机器人、换末端、换控制器或换动作空间，通常需要重新组织数据甚至重新训练。MiGenRL 里全 0 成功率常见原因，正是 qpos/action 表示、归一化统计、rollout action packing 不一致。

第三，它不是为开放语言泛化设计的。可以给 ACT 加 language embedding，把 language-conditioned ACT 做成多任务 policy，但这和 VLA foundation model 不是同一个量级。ACT 本身没有预训练 VLM 的世界知识，也没有大规模语言 grounding。它可以学“红色方块拿到蓝色盘子”，但这需要你的数据覆盖这些语言、物体和组合。

第四，它的数据分布通常窄。ACT 最擅长在一个清晰定义的任务族里把动作学稳；但如果真实部署里任务边界变成“厨房里任意要收拾的东西”，示教空间会迅速爆炸。你可以继续扩大 ACT 数据集，但没有强大的语义先验时，模型更像是在记很多局部模式，而不是组合已有知识解决新问题。

所以 ACT 的定位可以概括为：一个非常实用、可控、数据效率高的专用模仿学习 policy。它不是落后技术，在很多明确任务上仍然是最应该先尝试的基线。但它不是“大模型路线”的终点。

## 为什么机器人开始谈 VLA foundation model

机器人学习里长期有一个矛盾：真实世界任务很丰富，但机器人数据很贵。语言模型之所以能泛化，是因为互联网上有大量文本；视觉语言模型之所以能识别很多物体和语义，是因为图文数据很大；但机器人动作数据必须通过真实机器人、仿真、遥操作或视频标注获得，成本高得多。

如果每个任务都从零收集数据并训练一个 ACT / Diffusion Policy，那工程上会遇到几个问题：

1. 新任务冷启动慢。换一个物体、场景、机器人，可能又要收集几十到几百条 demo。
2. 长程任务难。把“拿起杯子”学好不等于能“整理厨房”。后者需要识别哪些东西该去哪、按什么顺序做、失败后怎么恢复。
3. 语言接口弱。机器人要进入真实应用，不可能只接收 one-hot task id。用户会说“把脏衣服放到篮子里”“把台面清一下”“把那个透明杯子拿过来”。
4. 跨机器人迁移差。单臂、双臂、移动操作、humanoid 的状态和动作不同，但很多物理知识是共享的，例如抓取、放置、推、拉、打开、擦拭。
5. 语义知识没有被充分利用。VLM 已经知道很多物体、场景和语言关系，但传统小 policy 没有直接继承这些知识。

VLA foundation model 的基本想法是：不要只训练一个小 policy 从 observation 到 action，而是把机器人控制接到一个已经有视觉语言能力的大模型上。模型输入不只是 qpos 和 image，还包括自然语言指令、任务上下文、甚至示例图像、子目标、历史记忆。模型输出也不是文本，而是连续或离散的机器人动作。

这里有一个关键挑战：语言 token 是离散的，机器人动作通常是高频连续控制。VLA 不能简单照搬 LLM 的 next-token prediction。它必须解决“如何让一个大视觉语言模型生成高频、精确、平滑、可执行的动作”。不同路线的差异，主要就在 action head 上：

1. ACT：Transformer/CVAE 直接预测 action chunk。
2. Diffusion Policy：用扩散模型生成动作序列。
3. GR00T：VLM 负责语义和 grounding，Diffusion Transformer 负责连续动作 denoising。
4. π0：VLM backbone 加 action expert，用 flow matching 生成连续 action chunk。
5. π0-FAST：把动作压成离散 token，让 VLA 可以像语言一样自回归预测动作 token。
6. π0.7：在 π0 路线基础上强调 diverse context conditioning，让模型不仅听“做什么”，还听“按什么策略做”。

从 ACT 到 VLA，真正变化的是问题定义。ACT 问的是：“当前 observation 下，专家接下来会怎么动？” VLA 问的是：“在这个视觉场景、语言目标、机器人身体和历史上下文下，什么行为能实现用户意图？” 这就是两者的根本差别。

## GR00T：NVIDIA 的开放 humanoid VLA 栈

GR00T 写作 GR00T，中间是两个零。它是 NVIDIA Isaac 生态里的 generalist robot foundation model 方向。公开资料里，GR00T N1 在 2025 年 3 月发布，N1.5 在 2025 年 6 月发布，N1.7 在官方 GitHub 上作为 Early Access 提供。NVIDIA 对它的定位是面向 humanoid robot reasoning and skills 的开放基础模型，支持通过 post-training 适配具体 embodiment、任务和环境。

从工程角度看，GR00T 不只是一个 checkpoint。它更像一个组合包：

1. VLA 模型权重和推理 / fine-tuning 代码。
2. LeRobot 风格的数据格式适配。
3. Hugging Face 上的模型和数据入口。
4. Isaac Sim / Isaac Lab / synthetic data blueprint 生态。
5. 面向 humanoid 和双臂操作的动作表示与部署接口。

这点很重要。GR00T 的价值不只在“模型参数比 ACT 大”，还在 NVIDIA 试图把仿真、合成数据、真实机器人、基础模型、部署加速放进同一个开发链路里。

### GR00T 的模型结构

GR00T N1 的官方介绍把模型分成两个系统：

1. System 2：Vision-Language Model。它负责看图、理解语言、做语义 grounding 和较慢的推理规划。
2. System 1：Diffusion Transformer。它负责把 System 2 的语义表示变成连续机器人动作。

这个命名借用了人类认知里的快慢系统，但从工程上可以简单理解为：VLM 负责“懂”，DiT action head 负责“动”。

ACT 里没有这种强 VLM backbone。ACT 的视觉 encoder 能从相机图像里提取任务相关特征，但它通常不是经过互联网图文预训练的语义模型。ACT 知道“这个像素模式下专家会夹这里”，但不一定知道“这是红色苹果，用户说 apple 时应该指它”。GR00T 的 VLM 则希望继承图文预训练带来的物体、关系、指令理解能力。

GR00T N1.5 继续沿用 VLM + DiT 的基本结构，但公开研究页提到几个变化：VLM 在预训练和 fine-tuning 中冻结，连接 vision encoder 和 LLM 的 adapter MLP 简化，并对视觉和文本 token embedding 加 layer normalization；NVIDIA 称这些改动改善了 language following 和 generalization。N1.5 还加入 FLARE 这类 future latent representation alignment 目标，让模型不只是模仿动作，也学习未来表示对齐，从而能更好利用人类视频。

GR00T N1.7 的官方 GitHub 说明则强调它换用了新的 VLM backbone，例如 Cosmos-Reason2-2B / Qwen3-VL，并采用 relative end-effector action space。相对末端动作空间的意义很大：如果动作是“从当前末端位姿往哪里增量移动”，而不是“去世界坐标某个绝对 pose”，那么跨机器人、跨场景和人类视频迁移会更自然。N1.7 还提到利用 20K 小时 EgoScale human video data 做预训练，强化从人类操作视频中学习 manipulation prior 的能力。

### GR00T 的数据策略

ACT 的数据通常是直接的 robot demonstrations。你采一批 HDF5，每条 episode 有图像、qpos、action、success，训练即可。

GR00T 的数据策略更像金字塔：

1. 底层是大量通用视频和人类行为数据，提供物体、场景、动作先验。
2. 中间是跨机器人和跨任务的 robot data，包括开源数据、内部数据、不同 humanoid / 双臂平台数据。
3. 上层是合成数据和具体任务 post-training 数据，用来适配目标机器人和目标任务。

NVIDIA 在 N1 技术博客中强调了 synthetic data blueprint 的作用：通过 Omniverse / Isaac 相关工具生成大量合成轨迹，以弥补真实机器人示教不足。N1.5 研究页进一步提到 DreamGen / neural trajectories、OpenXE、AgiBot-Beta、模拟 GR-1 数据等多源 mixture。

这和 ACT 最大差异在于：ACT 的泛化主要来自你手头那批示教数据，而 GR00T 希望在 fine-tune 之前就已经从大规模 heterogeneous data 学到“怎么抓、怎么放、怎么理解语言、怎么处理常见物体和场景”。post-training 不是从零学习，而是把基础能力对齐到你的 embodiment。

### GR00T 的开放性和工程接入

截至 2026-06-23，GR00T 的官方 GitHub 页面把 N1.7 标为 Early Access，并提供预训练权重、reference code、fine-tuning 和 inference 流程。它声明 N1.7 可在 Apache-2.0 下商业许可使用，但 EA 阶段支持和稳定性保证有限，完整 benchmark 和生产支持要等 GA。

对工程团队来说，GR00T 的吸引力有三点：

1. 它比很多公司内部 VLA 更可获得。权重、代码、数据格式和示例流程公开。
2. 它和 Isaac 生态天然贴近。对 embodied-arena 这种基于 Isaac Lab 的平台，仿真数据、synthetic trajectories、Sim2Real workflow 的概念更容易对接。
3. 它偏 humanoid / 双臂 / generalist manipulation。若目标机器人形态接近 humanoid 或双臂移动操作，GR00T 的预训练分布可能比普通单臂 policy 更相关。

但也要看到限制：

1. GR00T 不是拿来即用的万能机器人脑。它仍然需要 embodiment-specific modality config、动作空间定义、后训练数据和控制器适配。
2. 如果你的机器人不是 humanoid / 双臂，而是固定单臂、夹爪定义特殊、任务又很窄，GR00T 的复杂度未必划算。
3. 大模型推理和 fine-tuning 成本更高。NVIDIA 给 N1 的 post-training 最低配置曾提到 A6000 / 4090 级别，实际 N1.7 还要看具体配置和 batch。
4. 语言能力强不等于控制精度一定强。精密接触任务仍然可能需要小数据 fine-tune、RL refine、力控或低层控制器配合。

所以，GR00T 可以理解为“一个开放、Isaac 生态友好、面向 humanoid 的 VLA foundation model 栈”。它最像 ACT 的地方，是最终仍然要输出 action chunk / continuous action 并接控制器；最不像 ACT 的地方，是它的语义表示、预训练数据和跨任务目标完全不同。

## π 系列：Physical Intelligence 的通用物理智能路线

Physical Intelligence 通常简称 π。它的公开模型系列包括 π0、π0-FAST、π0.5、π0.7。和 GR00T 一样，π 也是 VLA 路线；但从公开叙事看，π 更强调 general-purpose physical intelligence，也就是让同一个模型在真实世界、多机器人、长程家居任务、开放语言指令下表现出组合泛化。

如果把 GR00T 看成 NVIDIA Isaac 生态下的开放 humanoid foundation model，那么 π 更像一个研究型、产品化潜力很强但开放程度分版本不同的通用机器人 foundation model 家族。π0 / openpi 已经开源，π0.5 和 π0.7 的完整能力则主要通过论文、博客和视频展示。

### π0：VLM + flow matching action expert

π0 是 Physical Intelligence 的第一代 generalist policy。官方博客把它描述为一个能接收图像、文本并输出低层动作的通用机器人基础模型。π0 论文标题是 “A Vision-Language-Action Flow Model for General Robot Control”，这已经点出核心：它不是普通 Transformer BC，而是 VLM + flow matching 的 VLA。

π0 的结构可以拆成两部分：

1. VLM backbone。它继承互联网规模图文预训练带来的视觉语义和语言理解能力。论文里提到使用 PaliGemma 初始化。
2. Action expert。它处理机器人状态和动作，通过 flow matching 生成高频连续 action chunk。

Flow matching 可以粗略理解为 diffusion 的近亲。扩散模型常见做法是从噪声逐步 denoise 成数据；flow matching 学一个从噪声分布流向数据分布的速度场。对机器人动作来说，它的好处是可以建模连续、多模态的动作序列，而不是只输出一个均值动作。多模态很重要：同一个场景下，专家可能从左边绕、也可能从右边绕；直接 MSE 学均值会得到一条谁都不像的中间轨迹。

π0 论文特别强调高频 action chunk，最高到 50Hz。这里和 ACT 有明显继承关系：π0 也承认 action chunk 对机器人控制很重要；区别是 ACT 用较小的 Transformer/CVAE 从任务数据里学 chunk，而 π0 把 chunk 生成接到 VLM foundation model 上，用 flow matching 处理连续动作分布。

π0 的预训练数据也不再是单一机器人。论文公开信息提到它在 7 种机器人配置、68 个任务上预训练，包括单臂、双臂、移动操作等平台，并能在 zero-shot、fine-tuning、以及和高层 VLM policy 组合时完成任务，例如洗衣、桌面清理、装盒等。

### openpi：π0 的开源入口

Physical Intelligence 在 2025 年 2 月发布 Open Sourcing π0，开放 π0 权重和代码，并发布 π0-FAST。openpi 的意义是让研究者可以真正 fine-tune 和复现实验，而不只是看公开视频。

公开说明里，π0 base 是标准预训练模型，训练于 OXE 和 Physical Intelligence 的 7 个机器人平台，主要设计目标是 fine-tuning；对于预训练中已有的任务，也可以 zero-shot 尝试。这个定位很实际：即使叫 foundation model，工程上仍然要靠 fine-tuning 才能在目标任务稳定落地。

### π0-FAST：把动作 token 化

π0-FAST 是另一个重要分支。FAST 是 Efficient Robot Action Tokenization。它试图解决 VLA 的一个效率问题：连续动作生成如果走 diffusion / flow matching，训练和推理都可能比自回归 token 预测更重；而语言模型生态已经非常擅长 next-token prediction。

FAST 的思路是把机器人动作序列压缩成离散 action token，让模型像生成文本 token 一样生成动作 token。这样可以利用成熟的 autoregressive VLM 训练方式，也能显著提升训练速度。Physical Intelligence 博客称 FAST 让 generalist policy 训练比之前快约 5 倍。

但动作 token 化也有代价。连续控制天然要求精度和平滑性；离散 token 的码本设计、压缩误差、解码平滑、长序列稳定性都会影响最终控制。π0-FAST 的价值不在于“离散动作一定优于 flow matching”，而在于它探索了 VLA scaling 的另一条路线：把机器人动作纳入 token 统一建模框架。

### π0.5：heterogeneous co-training 和 open-world generalization

π0.5 是 π0 的下一步，重点不只是模型结构，而是 co-training recipe。官方博客标题是 “A VLA with Open-World Generalization”。它要解决的问题是：一个 VLA 能不能到训练数据里没有的新家里，完成整理厨房、整理卧室、铺床、把东西放到抽屉这类长程任务？

π0.5 的核心原则是 heterogeneous co-training。也就是把不同类型的数据放在一起训练，包括：

1. 低层 robot demonstrations，有 image、language、state、action。
2. 高层语义数据，例如某个 observation 对应的子任务标签。
3. object detection、captioning、VQA 等 multimodal web / vision-language 数据。
4. 不同机器人形态、不同场景、不同任务粒度的数据。

这和 ACT 的训练数据观念差别很大。ACT 的训练样本通常必须有动作标签；没有 action 的图文数据对 ACT 基本没用。π0.5 则利用 VLM 架构的灵活性，让模型在有动作的数据上学控制，在没有动作但有语义的数据上学理解，在高层标签数据上学任务分解。这种混合训练让模型不仅知道“手怎么动”，也知道“厨房清理时哪些物体应该去哪里”。

π0.5 公开实验强调 entirely new homes。也就是说，测试厨房和卧室不在训练数据中。它能接受不同粒度的语言命令：高层命令如“把盘子放进水槽”，也可以是更具体的“拿起那个圆刷子”。这类能力不是靠某一个动作头突然变强，而是靠多源数据让模型同时学到语义、任务结构和低层控制。

从 ACT 视角看，π0.5 最大的新东西是“任务边界变宽”。ACT 很适合“把这个 can stack 到那个 can 上”；π0.5 想做的是“房间里这些东西需要被整理”，其中包含识别、选择、排序、导航、移动操作和恢复。

### π0.7：steerable context conditioning 和 emergent capabilities

π0.7 是 2026 年 4 月公开的新模型。官方博客称它在泛化上有 step-change，可以在未见环境中执行多种 dexterous tasks，能跟随新的语言命令，甚至完成训练数据中没有直接出现过的任务组合。论文标题强调 “Steerable Generalist Robotic Foundation Model with Emergent Capabilities”。

π0.7 的关键词是 steerable，不只是 instruction-following。它的核心想法是 diverse context conditioning：prompt 不只包含“做什么”的语言命令，还可以包含更丰富的多模态上下文，用来告诉模型“按什么策略做”。公开摘要里提到的上下文包括任务表现 metadata、subgoal images 等。

为什么这重要？因为大规模 heterogeneous robot data 里，同一句话可能对应很多策略。例如“清理台面”可以先拿杯子，也可以先擦污渍；“打开机器”取决于机器类型；“折衣服”对不同机器人、不同衣物、不同桌面都可能有不同策略。单靠语言指令可能不够区分这些策略。π0.7 通过更丰富上下文，把模型从“听一句话做动作”提升到“根据多模态提示选择一种策略”。

公开展示中，π0.7 的代表能力包括：

1. 未见厨房和卧室中的多阶段指令执行。
2. 对复杂语言引用的泛化。
3. 跨 embodiment zero-shot，例如让新机器人完成没有该机器人特定任务数据的 laundry folding。
4. 在某些复杂任务上达到接近专门 RL fine-tuned 模型的表现。

需要谨慎的是，π0.7 并不等于我们现在可以下载一个 checkpoint 直接接到任意机器人上。它更像是在说明 Physical Intelligence 的 scaling recipe：当数据、上下文、模型和训练方式足够丰富时，机器人 VLA 可能出现组合泛化能力。对工程团队的启发是：未来 robot foundation model 的输入接口可能不只是 image + instruction + qpos，而会包含历史记忆、子目标图、策略提示、失败反馈、性能标签等更丰富上下文。

## GR00T 和 π 的共同点

GR00T 和 π 来自不同团队，生态和开放策略不同，但它们和 ACT 相比有几个共同变化。

第一，它们都把 VLM 放到 policy 中央。ACT 的视觉 encoder 是为了控制任务服务的感知模块；VLA 的 VLM 是带有大量语义知识的基础模块。语言不是可选标签，而是核心条件。模型需要把“用户说的东西”和“图像里的物体、区域、状态”对齐，再把它变成动作。

第二，它们都重视 action chunk，但 action chunk 的生成方式更强。ACT 直接预测 chunk；GR00T 用 diffusion transformer denoise 连续动作；π0 用 flow matching action expert；π0-FAST 用 tokenized action 自回归。大家都承认机器人不能只输出单步动作，关键差异在 action distribution modeling。

第三，它们都依赖大规模 heterogeneous data。ACT 可以靠 50 条 demo 训练一个任务；GR00T / π 的基础能力来自跨任务、跨机器人、跨场景、甚至人类视频和 web multimodal 数据。没有这些数据，大模型只是大，不会自动泛化。

第四，它们都不是纯规划器。它们不像传统任务规划系统那样显式输出符号计划再调用 motion planner，而是端到端或半端到端地输出动作。高层语义和低层控制在同一个模型或紧耦合系统中学习。

第五，它们都仍然需要后训练。foundation model 提供先验，不替代机器人适配。目标机器人有自己的相机标定、控制频率、夹爪、动力学、关节限制、安全约束。post-training / fine-tuning / RL refine / controller integration 仍然是工程主体。

第六，它们都把数据闭环看得比单个算法更重要。GR00T 有 Isaac synthetic data 和 post-training workflow；π 有多机器人采集、co-training、context conditioning、online RL token、memory 等连续研究。大模型路线的竞争，不只是“谁的 Transformer 更大”，而是谁能持续产生、清洗、融合和利用高质量 embodied data。

## GR00T 和 π 的区别

从公开资料看，GR00T 和 π 的差异可以从四个角度理解。

### 生态定位不同

GR00T 是 NVIDIA Isaac 生态的一部分。它天然和 Isaac Sim、Isaac Lab、Omniverse、synthetic data blueprint、Hugging Face、LeRobot 数据格式、TensorRT / Jetson / NVIDIA GPU 部署结合。对于已经在 Isaac Lab 里做仿真、任务、资产和数据生成的团队，GR00T 的工程路径更清晰。

π 是 Physical Intelligence 的通用物理智能路线。它更强调真实世界 generalist policy、open-world home environments、跨机器人数据和 emergent compositional generalization。它的研究目标更广，但完整工程栈不一定像 GR00T 那样围绕 Isaac 公开。

### 开放程度不同

GR00T N1 / N1.5 / N1.7 有官方 GitHub 和 Hugging Face 权重入口。N1.7 是 Early Access，但代码和模型可用性相对明确。

π0 / openpi 开源，适合研究者实际 fine-tune。π0.5 和 π0.7 的最新能力主要通过论文、博客、视频和部分代码生态体现，不能简单假设有同等开放权重可直接部署。讨论 π 时要分清“π0 开源模型”和“π0.7 论文展示能力”。

### 目标 embodiment 不同

GR00T 公开叙事更偏 humanoid、双臂、semi-humanoid、多指或类人操作，尤其强调 generalized humanoid robot skills。它当然也讨论 cross-embodiment，但目标用户明显是 humanoid robot developers。

π 系列覆盖单臂、双臂、移动操作等多种机器人，并在 π0.5 / π0.7 中强调家居移动操作和跨 embodiment 泛化。它不是只服务 humanoid，而是希望构建更一般的 robot foundation model。

### 技术重点不同

GR00T 的公开架构重点是 VLM + Diffusion Transformer，以及 synthetic data、人类视频、relative EEF action space、FLARE 这类与 NVIDIA 数据和模型生态结合的改进。

π 的公开技术重点是 VLM + flow matching action expert、FAST action tokenization、heterogeneous co-training、diverse context conditioning。π 更强调通过不同类型上下文和数据任务来让模型可控、可组合、可泛化。

两者不是互斥路线。实际上它们都在逼近同一个问题：如何让 VLM 的语义能力可靠地落到连续机器人控制上。

## ACT、GR00T、π 的深层差异

下面从更底层的技术维度展开。

### 1. 从“状态到动作”到“语义目标到动作”

ACT 的 policy input 通常是 observation：图像、qpos、也可能有任务 id 或 language embedding。训练目标是拟合专家 action chunk。语言如果存在，也常常只是条件变量。

GR00T / π 的 input 更像一个多模态 prompt：当前图像、语言指令、机器人状态、可能还有 embodiment tag、历史、子目标或策略上下文。它们要做的不只是局部动作预测，还包括把语言目标 grounding 到场景中。例如“把苹果拿到盘子里”，模型要区分苹果和干扰物，理解盘子是目标容器，选择哪个手抓，生成抓取和放置动作。

这就是为什么 VLM backbone 重要。没有 VLM，模型对“苹果”“盘子”“脏衣服”“水槽”“未铺好的床”这些概念的理解只能来自机器人数据；而机器人数据远远不够覆盖全部语义。

### 2. 从小数据 imitation 到大数据预训练加小数据适配

ACT 常见训练范式是收集目标任务 demo，然后从头训练或在小模型上训练。数据越接近部署分布越好。

GR00T / π 是 foundation model 范式：先用大规模数据训练一个通用模型，再针对目标机器人和任务做 post-training。这里的 post-training 有点像把大语言模型 fine-tune 到某个企业任务：你不是教它从零认识语言，而是教它适配你的格式、偏好和约束。

这种范式的优势在低数据和泛化场景下更明显。如果任务很窄、数据很好收、部署分布固定，ACT 可能更快更稳。如果任务开放、语言变化大、物体和场景多，foundation model 的先验才有发挥空间。

### 3. 从固定 action space 到跨 embodiment action abstraction

ACT 的动作空间通常固定，例如 14 维双臂关节目标、末端 pose + gripper、或者项目里的 geometry active frame 表示。模型参数和这个维度绑定。

GR00T / π 必须面对不同机器人：单臂、双臂、移动底盘、humanoid、不同夹爪、不同相机。它们需要某种 embodiment abstraction。常见办法包括：

1. 使用 embodiment tag 或 robot-specific modality config。
2. 把动作表示成末端相对位移，而不是绝对关节命令。
3. 用 action expert 处理 robot state/action，不让 VLM 直接承担所有低层差异。
4. 对不同机器人共享高层语义表示，但保留 embodiment-specific heads 或 adapters。

这也是为什么 GR00T N1.7 强调 relative EEF action space。对于跨机器人学习，统一的“手相对于当前 pose 怎么动”比“第 7 个关节转多少”更容易迁移。

### 4. 从单一监督 loss 到生成式动作建模

ACT 原始实现用 CVAE + L1/L2 action reconstruction + KL。它可以表达一定动作多样性，但测试时通常取 latent prior mean，所以很多实现更接近确定性 chunk predictor。

GR00T / π 使用 diffusion / flow matching，是更强的动作分布建模方式。对于接触任务，多模态动作很常见。比如把杯子放到水槽，左边绕和右边绕都可以；如果模型学平均，可能撞到中间障碍。生成式动作模型可以从条件分布中采样或求解更合理轨迹。

不过生成式不必然更好。它带来推理步数、延迟、稳定性和控制接口复杂度。对于高频闭环任务，小模型 ACT 可能更可控。

### 5. 从任务成功率到开放泛化能力

ACT 的评估通常是某个任务的 success rate，训练和测试分布差别可控。例如物体位置随机、背景变化、轻微干扰。

GR00T / π 的评估更强调：

1. 未见物体。
2. 未见语言组合。
3. 未见房间或厨房。
4. 跨机器人迁移。
5. 少量 demo post-training。
6. 长程多阶段任务。

这类评估更难，也更容易受 benchmark 定义影响。公开视频很有说服力，但工程判断仍要看完整失败案例、episode 数量、初始化分布、人工介入、控制频率、安全约束和是否使用额外 planner。

## 对 MiGenRL / embodied-arena 的意义

embodied-arena 当前有 Isaac Lab / Isaac Sim、任务管理、资产管理、数据生成、MiGenRL BC/ACT 训练和 rollout 闭环。理解 GR00T / π 后，最直接的问题是：我们应该怎么把这些思想用到现有系统里？

### 继续把 ACT 做扎实仍然有价值

ACT 不是被 VLA 淘汰的旧东西。相反，它是 VLA 落地前最重要的工程基线之一。原因很简单：

1. ACT 的数据契约清晰，容易定位问题。
2. ACT 训练成本低，适合快速验证 action representation、observation、reset、success。
3. 许多明确任务不需要 foundation model。
4. 即使用 GR00T / π，也需要一个 local specialist baseline 判断大模型是否真的带来收益。

在 MiGenRL 里，如果一个 task 的 ACT 在 train layout 都全 0，直接换 VLA 大概率只会把问题复杂化。应该先确认 qpos/action 维度、归一化、reset、controller、success、camera 是否正确。ACT 是系统健康检查工具。

### VLA 接入的第一步不是换模型，而是改数据规范

如果未来要接 GR00T 或 openpi，最先要做的不是写一个 `policy_backend: groot`，而是保证数据能表达 VLA 所需信息：

1. 每条 episode 有自然语言 instruction，而不是只有 task name。
2. 有一致的 camera stream，包含机器人执行所需的视觉上下文。
3. qpos/action 表示清楚，最好能转成相对 EEF 或通用末端表示。
4. 有 embodiment metadata，例如机器人类型、夹爪、关节、控制频率、坐标系。
5. 有 success / failure / subtask 标签，方便后训练、筛选和高层语义学习。
6. 如果做长程任务，要能保存阶段、子目标、关键帧、对象状态变化。

ACT 数据只需要“专家怎么动”；VLA 数据还需要“专家为什么这么动、用户想要什么、场景里哪些对象相关、这个机器人身体是什么”。

### GR00T 对 embodied-arena 的潜在路径

GR00T 和 embodied-arena 的结合点比较自然，因为两者都在 Isaac 生态上。可能路径包括：

1. 把 MiGen / MiGenRL 的 episode 转成 GR00T 支持的 LeRobot-like format。
2. 为机器人定义 modality config，包括相机、state、action、embodiment tag。
3. 将现有 geometry_active_frame action representation 映射到 relative EEF action space。
4. 用少量目标任务 demo fine-tune GR00T。
5. 在 Isaac Lab 环境里做 open-loop evaluation 和 closed-loop rollout。
6. 对成功 rollout 做筛选，回流到 MiGenRL 数据集。

这里最难的不是调用模型，而是动作语义对齐。GR00T 输出的相对 EEF action 必须被稳定转换成项目 env action。项目 AGENTS.md 已经特别强调 Isaac Lab 新版四元数顺序是 `[x, y, z, w]`，这类细节在 VLA 接入时会更危险：如果动作旋转顺序错了，大模型输出看起来正常，执行会完全错。

还要注意 Warp array 到 torch tensor 的转换。VLA 输入通常需要 torch tensor；Isaac Lab 数据可能是 Warp array，不能直接 `.to()`。这些底层契约不正确时，foundation model 没有机会表现能力。

### π / openpi 对 embodied-arena 的潜在路径

π0 / openpi 的接入更像研究路径。它的价值在于理解 VLM + flow matching action expert 的训练和 fine-tuning 范式，并验证项目数据能不能支持 VLA。

可能路径包括：

1. 选一个已有 ACT 成功率较高的任务，作为 VLA 接入 smoke test。
2. 生成带语言指令的多场景、多物体、多干扰物 demo。
3. 将数据转成 openpi 需要的格式。
4. 先做 fine-tuning，而不是期待 zero-shot。
5. 与 ACT 在 train layout、unseen layout、unseen object、language variation 上对比。
6. 分析 VLA 的失败类型：语义错、抓取错、动作不稳、控制器错、长程记忆错。

π0.5 / π0.7 的思想也能影响数据设计。比如即使暂时不用 π0.7，也可以在 MiGen 数据里增加 subgoal image、stage label、object affordance、成功质量评分、失败原因。这些上下文未来可能成为 steerable VLA 的 prompt 条件。

### 不要把 VLA 当成 motion planner 替代品

ACT、GR00T、π 都是 policy，不是严格意义的几何规划器。对于机器人执行，仍然需要：

1. 安全约束。
2. 关节限位。
3. 碰撞检测。
4. 速度和加速度限制。
5. 低层控制器。
6. 必要时的 motion planning 或 trajectory filtering。

在 curobo / Isaac Lab 环境里，VLA 可以给目标、增量或 action chunk，但不应该无条件绕过已有控制安全层。尤其是 humanoid / 双臂任务，动作空间维度高，接触复杂，policy 输出需要被控制器和环境约束消化。

## 用 ACT 用户能懂的方式理解三者

如果你已经会训练 ACT，可以用下面的类比快速理解。

### ACT 是会模仿一个师傅的徒弟

你给它看 50 条 demo，它学会某个师傅在某个工作台上怎么完成任务。师傅示教越一致，任务越清晰，硬件越稳定，它学得越好。它不需要懂很多世界知识，只要“看到这个局面，手这样动”。

它的缺点也像徒弟：换了厨房、换了工具、换了说法，它不一定知道你想干什么。

### GR00T 是带通用视觉语言知识的 humanoid 技能底座

它已经看过很多机器人数据、人类视频、合成轨迹，知道一些常见物体和动作模式。你再教它你的机器人怎么表示状态、怎么输出动作、这个任务的具体偏好。它更像一个有经验的实习工程师：懂很多概念，但必须接入你们公司的工具链和接口，才能稳定干活。

它的优势是跨任务和低数据适配；难点是模型大、接口复杂、调试链路长。

### π 是试图把机器人变成通用物理智能体的研究路线

π0 证明 VLM + flow matching 可以做多机器人 generalist policy；π0.5 证明 heterogeneous co-training 能带来新家居场景的 open-world generalization；π0.7 进一步强调通过丰富上下文 steer 模型，让它组合已有技能解决未见任务。

它更像“机器人版基础模型研究前沿”。对工程团队的价值，不只是某个 checkpoint，而是它提示我们未来数据和模型接口会变成什么样。

## 何时选 ACT，何时看 GR00T / π

实际项目里不要按热度选模型，而要按问题选。

### 优先用 ACT 的情况

1. 任务边界清晰，例如 stack、pick-place、open drawer、press button。
2. 机器人固定，action space 固定。
3. 数据可以通过 teleop / planner 快速收集。
4. 成功标准明确，episode 不太长。
5. 你需要快速证明 env、controller、data pipeline 是通的。
6. 训练资源有限，推理延迟敏感。
7. 语言不是主要变量，或语言命令种类很少。

这类场景下，ACT 的简单性是优势。大模型可能带来额外复杂度，却不一定提升成功率。

### 值得看 GR00T 的情况

1. 机器人形态接近 humanoid、双臂或 semi-humanoid。
2. 任务需要语言 grounding，例如“拿指定物体”“根据描述操作目标”。
3. 希望利用 Isaac synthetic data 和 NVIDIA 工具链。
4. 目标是少量 demo 适配多个相关任务。
5. 需要跨场景、跨对象泛化，而不是只在一个 layout 上成功。
6. 团队有足够 GPU 和工程时间处理 fine-tuning、格式转换、部署。

GR00T 的现实定位是：当 ACT 已经证明系统通了，但任务规模和泛化需求开始超过专用 policy，可以考虑引入。

### 值得看 π / openpi 的情况

1. 你想研究 VLA foundation model 的训练机制，而不只是使用 NVIDIA 栈。
2. 任务涉及多机器人、多数据源、多粒度语言。
3. 希望探索 flow matching / action expert / FAST action tokenization。
4. 长程家居或开放场景任务是核心目标。
5. 你愿意围绕 openpi 适配数据格式并做实验。
6. 你关注 π0.5 / π0.7 展示出的 co-training、context conditioning、emergent generalization 思想。

对多数工程团队来说，π0/openpi 更适合作为研究基线和未来路线参考；π0.7 的能力暂时更应看作技术方向，而不是随手可部署的组件。

## 常见误解

### 误解一：GR00T / π 是 ACT 的放大版

不准确。它们都可能预测 action chunk，但 foundation model 的核心是预训练、多模态语义和跨任务泛化。把 ACT hidden dim 放大，不会自然得到 GR00T / π；没有 VLM 先验和 heterogeneous robot data，模型只是更大的 BC。

### 误解二：VLA 有语言能力，所以不需要任务数据

不准确。语言能力让模型更容易理解目标，但机器人控制仍然需要 embodiment-specific 数据。你的相机位置、夹爪、动作频率、控制器、物体动力学都会影响执行。zero-shot 是研究目标，不是稳定工程默认值。

### 误解三：VLA 能替代所有低层控制

不准确。VLA 输出动作，但真实机器人还需要控制器、限位、安全、碰撞处理、异常恢复。大模型越通用，越要把它放进可靠的执行框架，而不是直接把输出写进电机。

### 误解四：ACT 没有语言就不能多任务

也不准确。ACT 可以扩展成 language-conditioned 或 task-conditioned policy，也可以多任务训练。只是它缺少大规模 VLM 预训练先验，因此在开放语言和开放场景上不如 VLA 路线自然。

### 误解五：公开 benchmark 高就一定适合本项目

不准确。机器人 benchmark 对初始化分布、对象集合、场景复杂度、episode 长度、是否人工 reset、失败如何计数都很敏感。引入任何大模型前，都应该在项目自己的 train layout / unseen layout / unseen object / language variation 上做对照。

## 对下一步技术路线的建议

如果目标是在 embodied-arena 里稳步推进，从 ACT 到 VLA 可以分四阶段。

第一阶段：把 ACT baseline 做成可靠标尺。每个任务至少有一个可以复现实验的 ACT 配置，明确 qpos/action 表示、chunk size、归一化、success、reset。没有这个标尺，就无法判断 VLA 的收益。

第二阶段：升级数据 schema。给 episode 增加 instruction、embodiment metadata、object metadata、stage/subgoal、failure reason、quality score。即使暂时不用 VLA，这些信息也能改善调试和数据筛选。

第三阶段：选择一个低风险任务接 open model。不要一开始做长程开放家居任务。先选一个 ACT 已经能成功、但有一定语言或物体泛化需求的任务。用 GR00T 或 openpi fine-tune，对比 ACT 在未见物体、语言改写和 layout variation 下的表现。

第四阶段：建立 rollout 数据回流。VLA 的价值在数据闭环中放大。成功 rollout 可以加入训练，失败 rollout 可以标注失败原因，仿真可以生成更多 variation。最终系统不应该只是“换一个 policy backend”，而是“让 foundation model 参与数据飞轮”。

## 一个工程判断清单

在决定引入 GR00T / π 前，可以用下面清单自检：

| 问题 | 如果答案是否定的，风险 |
| --- | --- |
| ACT baseline 是否已跑通？ | 无法区分模型问题和环境/数据问题 |
| 是否有稳定相机和 qpos/action schema？ | VLA 输入输出难以对齐 |
| 是否有自然语言 instruction？ | VLA 的语言能力无法发挥 |
| 是否能导出 LeRobot / openpi 所需格式？ | fine-tuning 成本会卡在数据转换 |
| 是否能定义相对 EEF action？ | 跨 embodiment / GR00T 接入难 |
| 是否有 GPU 预算？ | 大模型 fine-tuning 和推理成本不可忽略 |
| 是否有项目内 benchmark？ | 只能相信公开视频或外部指标 |
| 是否能保存失败案例？ | 无法进行数据闭环和错误分析 |

如果这些基础还没准备好，优先投入数据和评估基础设施，而不是直接追最新模型。

## 总结

ACT、GR00T、π 不是简单的新旧关系，而是三个层级的问题解法。

ACT 是任务级模仿学习的强基线。它用 action chunking 把高频控制变成更稳定的短时动作序列预测，适合明确任务、固定机器人和快速工程闭环。

GR00T 是 NVIDIA 面向 humanoid / generalist robot skills 的开放 VLA foundation model 栈。它用 VLM 做语义理解，用 diffusion transformer 生成连续动作，并依托 Isaac / synthetic data / post-training 生态。对 embodied-arena 这类 Isaac Lab 平台，GR00T 是最自然的外部大模型候选之一。

π 系列是 Physical Intelligence 对通用物理智能的 VLA 探索。π0 用 VLM + flow matching action expert 建立 generalist policy；π0-FAST 探索动作 token 化；π0.5 通过 heterogeneous co-training 推动 open-world generalization；π0.7 通过 diverse context conditioning 强调 steerable 和组合泛化。

从 ACT 迁移到 GR00T / π，最重要的认知变化是：模型不再只是拟合某个任务里的专家动作，而是在大规模多模态、多机器人、多任务数据中学习物理世界和语言目标的共同结构。工程上，真正的门槛也不只是换模型，而是把数据、动作表示、语言指令、评估和回流闭环升级到 foundation model 能使用的形态。

## 参考资料

- [NVIDIA Isaac GR00T 官方 GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [NVIDIA 技术博客：Accelerate Generalist Humanoid Robot Development with NVIDIA Isaac GR00T N1](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/)
- [NVIDIA Research：GR00T N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/)
- [Physical Intelligence：π0 Our First Generalist Policy](https://www.pi.website/blog/pi0)
- [π0 paper：A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [Physical Intelligence：Open Sourcing π0](https://www.pi.website/blog/openpi)
- [Physical Intelligence：π0.5 A VLA with Open-World Generalization](https://www.pi.website/blog/pi05)
- [π0.5 paper](https://arxiv.org/abs/2504.16054)
- [Physical Intelligence：π0.7 a Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07)
- [π0.7 paper](https://arxiv.org/abs/2604.15483)
- [ACT paper：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [ALOHA / ACT project page](https://tonyzhaozh.github.io/aloha/)
- [ACT GitHub](https://github.com/tonyzhaozh/act)
