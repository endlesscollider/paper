// 主题聚类（Topic Clusters）
//
// 原始 tags 有 600+ 个，大部分只出现过 1 次，直接铺成标签云根本没法「浏览」——
// 只能靠精确记住关键词去搜索。这个文件手工把高频标签聚类成几十个语义主题，
// 作为「按标签浏览」页面的核心导航层：用户先选主题（粗粒度、语义化），
// 再在结果里用细分标签/分类做二次筛选。
//
// 新增文章时，如果引入了新的高频概念簇，应该在这里补充对应的 topic
// （而不是让它散落在原始标签云里没人找得到）。

export const topicClusters = [
  {
    name: 'VLA 架构与基础',
    icon: '🤖',
    tags: ['VLA', '基础模型', '自回归', 'Token化', '连续动作', '动作分块', 'Action Chunking', 'π₀', 'GR00T', 'π0.6', 'OpenVLA', 'RT-2', '通用策略', '跨体学习'],
  },
  {
    name: 'VLA 强化学习后训练',
    icon: '🎯',
    tags: ['PPO', 'GRPO', 'SAC', 'Off-Policy', 'off-policy', 'On-Policy', 'on-policy', 'Offline RL', 'Residual RL', '策略优化', '奖励设计', '稀疏奖励', '过程奖励', 'VLM奖励', 'reward-free', '偏好优化', '偏好对齐', '后训练'],
  },
  {
    name: '持续学习与防遗忘',
    icon: '🧩',
    tags: ['持续学习', '灾难性遗忘', '经验回放', '终身学习', 'EWC', '知识蒸馏', '自蒸馏', '记忆机制'],
  },
  {
    name: '扩散模型与 Flow Matching',
    icon: '🌊',
    tags: ['扩散模型', 'Flow Matching', 'DiT', 'VAE', '扩散策略', 'Consistency Model', '生成模型', '条件生成', '视频生成'],
  },
  {
    name: 'LoRA / 参数高效微调',
    icon: '🪶',
    tags: ['LoRA', 'PEFT', '参数高效微调', 'QLoRA', 'AdaLoRA', 'DoRA', '微调', '模型合并'],
  },
  {
    name: 'Transformer 与 Attention',
    icon: '🔷',
    tags: ['Transformer', 'Attention', '注意力机制', 'MHA', 'MQA', 'GQA', 'MLA', 'KV-Cache'],
  },
  {
    name: '位置编码与长度外推',
    icon: '📏',
    tags: ['位置编码', 'RoPE', 'ALiBi', 'NoPE', '长度外推'],
  },
  {
    name: '归一化方法',
    icon: '📐',
    tags: ['归一化', 'BatchNorm', 'LayerNorm', 'RMSNorm', 'GroupNorm', 'InstanceNorm'],
  },
  {
    name: '激活函数与门控',
    icon: '⚡',
    tags: ['激活函数', 'GLU', 'SwiGLU', 'GeGLU', 'ReLU', '门控机制'],
  },
  {
    name: '优化器',
    icon: '🏔️',
    tags: ['优化器', 'Adam', 'AdamW', 'Muon', 'SGD', 'Momentum', 'Newton-Schulz'],
  },
  {
    name: '损失函数',
    icon: '📉',
    tags: ['损失函数', '交叉熵', 'Focal Loss', '对比学习', '类别不平衡'],
  },
  {
    name: '权重初始化',
    icon: '🎲',
    tags: ['权重初始化', 'Xavier', 'Kaiming', 'Fixup', '初始化'],
  },
  {
    name: '正则化与过拟合',
    icon: '🛡️',
    tags: ['正则化', '过拟合', 'Dropout', 'DropPath', 'Mixup', 'CutMix', '数据增强'],
  },
  {
    name: '序列建模架构',
    icon: '🔁',
    tags: ['序列建模', 'RNN', 'SSM', '状态空间模型', 'Mamba', 'RWKV', '选择性扫描'],
  },
  {
    name: '模型量化与压缩',
    icon: '🗜️',
    tags: ['模型量化', '量化', 'PTQ', 'QAT', 'GPTQ', 'AWQ', 'INT4', '模型压缩', '显存优化', '混合精度'],
  },
  {
    name: '强化学习基础',
    icon: '🧠',
    tags: ['强化学习', '策略梯度', '优势函数', 'GAE', 'TD学习', 'Q学习', 'Actor-Critic', '价值函数'],
  },
  {
    name: '离线 / 目标条件 RL',
    icon: '📦',
    tags: ['离线RL', 'Offline RL', 'CQL', 'IQL', 'AWR', '目标条件RL', '离线到在线RL', '经验回放'],
  },
  {
    name: '世界模型',
    icon: '🌍',
    tags: ['世界模型', 'Model-Based RL', 'DreamerV3', '想象训练', '视频预测'],
  },
  {
    name: '机器人学基础',
    icon: '🦾',
    tags: ['机器人学', '运动学', '雅可比矩阵', '正运动学', '执行器', '灵巧手', '触觉'],
  },
  {
    name: '物理仿真与动力学',
    icon: '🪢',
    tags: [
      '物理仿真',
      '动力学',
      '刚体动力学',
      '机器人动力学',
      '刚体',
      '碰撞',
      '碰撞检测',
      '接触力学',
      '约束雅可比',
      '冲量',
      '软体仿真',
      '布料仿真',
      '绳索仿真',
      '触觉仿真',
      '数值积分',
      '有限元',
      '物质点法',
    ],
  },
  {
    name: '机器人硬件',
    icon: '⚙️',
    tags: ['硬件', '电机', '真实机器人'],
  },
  {
    name: '模仿学习与数据',
    icon: '📼',
    tags: ['模仿学习', '数据集', '数据合成', 'ACT', '机器人数据', '数据高效'],
  },
  {
    name: 'Sim-to-Real 与仿真',
    icon: '🕹️',
    tags: ['Sim-to-Real', '仿真', '泛化'],
  },
  {
    name: 'VLM 与多模态',
    icon: '🖼️',
    tags: ['VLM', '多模态', 'CLIP', 'SigLIP', '双编码器'],
  },
  {
    name: '量化交易',
    icon: '📈',
    tags: ['量化交易', '截面模型', '分布漂移'],
  },
  {
    name: '数学与概率基础',
    icon: '📚',
    tags: ['概率论', '线性代数', '数学基础', '群论', '偏差方差权衡'],
  },
  {
    name: '大模型训练基础设施',
    icon: '🏗️',
    tags: ['分布式训练', '数据并行', '训练基础设施', '大模型推理', '推理优化'],
  },
]

/** 根据文章 tags 判断是否属于某个主题簇（OR 匹配：命中任意一个关键 tag 即算） */
export function articleMatchesTopic(article, topic) {
  const set = new Set(topic.tags)
  return (article.tags || []).some(t => set.has(t))
}
