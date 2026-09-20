---
title: 前置知识
---

# 前置知识

本栏目整理机器人策略学习所需的核心前置概念，覆盖强化学习、扩散模型、Transformer、机器人学、数学基础等方向。下方列表由文章 frontmatter 自动生成，新增文章会自动出现，无需手动维护。也可以去 [🧭 找文章](/tags) 按主题/标签筛选全站文章。

<script setup>
import { data as prereqs } from '../.vitepress/theme/prereqs.data.mts'
import DirectoryIndex from '../.vitepress/theme/components/DirectoryIndex.vue'
</script>

<DirectoryIndex
  :items="prereqs"
  :group-order="['前置知识', '强化学习', '深度学习基础', '机器人基础', '物理仿真', '数学基础']"
  :group-icons="{ 前置知识: '🧠', 强化学习: '🤖', 深度学习基础: '🧬', 机器人基础: '🦾', 物理仿真: '🪢', 数学基础: '📐' }"
/>
