---
title: 数据集
---

# 数据集

本栏目收录用于三维部件分割、开放词汇 3D 分割及高斯泼溅场景重建评测的公开数据集，覆盖 CAD 合成、真实扫描与专项 3DGS 评测三类来源。每篇条目说明数据规模、标注粒度、获取方式，以及在 3DGS 部件分割流程中的适用位置。

<script setup>
import { computed } from 'vue'
import { data as articles } from '../.vitepress/theme/articles.data.mts'
import DirectoryIndex from '../.vitepress/theme/components/DirectoryIndex.vue'

const items = computed(() => articles.filter(a => a.link.startsWith('/数据集/')))
</script>

<DirectoryIndex
  :items="items"
  :group-order="['CAD 合成', '真实扫描', '3DGS 评测']"
  :group-icons="{ 'CAD 合成': '🔷', '真实扫描': '📷', '3DGS 评测': '🔬' }"
/>
