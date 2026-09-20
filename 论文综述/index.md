---
title: 论文综述
---

# 论文综述

本栏目对深度学习、强化学习与机器人学习领域的核心方向进行系统性综述，并对代表性论文做逐段深度精读。下方列表由文章 frontmatter 自动生成，新增文章会自动出现，无需手动维护。

<script setup>
import { computed } from 'vue'
import { data as articles } from '../.vitepress/theme/articles.data.mts'
import DirectoryIndex from '../.vitepress/theme/components/DirectoryIndex.vue'

const items = computed(() => articles.filter(a => a.link.startsWith('/论文综述/')))
</script>

<DirectoryIndex :items="items" :group-order="['综述', '精读']" :group-icons="{ 综述: '🗺️', 精读: '🔬' }" />
