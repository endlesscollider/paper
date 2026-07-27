---
layout: page
title: 每日快报
---

<script setup>
import { ref, computed } from 'vue'
import { withBase } from 'vitepress'
import { data as articles } from '../.vitepress/theme/articles.data.mts'

const dailyArticles = computed(() => {
  return [...articles]
    .filter(a => a.category === '每日快报')
    .sort((a, b) => b.order - a.order)
})
</script>

# 🗞️ 每日快报

> 每日自动搜罗 **机器人**、**深度学习**、**强化学习**、**机器人仿真**、**金融量化** 领域的重要论文、项目和资讯。
>
> 由 GitHub Actions + LLM 全自动生成，每天 UTC 16:00（北京时间 0:00）更新。

<div class="daily-list">
  <div v-for="article in dailyArticles" :key="article.link" class="daily-card">
    <a :href="withBase(article.link)">
      <span class="daily-date">{{ article.title.replace('每日快报 ', '') }}</span>
      <span class="daily-arrow">→</span>
    </a>
  </div>
  <div v-if="!dailyArticles.length" class="daily-empty">
    暂无快报，明天见 👋
  </div>
</div>

<style>
.daily-list {
  margin-top: 24px;
}
.daily-card {
  margin-bottom: 8px;
}
.daily-card a {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-radius: 12px;
  border: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg-soft);
  text-decoration: none;
  color: var(--vp-c-text-1);
  transition: all 0.2s;
}
.daily-card a:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateX(4px);
}
.daily-date {
  font-size: 16px;
  font-weight: 500;
}
.daily-arrow {
  color: var(--vp-c-text-3);
}
.daily-empty {
  text-align: center;
  padding: 48px;
  color: var(--vp-c-text-3);
  font-size: 16px;
}
</style>
