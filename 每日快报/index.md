---
layout: page
title: 每日快报
---

<script setup>
import { ref, computed } from 'vue'
import { withBase } from 'vitepress'
import { data as dailyArticles } from '../.vitepress/theme/daily.data.mts'
</script>

# 🗞️ 每日快报

> 每日自动搜罗 **机器人**、**深度学习**、**强化学习**、**机器人仿真**、**金融量化** 领域的重要论文、项目和资讯。
>
> 由 GitHub Actions + LLM 全自动生成，每天 UTC 16:00（北京时间 0:00）更新。

<div class="daily-list">
  <div v-for="article in dailyArticles" :key="article.link" class="daily-card" :class="{ 'daily-empty-card': article.isEmpty }">
    <a :href="withBase(article.link)" class="daily-card-link">
      <div class="daily-header">
        <span class="daily-date">{{ article.date }}</span>
        <span class="daily-weekday">{{ article.weekday }}</span>
        <span v-if="!article.isEmpty" class="daily-count">{{ article.totalItems }} 条</span>
        <span v-else class="daily-no-data">暂无数据</span>
      </div>
      <div v-if="article.sections.length" class="daily-sections">
        <span v-for="sec in article.sections" :key="sec.name" class="daily-section-tag">
          {{ sec.emoji }} {{ sec.name }} <b>{{ sec.count }}</b>
        </span>
      </div>
      <div v-if="article.highlights.length" class="daily-highlights">
        <div v-for="(h, idx) in article.highlights.slice(0, 3)" :key="idx" class="daily-highlight-item">
          <span class="highlight-bullet">🔥</span>
          <span class="highlight-text">{{ h.title }}</span>
        </div>
      </div>
    </a>
  </div>
  <div v-if="!dailyArticles.length" class="daily-empty">
    暂无快报，明天见 👋
  </div>
</div>

<style>
.daily-list {
  margin-top: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.daily-card-link {
  display: block;
  padding: 18px 22px;
  border-radius: 12px;
  border: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg-soft);
  text-decoration: none !important;
  color: var(--vp-c-text-1);
  transition: all 0.2s;
}

.daily-card-link:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.daily-empty-card .daily-card-link {
  opacity: 0.5;
}

.daily-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.daily-date {
  font-size: 17px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.daily-weekday {
  font-size: 13px;
  color: var(--vp-c-text-3);
  background: var(--vp-c-bg-mute);
  padding: 2px 8px;
  border-radius: 4px;
}

.daily-count {
  margin-left: auto;
  font-size: 13px;
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

.daily-no-data {
  margin-left: auto;
  font-size: 13px;
  color: var(--vp-c-text-3);
}

.daily-sections {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}

.daily-section-tag {
  font-size: 12px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-mute);
  padding: 3px 8px;
  border-radius: 6px;
}

.daily-section-tag b {
  color: var(--vp-c-text-1);
  margin-left: 2px;
}

.daily-highlights {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-top: 8px;
  border-top: 1px solid var(--vp-c-divider);
}

.daily-highlight-item {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  font-size: 13px;
  line-height: 1.5;
  color: var(--vp-c-text-2);
}

.highlight-bullet {
  flex-shrink: 0;
  font-size: 12px;
}

.highlight-text {
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 1;
  -webkit-box-orient: vertical;
}

.daily-empty {
  text-align: center;
  padding: 48px;
  color: var(--vp-c-text-3);
  font-size: 16px;
}
</style>
