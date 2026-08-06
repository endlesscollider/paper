---
layout: page
title: 前置知识搜索
---

<script setup>
import PrereqSearch from './.vitepress/theme/components/PrereqSearch.vue'
</script>

<div class="prereq-page">
  <div class="prereq-page-header">
    <h1 class="prereq-page-title">🧠 前置知识搜索</h1>
    <p class="prereq-page-desc">按标签筛选或关键词搜索，快速定位你需要的基础概念</p>
  </div>
  <PrereqSearch />
</div>

<style>
.prereq-page {
  max-width: 900px;
  margin: 0 auto;
  padding: 32px 24px;
}

.prereq-page-header {
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--vp-c-border);
}

.prereq-page-title {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--vp-c-brand-1), var(--vp-c-brand-2, #6366f1));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 8px 0;
}

.prereq-page-desc {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 15px;
}

@media (min-width: 768px) {
  .prereq-page {
    padding: 48px 48px;
  }
  .prereq-page-title {
    font-size: 32px;
  }
}
</style>
