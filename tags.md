---
layout: page
title: 找文章
---

<script setup>
import TagCloud from './.vitepress/theme/components/TagCloud.vue'
</script>

<div class="tags-page">
  <div class="tags-page-header">
    <h1 class="tags-page-title">🧭 找文章</h1>
    <p class="tags-page-desc">全站文章浏览入口：先按内容类型筛一层，再按主题或标签定位。顶部导航栏的搜索框适合"我知道要找什么关键词"的临时查询；这里适合"我想看看某个方向都有什么"。</p>
  </div>
  <TagCloud />
</div>

<style>
.tags-page {
  max-width: 1152px;
  margin: 0 auto;
  padding: 32px 24px;
}

.tags-page-header {
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--vp-c-border);
}

.tags-page-title {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--vp-c-brand-1), var(--vp-c-brand-2, #6366f1));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 8px 0;
}

.tags-page-desc {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 14px;
  line-height: 1.6;
}

@media (min-width: 768px) {
  .tags-page {
    padding: 48px 48px;
  }
  .tags-page-title {
    font-size: 32px;
  }
}

@media (min-width: 1280px) {
  .tags-page {
    padding: 48px 64px;
  }
}
</style>
