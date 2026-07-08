---
layout: page
title: 按标签浏览
---

<script setup>
import TagCloud from './.vitepress/theme/components/TagCloud.vue'
</script>

<div class="tags-page">
  <div class="tags-page-header">
    <h1 class="tags-page-title">按标签浏览</h1>
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
  margin: 0;
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
