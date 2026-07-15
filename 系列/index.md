---
title: 系列文章
---

# 系列文章

长篇系列教程，每个系列由多章组成，系统性地讲解一个完整主题。

<script setup>
import { data as articles } from '../.vitepress/theme/articles.data.mts'
import ArticleCard from '../.vitepress/theme/components/ArticleCard.vue'
import { computed } from 'vue'

const seriesArticles = computed(() => {
  return articles.filter(a => a.series)
})
</script>

<div class="article-grid" v-if="seriesArticles.length">
  <ArticleCard
    v-for="article in seriesArticles"
    :key="article.link"
    :title="article.title"
    :link="article.link"
    :star="article.star"
    :category="article.category"
    :tags="article.tags"
    :series="article.series"
  />
</div>

<p v-else style="color: var(--vp-c-text-3); font-style: italic;">
暂无系列文章，敬请期待。
</p>

<style>
.article-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  margin-top: 24px;
}
@media (min-width: 640px) {
  .article-grid {
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  }
}
</style>
