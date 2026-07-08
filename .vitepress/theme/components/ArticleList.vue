<script setup>
import { computed } from 'vue'
import { data as articles } from '../articles.data.mts'
import ArticleCard from './ArticleCard.vue'

const props = defineProps({
  category: { type: String, default: '' },
  tag: { type: String, default: '' },
})

const filtered = computed(() => {
  let list = articles
  if (props.category) {
    list = list.filter(a => a.category === props.category)
  }
  if (props.tag) {
    list = list.filter(a => a.tags.includes(props.tag))
  }
  return list
})
</script>

<template>
  <div class="article-grid">
    <ArticleCard
      v-for="article in filtered"
      :key="article.link"
      :title="article.title"
      :link="article.link"
      :star="article.star"
      :category="article.category"
      :tags="article.tags"
    />
  </div>
</template>
