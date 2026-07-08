<script setup>
import { onMounted } from 'vue'
import { useData, useRoute } from 'vitepress'
import { useRecentlyViewed } from '../composables/useRecentlyViewed'

const { frontmatter, title } = useData()
const route = useRoute()
const { record } = useRecentlyViewed()

onMounted(() => {
  // 只记录有 category 的文章页面（排除首页、标签页等）
  if (frontmatter.value.category) {
    const pageTitle = frontmatter.value.title || title.value || document.title
    record(pageTitle, route.path)
  }
})
</script>

<template>
  <span style="display: none;" />
</template>
