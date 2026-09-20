<script setup>
import { onMounted } from 'vue'
import { useData, useRoute } from 'vitepress'
import { useRecentlyViewed } from '../composables/useRecentlyViewed'

const { frontmatter, title, site } = useData()
const route = useRoute()
const { record } = useRecentlyViewed()

onMounted(() => {
  // 只记录有 category 的文章页面（排除首页、标签页等）
  // 每日快报是时效性内容，不算"文章"，不计入最近浏览
  if (frontmatter.value.category && frontmatter.value.category !== '每日快报') {
    const pageTitle = frontmatter.value.title || title.value || document.title
    // 去掉 .html 后缀，使 link 格式与 articles.data.mts 保持一致
    let link = route.path.replace(/\.html$/, '')
    // 去掉站点 base 前缀（生产环境部署在 /paper/ 子路径下），
    // 否则存下来的 link 会和 articles.data.mts 里不带 base 的 link 对不上，
    // 导致首页"最近浏览"匹配不到文章数据，卡片显示为空壳
    const base = site.value.base
    if (base && base !== '/' && link.startsWith(base)) {
      link = '/' + link.slice(base.length)
    }
    // 浏览器的 route.path 对中文等非 ASCII 字符是 percent-encoded 的
    // （如 /前置知识/xxx 会变成 /%E5%89%8D...），而 articles.data.mts
    // 里的 link 是明文 Unicode。这里统一 decode 后再存，
    // 否则首页"最近浏览"匹配不到文章数据，卡片会退化成空壳（无分类/标签/星级）
    try {
      link = decodeURIComponent(link)
    } catch {}
    record(pageTitle, link)
  }
})
</script>

<template>
  <span style="display: none;" />
</template>
