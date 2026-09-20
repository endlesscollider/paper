<script setup>
import { ref, computed } from 'vue'
import ArticleCard from './ArticleCard.vue'

// items: [{ title, link, order, tags, category, star }]
// groupOrder: 可选，指定 category 分组的显示顺序，未列出的分组按字母序排在后面
const props = defineProps({
  items: { type: Array, required: true },
  groupOrder: { type: Array, default: () => [] },
  groupIcons: { type: Object, default: () => ({}) },
})

const searchQuery = ref('')

const filteredItems = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  if (!q) return props.items
  return props.items.filter(a =>
    a.title.toLowerCase().includes(q) ||
    (a.tags || []).some(t => t.toLowerCase().includes(q))
  )
})

const groups = computed(() => {
  const map = new Map()
  for (const item of filteredItems.value) {
    const cat = item.category || '未分类'
    if (!map.has(cat)) map.set(cat, [])
    map.get(cat).push(item)
  }
  for (const list of map.values()) {
    list.sort((a, b) => (a.order ?? 999) - (b.order ?? 999) || a.link.localeCompare(b.link))
  }
  const known = props.groupOrder.filter(g => map.has(g))
  const rest = [...map.keys()].filter(g => !props.groupOrder.includes(g)).sort()
  return [...known, ...rest].map(cat => ({ category: cat, items: map.get(cat) }))
})

const totalCount = computed(() => filteredItems.value.length)
</script>

<template>
  <div class="dir-index-wrapper">
    <div class="dir-search-bar">
      <div class="dir-search-input-wrapper">
        <span class="dir-search-icon">🔍</span>
        <input
          v-model="searchQuery"
          type="text"
          placeholder="搜索标题或标签…"
          class="dir-search-input"
        />
        <button v-if="searchQuery" class="dir-search-clear" @click="searchQuery = ''">✕</button>
      </div>
      <span class="dir-result-count">共 <strong>{{ totalCount }}</strong> 篇</span>
    </div>

    <div v-for="group in groups" :key="group.category" class="dir-group">
      <h2 class="dir-group-title">
        <span v-if="groupIcons[group.category]">{{ groupIcons[group.category] }}</span>
        {{ group.category }}
        <span class="dir-group-count">{{ group.items.length }}</span>
      </h2>
      <div class="article-grid">
        <ArticleCard
          v-for="article in group.items"
          :key="article.link"
          :title="article.title"
          :link="article.link"
          :star="article.star"
          :category="article.category"
          :tags="article.tags"
        />
      </div>
    </div>

    <div v-if="totalCount === 0" class="dir-empty-state">
      <p>没有匹配的文章</p>
    </div>
  </div>
</template>

<style scoped>
.dir-index-wrapper {
  margin-top: 16px;
}

.dir-search-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 28px;
}

.dir-search-input-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  transition: border-color 0.25s, box-shadow 0.25s;
}

.dir-search-input-wrapper:focus-within {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}

.dir-search-icon {
  font-size: 16px;
  opacity: 0.6;
  flex-shrink: 0;
}

.dir-search-input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  font-size: 15px;
  color: var(--vp-c-text-1);
}

.dir-search-input::placeholder {
  color: var(--vp-c-text-3);
}

.dir-search-clear {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  color: var(--vp-c-text-3);
  padding: 2px 6px;
  border-radius: 4px;
}

.dir-search-clear:hover {
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-1);
}

.dir-result-count {
  font-size: 14px;
  color: var(--vp-c-text-2);
  white-space: nowrap;
}

.dir-result-count strong {
  color: var(--vp-c-brand-1);
}

.dir-group {
  margin-bottom: 36px;
}

.dir-group-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 19px;
  font-weight: 600;
  margin: 0 0 16px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--vp-c-border);
}

.dir-group-count {
  font-size: 13px;
  font-weight: 400;
  color: var(--vp-c-text-3);
  background: var(--vp-c-default-soft);
  padding: 1px 9px;
  border-radius: 10px;
}

.dir-empty-state {
  text-align: center;
  padding: 48px 24px;
  color: var(--vp-c-text-3);
}
</style>
