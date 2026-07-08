<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { data as articles } from '../articles.data.mts'
import ArticleCard from './ArticleCard.vue'

const activeCategory = ref('全部')
const activeTag = ref('')
const showRareTags = ref(false)

const categories = ['全部', '综述', '精读', '工程实践', '工程项目']

const categoryIcons = {
  '全部': '📚',
  '综述': '🗺️',
  '精读': '🔬',
  '工程实践': '🛠️',
  '工程项目': '🚀',
}

// 从 URL query 恢复状态
function readStateFromURL() {
  if (typeof window === 'undefined') return
  const params = new URLSearchParams(window.location.search)
  const cat = params.get('category')
  const tag = params.get('tag')
  if (cat && categories.includes(cat)) {
    activeCategory.value = cat
  }
  if (tag) {
    activeTag.value = tag
  }
}

// 将状态写入 URL query（replace 不产生历史记录条目）
function writeStateToURL() {
  if (typeof window === 'undefined') return
  const params = new URLSearchParams()
  if (activeCategory.value && activeCategory.value !== '全部') {
    params.set('category', activeCategory.value)
  }
  if (activeTag.value) {
    params.set('tag', activeTag.value)
  }
  const query = params.toString()
  const newURL = window.location.pathname + (query ? '?' + query : '')
  window.history.replaceState(null, '', newURL)
}

onMounted(() => {
  readStateFromURL()
  window.addEventListener('popstate', readStateFromURL)
})

watch([activeCategory, activeTag], writeStateToURL)

const articlesInCategory = computed(() => {
  if (activeCategory.value === '全部') return articles
  return articles.filter(a => a.category === activeCategory.value)
})

const tags = computed(() => {
  const tagMap = {}
  for (const article of articlesInCategory.value) {
    for (const tag of article.tags) {
      if (!tagMap[tag]) tagMap[tag] = 0
      tagMap[tag]++
    }
  }
  return Object.entries(tagMap)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name, count }))
})

const mainTags = computed(() => tags.value.filter(t => t.count > 1))
const rareTags = computed(() => tags.value.filter(t => t.count === 1))

const filteredArticles = computed(() => {
  let list = articlesInCategory.value
  if (activeTag.value) {
    list = list.filter(a => a.tags.includes(activeTag.value))
  }
  return list
})

const sortMode = ref('star') // 'star' | 'new' | 'old'

const sortedArticles = computed(() => {
  const list = [...filteredArticles.value]
  if (sortMode.value === 'star') {
    list.sort((a, b) => b.star - a.star || a.order - b.order)
  } else if (sortMode.value === 'new') {
    list.sort((a, b) => b.order - a.order)
  } else {
    list.sort((a, b) => a.order - b.order)
  }
  return list
})

function switchCategory(cat) {
  activeCategory.value = cat
  activeTag.value = ''
}

function toggleTag(tag) {
  activeTag.value = activeTag.value === tag ? '' : tag
}
</script>

<template>
  <div class="tag-cloud-wrapper">
    <!-- 分类 Tab -->
    <div class="category-tabs">
      <button
        v-for="cat in categories"
        :key="cat"
        class="category-tab"
        :class="{ active: activeCategory === cat }"
        @click="switchCategory(cat)"
      >
        <span class="category-icon">{{ categoryIcons[cat] }}</span>
        <span class="category-label">{{ cat }}</span>
      </button>
    </div>

    <!-- 标签云 -->
    <div class="tag-cloud-section" v-if="tags.length">
      <div class="tag-cloud">
        <button
          v-for="tag in mainTags"
          :key="tag.name"
          class="tag-btn"
          :class="{ active: activeTag === tag.name }"
          @click="toggleTag(tag.name)"
        >
          <span class="tag-name"># {{ tag.name }}</span>
          <span class="tag-count">{{ tag.count }}</span>
        </button>
      </div>

      <!-- 折叠的低频标签 -->
      <div v-if="rareTags.length" class="rare-tags-section">
        <button class="rare-tags-toggle" @click="showRareTags = !showRareTags">
          <span>{{ showRareTags ? '收起' : `展开其余 ${rareTags.length} 个标签` }}</span>
          <span class="toggle-arrow" :class="{ expanded: showRareTags }">›</span>
        </button>
        <div class="tag-cloud rare-tags" v-show="showRareTags">
          <button
            v-for="tag in rareTags"
            :key="tag.name"
            class="tag-btn"
            :class="{ active: activeTag === tag.name }"
            @click="toggleTag(tag.name)"
          >
            <span class="tag-name"># {{ tag.name }}</span>
          </button>
        </div>
      </div>
    </div>

    <!-- 结果统计 -->
    <div class="result-bar">
      <span class="result-hint">
        共 <strong>{{ filteredArticles.length }}</strong> 篇文章
      </span>
      <div class="sort-controls">
        <button class="sort-btn" :class="{ active: sortMode === 'star' }" @click="sortMode = 'star'">⭐ 星级</button>
        <button class="sort-btn" :class="{ active: sortMode === 'new' }" @click="sortMode = 'new'">🆕 最新</button>
        <button class="sort-btn" :class="{ active: sortMode === 'old' }" @click="sortMode = 'old'">📅 最早</button>
      </div>
      <span v-if="activeTag" class="active-filter" @click="activeTag = ''">
        清除筛选 ✕
      </span>
    </div>

    <!-- 文章列表 -->
    <div class="article-grid">
      <ArticleCard
        v-for="article in sortedArticles"
        :key="article.link"
        :title="article.title"
        :link="article.link"
        :star="article.star"
        :category="article.category"
        :tags="article.tags"
      />
    </div>

    <!-- 空状态 -->
    <div v-if="filteredArticles.length === 0" class="empty-state">
      <p>暂无匹配的文章</p>
    </div>
  </div>
</template>
