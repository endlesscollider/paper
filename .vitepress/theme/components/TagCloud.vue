<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { data as articles } from '../articles.data.mts'
import { topicClusters, articleMatchesTopic } from '../topicClusters.mjs'
import ArticleCard from './ArticleCard.vue'

const searchQuery = ref('')
const activeCategory = ref('全部')
const activeTopic = ref('')
const activeTag = ref('')
const showRareTags = ref(false)
const sortMode = ref('star') // 'star' | 'new' | 'old'

// 内容类型 Tab：用「路径前缀 / category」双重匹配，修复历史 bug：
// - 硬件基础目录下文章的 category 实际写的是"执行器基础"等子分类，从不等于"硬件基础"，
//   如果直接用 category === '硬件基础' 匹配，这个 tab 永远是空的。
// - 前置知识目录下有少数文章 category 被标成"机器人基础"/"数学基础"等子分类，
//   直接用 category === '前置知识' 匹配会漏掉这些文章。
// 统一改成按目录路径前缀匹配，从根本上避免 category 字段被挪用做子分类时导致的漏匹配。
const categories = [
  { key: '全部', match: () => true },
  { key: '综述', match: a => a.category === '综述' },
  { key: '精读', match: a => a.category === '精读' },
  { key: '前置知识', match: a => a.link.startsWith('/前置知识/') },
  { key: '工程实践', match: a => a.link.startsWith('/工程实践/') },
  { key: '工程项目', match: a => a.link.startsWith('/工程项目/') },
  { key: '系列文章', match: a => a.link.startsWith('/系列/') },
  { key: '硬件基础', match: a => a.link.startsWith('/硬件基础/') },
]

const categoryIcons = {
  '全部': '📚',
  '综述': '🗺️',
  '精读': '🔬',
  '前置知识': '🧠',
  '工程实践': '🛠️',
  '工程项目': '🚀',
  '系列文章': '📖',
  '硬件基础': '⚙️',
}

const categoryNames = categories.map(c => c.key)

// 从 URL query 恢复状态，方便从首页/其他页面直接带参数跳转进来
function readStateFromURL() {
  if (typeof window === 'undefined') return
  const params = new URLSearchParams(window.location.search)
  const cat = params.get('category')
  const topic = params.get('topic')
  const tag = params.get('tag')
  const sort = params.get('sort')
  const q = params.get('q')
  if (cat && categoryNames.includes(cat)) activeCategory.value = cat
  if (topic) activeTopic.value = topic
  if (tag) activeTag.value = tag
  if (sort && ['star', 'new', 'old'].includes(sort)) sortMode.value = sort
  if (q) searchQuery.value = q
}

function writeStateToURL() {
  if (typeof window === 'undefined') return
  const params = new URLSearchParams()
  if (activeCategory.value && activeCategory.value !== '全部') params.set('category', activeCategory.value)
  if (activeTopic.value) params.set('topic', activeTopic.value)
  if (activeTag.value) params.set('tag', activeTag.value)
  if (sortMode.value && sortMode.value !== 'star') params.set('sort', sortMode.value)
  if (searchQuery.value) params.set('q', searchQuery.value)
  const query = params.toString()
  const newURL = window.location.pathname + (query ? '?' + query : '')
  window.history.replaceState(null, '', newURL)
}

onMounted(() => {
  readStateFromURL()
  window.addEventListener('popstate', readStateFromURL)
})

watch([activeCategory, activeTopic, activeTag, sortMode, searchQuery], writeStateToURL)

// 当前内容类型 Tab 下的全部文章
const articlesInCategory = computed(() => {
  const cat = categories.find(c => c.key === activeCategory.value)
  return cat ? articles.filter(cat.match) : articles
})

// 主题速览：只展示在当前 Tab 下确实有文章命中的主题，按命中数从多到少排（0 篇的主题直接隐藏，
// 避免用户点进去发现是死胡同）
const topicsWithCount = computed(() => {
  return topicClusters
    .map(topic => ({
      ...topic,
      count: articlesInCategory.value.filter(a => articleMatchesTopic(a, topic)).length,
    }))
    .filter(t => t.count > 0)
    .sort((a, b) => b.count - a.count)
})

// 细分标签云：按出现频率排序，只固定展示前 24 个作为「主标签」，其余全部收进折叠区。
// 用固定数量而不是「出现次数 > N」这种阈值，是因为不同 Tab/主题下文章子集大小差异很大，
// 阈值在小样本里会失效（比如硬件基础 Tab 只有 9 篇文章，几乎所有标签都只出现 1 次）。
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

const MAIN_TAG_LIMIT = 24
const mainTags = computed(() => tags.value.slice(0, MAIN_TAG_LIMIT))
const rareTags = computed(() => tags.value.slice(MAIN_TAG_LIMIT))

// 最终过滤：内容类型 Tab（必选）+ 主题 或 细分标签（二选一，避免交叉条件把结果筛成空集）
// + 标题/标签关键词临时搜索框（可与前两者叠加，用于在已筛选的子集里快速定位）
const filteredArticles = computed(() => {
  let list = articlesInCategory.value
  if (activeTopic.value) {
    const topic = topicClusters.find(t => t.name === activeTopic.value)
    if (topic) list = list.filter(a => articleMatchesTopic(a, topic))
  } else if (activeTag.value) {
    list = list.filter(a => a.tags.includes(activeTag.value))
  }
  const q = searchQuery.value.trim().toLowerCase()
  if (q) {
    list = list.filter(a =>
      a.title.toLowerCase().includes(q) ||
      (a.tags || []).some(t => t.toLowerCase().includes(q))
    )
  }
  return list
})

const sortedArticles = computed(() => {
  const list = [...filteredArticles.value]
  if (sortMode.value === 'star') {
    list.sort((a, b) => b.star - a.star || a.order - b.order)
  } else if (sortMode.value === 'new') {
    list.sort((a, b) => b.lastUpdated - a.lastUpdated)
  } else {
    list.sort((a, b) => a.order - b.order)
  }
  return list
})

function switchCategory(cat) {
  activeCategory.value = cat
  activeTopic.value = ''
  activeTag.value = ''
}

function toggleTopic(name) {
  activeTopic.value = activeTopic.value === name ? '' : name
  activeTag.value = ''
}

function toggleTag(tag) {
  activeTag.value = activeTag.value === tag ? '' : tag
  activeTopic.value = ''
}

function clearFilters() {
  activeTopic.value = ''
  activeTag.value = ''
  searchQuery.value = ''
}
</script>

<template>
  <div class="tag-cloud-wrapper">
    <!-- 临时关键词查询：只在当前已筛选的子集里做标题/标签的即时过滤，
         不是全站搜索（全站搜索用顶部导航栏的 Search，那个是"临时查询"，这里是"浏览定位"） -->
    <div class="quick-filter-bar">
      <div class="quick-filter-input-wrapper">
        <span class="quick-filter-icon">🔎</span>
        <input
          v-model="searchQuery"
          type="text"
          placeholder="在当前筛选结果里按标题/标签快速定位…"
          class="quick-filter-input"
        />
        <button v-if="searchQuery" class="quick-filter-clear" @click="searchQuery = ''">✕</button>
      </div>
    </div>

    <!-- 内容类型 Tab -->
    <div class="category-tabs">
      <button
        v-for="cat in categories"
        :key="cat.key"
        class="category-tab"
        :class="{ active: activeCategory === cat.key }"
        @click="switchCategory(cat.key)"
      >
        <span class="category-icon">{{ categoryIcons[cat.key] }}</span>
        <span class="category-label">{{ cat.key }}</span>
      </button>
    </div>

    <!-- 主题速览：这是找文章的主入口，人工把高频概念聚类成语义主题，
         比原始标签云更适合"我想找某个方向的所有文章"这种浏览场景 -->
    <div class="topic-section" v-if="topicsWithCount.length">
      <div class="topic-section-header">
        <span class="topic-section-title">🧭 主题速览</span>
        <span class="topic-section-desc">按方向浏览，点击进入</span>
      </div>
      <div class="topic-grid">
        <button
          v-for="topic in topicsWithCount"
          :key="topic.name"
          class="topic-tile"
          :class="{ active: activeTopic === topic.name }"
          @click="toggleTopic(topic.name)"
        >
          <span class="topic-icon">{{ topic.icon }}</span>
          <span class="topic-name">{{ topic.name }}</span>
          <span class="topic-count">{{ topic.count }}</span>
        </button>
      </div>
    </div>

    <!-- 细分标签云：主题速览覆盖不到的更精确的关键词，仍保留作为补充筛选手段 -->
    <div class="tag-cloud-section" v-if="tags.length">
      <div class="topic-section-header">
        <span class="topic-section-title">🏷️ 细分标签</span>
        <span class="topic-section-desc">更精确的关键词筛选</span>
      </div>
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

      <!-- 折叠的长尾标签 -->
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
            <span class="tag-count">{{ tag.count }}</span>
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
      <span v-if="activeTopic || activeTag || searchQuery" class="active-filter" @click="clearFilters">
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
        :series="article.series"
      />
    </div>

    <!-- 空状态 -->
    <div v-if="filteredArticles.length === 0" class="empty-state">
      <p>暂无匹配的文章</p>
    </div>
  </div>
</template>
