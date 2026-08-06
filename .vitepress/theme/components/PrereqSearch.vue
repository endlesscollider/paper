<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { data as prereqs } from '../prereqs.data.mts'

const searchQuery = ref('')
const activeTag = ref('')
const showAllTags = ref(false)

// 从 URL query 恢复状态
function readStateFromURL() {
  if (typeof window === 'undefined') return
  const params = new URLSearchParams(window.location.search)
  const tag = params.get('tag')
  const q = params.get('q')
  if (tag) activeTag.value = tag
  if (q) searchQuery.value = q
}

function writeStateToURL() {
  if (typeof window === 'undefined') return
  const params = new URLSearchParams()
  if (activeTag.value) params.set('tag', activeTag.value)
  if (searchQuery.value) params.set('q', searchQuery.value)
  const query = params.toString()
  const newURL = window.location.pathname + (query ? '?' + query : '')
  window.history.replaceState(null, '', newURL)
}

onMounted(() => {
  readStateFromURL()
  window.addEventListener('popstate', readStateFromURL)
})

watch([activeTag, searchQuery], writeStateToURL)

// 构建标签统计
const tags = computed(() => {
  const tagMap = {}
  for (const item of prereqs) {
    for (const tag of item.tags) {
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

// 过滤文章
const filteredPrereqs = computed(() => {
  let list = prereqs

  // 标签过滤
  if (activeTag.value) {
    list = list.filter(a => a.tags.includes(activeTag.value))
  }

  // 搜索过滤（匹配标题和标签）
  if (searchQuery.value.trim()) {
    const q = searchQuery.value.trim().toLowerCase()
    list = list.filter(a =>
      a.title.toLowerCase().includes(q) ||
      a.tags.some(t => t.toLowerCase().includes(q))
    )
  }

  return list
})

function toggleTag(tag) {
  activeTag.value = activeTag.value === tag ? '' : tag
}

function clearFilters() {
  activeTag.value = ''
  searchQuery.value = ''
}
</script>

<template>
  <div class="prereq-search-wrapper">
    <!-- 搜索框 -->
    <div class="search-bar">
      <div class="search-input-wrapper">
        <span class="search-icon">🔍</span>
        <input
          v-model="searchQuery"
          type="text"
          placeholder="搜索前置知识（标题或标签）…"
          class="search-input"
        />
        <button v-if="searchQuery" class="search-clear" @click="searchQuery = ''">✕</button>
      </div>
    </div>

    <!-- 标签云 -->
    <div class="prereq-tag-section" v-if="tags.length">
      <div class="prereq-tag-cloud">
        <button
          v-for="tag in mainTags"
          :key="tag.name"
          class="prereq-tag-btn"
          :class="{ active: activeTag === tag.name }"
          @click="toggleTag(tag.name)"
        >
          <span class="prereq-tag-name"># {{ tag.name }}</span>
          <span class="prereq-tag-count">{{ tag.count }}</span>
        </button>
      </div>

      <!-- 折叠的低频标签 -->
      <div v-if="rareTags.length" class="rare-tags-section">
        <button class="rare-tags-toggle" @click="showAllTags = !showAllTags">
          <span>{{ showAllTags ? '收起' : `展开其余 ${rareTags.length} 个标签` }}</span>
          <span class="toggle-arrow" :class="{ expanded: showAllTags }">›</span>
        </button>
        <div class="prereq-tag-cloud rare-tags" v-show="showAllTags">
          <button
            v-for="tag in rareTags"
            :key="tag.name"
            class="prereq-tag-btn"
            :class="{ active: activeTag === tag.name }"
            @click="toggleTag(tag.name)"
          >
            <span class="prereq-tag-name"># {{ tag.name }}</span>
          </button>
        </div>
      </div>
    </div>

    <!-- 结果统计 -->
    <div class="result-bar">
      <span class="result-hint">
        共 <strong>{{ filteredPrereqs.length }}</strong> 篇前置知识
        <span v-if="prereqs.length !== filteredPrereqs.length">
          （总计 {{ prereqs.length }} 篇）
        </span>
      </span>
      <span
        v-if="activeTag || searchQuery"
        class="active-filter"
        @click="clearFilters"
      >
        清除筛选 ✕
      </span>
    </div>

    <!-- 文章列表 -->
    <div class="prereq-list">
      <a
        v-for="item in filteredPrereqs"
        :key="item.link"
        :href="item.link"
        class="prereq-item"
      >
        <div class="prereq-item-title">{{ item.title }}</div>
        <div class="prereq-item-tags" v-if="item.tags.length">
          <span
            v-for="tag in item.tags"
            :key="tag"
            class="prereq-item-tag"
            :class="{ highlighted: tag === activeTag }"
          >
            {{ tag }}
          </span>
        </div>
      </a>
    </div>

    <!-- 空状态 -->
    <div v-if="filteredPrereqs.length === 0" class="empty-state">
      <p>没有匹配的前置知识文章</p>
    </div>
  </div>
</template>

<style scoped>
.prereq-search-wrapper {
  margin-top: 16px;
}

/* 搜索框 */
.search-bar {
  margin-bottom: 20px;
}

.search-input-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  transition: border-color 0.25s, box-shadow 0.25s;
}

.search-input-wrapper:focus-within {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}

.search-icon {
  font-size: 16px;
  opacity: 0.6;
  flex-shrink: 0;
}

.search-input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  font-size: 15px;
  color: var(--vp-c-text-1);
}

.search-input::placeholder {
  color: var(--vp-c-text-3);
}

.search-clear {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  color: var(--vp-c-text-3);
  padding: 2px 6px;
  border-radius: 4px;
}

.search-clear:hover {
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-1);
}

/* 标签云 */
.prereq-tag-section {
  margin-bottom: 20px;
}

.prereq-tag-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.prereq-tag-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.prereq-tag-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.prereq-tag-btn.active {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-1);
  color: #fff;
}

.prereq-tag-count {
  font-size: 11px;
  opacity: 0.7;
  background: var(--vp-c-default-soft);
  padding: 0 5px;
  border-radius: 8px;
}

.prereq-tag-btn.active .prereq-tag-count {
  background: rgba(255, 255, 255, 0.25);
}

/* 低频标签折叠 */
.rare-tags-section {
  margin-top: 10px;
}

.rare-tags-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: none;
  border: none;
  color: var(--vp-c-text-3);
  font-size: 13px;
  cursor: pointer;
  padding: 4px 0;
}

.rare-tags-toggle:hover {
  color: var(--vp-c-brand-1);
}

.toggle-arrow {
  display: inline-block;
  transition: transform 0.2s;
}

.toggle-arrow.expanded {
  transform: rotate(90deg);
}

.rare-tags {
  margin-top: 8px;
}

/* 结果栏 */
.result-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  padding: 8px 0;
  border-bottom: 1px solid var(--vp-c-border);
  font-size: 14px;
  color: var(--vp-c-text-2);
}

.result-hint strong {
  color: var(--vp-c-brand-1);
}

.active-filter {
  margin-left: auto;
  cursor: pointer;
  color: var(--vp-c-text-3);
  font-size: 13px;
  padding: 2px 8px;
  border-radius: 4px;
}

.active-filter:hover {
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-1);
}

/* 文章列表 */
.prereq-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.prereq-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  text-decoration: none;
  color: var(--vp-c-text-1);
  transition: background 0.2s;
}

.prereq-item:hover {
  background: var(--vp-c-bg-soft);
}

.prereq-item-title {
  font-size: 15px;
  font-weight: 500;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prereq-item-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  flex-shrink: 0;
}

.prereq-item-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-3);
  white-space: nowrap;
}

.prereq-item-tag.highlighted {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 48px 24px;
  color: var(--vp-c-text-3);
}

/* 响应式 */
@media (max-width: 640px) {
  .prereq-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
  }

  .prereq-item-title {
    white-space: normal;
  }
}
</style>
