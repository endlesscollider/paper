<script setup>
import { ref, onMounted } from 'vue'
import { getAllProgress } from '../composables/useReadProgress'

const props = defineProps({
  title: { type: String, required: true },
  link: { type: String, required: true },
  star: { type: Number, default: 0 },
  category: { type: String, default: '' },
  tags: { type: Array, default: () => [] },
  showProgress: { type: Boolean, default: true },
})

const progressLabel = ref('')
const progressClass = ref('')

onMounted(() => {
  if (!props.showProgress) return

  const allProgress = getAllProgress()
  const candidates = [
    props.link,
    props.link + '.html',
    props.link + '/',
    props.link.replace(/\/$/, ''),
  ]

  let found = 0
  for (const candidate of candidates) {
    if (allProgress[candidate]?.progress) {
      found = Math.max(found, allProgress[candidate].progress)
    }
  }

  if (found > 0 && found < 90) {
    progressLabel.value = `已浏览${found}%`
    progressClass.value = 'progress-reading'
  } else if (found >= 90) {
    progressLabel.value = '已读完'
    progressClass.value = 'progress-done'
  }
})
</script>

<template>
  <a :href="link" class="article-card">
    <!-- 右上角浏览进度 -->
    <span v-if="progressLabel" class="card-progress" :class="progressClass">
      {{ progressLabel }}
    </span>
    <div class="article-card-body">
      <div class="article-title-row">
        <h3 class="article-title">{{ title }}</h3>
        <span v-if="star" class="article-stars" :title="star + ' 星'">
          <span v-for="s in star" :key="s" class="star filled">★</span>
          <span v-for="s in (5 - star)" :key="'e' + s" class="star empty">☆</span>
        </span>
      </div>
      <div class="article-meta">
        <span v-if="category" class="article-category">{{ category }}</span>
        <span v-for="tag in tags" :key="tag" class="article-tag">{{ tag }}</span>
      </div>
    </div>
    <div class="article-arrow">→</div>
  </a>
</template>

<style scoped>
.article-card {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px;
  border-radius: 12px;
  border: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  text-decoration: none !important;
  color: inherit;
}

.article-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06), 0 2px 4px rgba(0, 0, 0, 0.04);
  transform: translateY(-2px);
}

.card-progress {
  position: absolute;
  top: 10px;
  right: 12px;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 8px;
  font-weight: 500;
  line-height: 1.4;
}

.progress-reading {
  background: #fff3cd;
  color: #856404;
}

.progress-done {
  background: #d4edda;
  color: #155724;
}

:global(.dark) .progress-reading {
  background: #4a3f00;
  color: #ffc107;
}

:global(.dark) .progress-done {
  background: #0d3320;
  color: #28a745;
}

.article-card-body {
  flex: 1;
  min-width: 0;
}

.article-title-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 10px;
}

.article-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0;
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  flex: 1;
}

.article-card:hover .article-title {
  color: var(--vp-c-brand-1);
}

.article-stars {
  flex-shrink: 0;
  font-size: 12px;
  letter-spacing: -1px;
  white-space: nowrap;
}

.article-stars .star.filled {
  color: #f5a623;
}

.article-stars .star.empty {
  color: var(--vp-c-text-4, #ddd);
}

.article-meta {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.article-category {
  font-size: 12px;
  padding: 2px 10px;
  border-radius: 10px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

.article-tag {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-3);
}

.article-arrow {
  font-size: 18px;
  color: var(--vp-c-text-3);
  opacity: 0;
  transform: translateX(-4px);
  transition: all 0.25s ease;
  margin-left: 12px;
  flex-shrink: 0;
}

.article-card:hover .article-arrow {
  opacity: 1;
  transform: translateX(0);
  color: var(--vp-c-brand-1);
}

:global(.dark) .article-card {
  background: var(--vp-c-bg-soft);
}

:global(.dark) .article-card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
</style>
