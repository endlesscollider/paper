---
layout: home

hero:
  name: "机器人学习笔记"
  text: "深度学习 · 强化学习 · 机器人控制"
  tagline: 面向工程实践的机器人策略学习知识库 — 从基础概念到前沿论文，系统梳理每一个关键环节
  image:
    src: /robot-brain.svg
    alt: Robot Learning
  actions:
    - theme: brand
      text: 论文阅读
      link: /论文综述/
    - theme: alt
      text: 工程笔记
      link: /工程实践/
    - theme: alt
      text: 🧭 找文章
      link: /tags

features:
  - icon: 🔬
    title: 论文综述 & 精读
    details: 深度 RL、模仿学习、VLA 大模型、Sim-to-Real、扩散策略 — 系统综述 + 逐段精读
    link: /论文综述/
    linkText: 查看全部 →
  - icon: 📚
    title: 前置知识
    details: 策略梯度、DDPM、Flow Matching、Consistency Model… 每篇论文背后的基础概念，一次讲透
    link: /前置知识/
    linkText: 查看全部 →
  - icon: 🔧
    title: 工程实践
    details: ACT Decoder 架构、双臂协调训练、MiGenRL RL 微调实现 — 代码级深度剖析
    link: /工程实践/
    linkText: 查看全部 →
  - icon: 🧠
    title: Transformer → VLA 教程
    details: 从 Attention 手算到 ACT/VLA 机器人策略，零基础友好的完整学习路径
    link: /transformer_vla_tutorial/
    linkText: 进入教程 →
---

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { withBase } from 'vitepress'
import { data as articles } from './.vitepress/theme/articles.data.mts'
import ArticleCard from './.vitepress/theme/components/ArticleCard.vue'

const PAGE_STEP = 12 // 每次"展开更多"增加的卡片数

// 每个板块各自维护"当前显示数量"，默认展示一行 4 张 × 3 行 = 12 张
const visibleCounts = reactive({
  recent: PAGE_STEP,
  top: PAGE_STEP,
  latest: PAGE_STEP,
})

// 最近浏览（本地记录，不受展开数量限制影响排序）
const recentlyViewedRaw = ref([])

onMounted(() => {
  try {
    const raw = localStorage.getItem('recently-viewed-articles')
    if (raw) {
      const items = JSON.parse(raw)
      // 将 recently-viewed 的 item 与 articles 数据匹配，补全 star/category/tags
      // 兼容旧数据：早期版本存的 link 是浏览器 percent-encode 过的中文路径，
      // 和 articles.data.mts 里的明文 Unicode link 对不上，需要 decode 后再匹配，
      // 否则卡片会退化成没有分类/标签/星级的空壳
      recentlyViewedRaw.value = items.map(item => {
        let decodedLink = item.link
        try {
          decodedLink = decodeURIComponent(item.link)
        } catch {}
        const match = articles.find(a => a.link === decodedLink || a.link === item.link)
        return match
          ? { ...match }
          : { title: item.title, link: decodedLink, star: 0, category: '', tags: [] }
      })
    }
  } catch {}
})

const recentlyViewed = computed(() => recentlyViewedRaw.value.slice(0, visibleCounts.recent))

// 按星级排序的推荐文章
const topArticlesAll = computed(() => {
  return [...articles].sort((a, b) => b.star - a.star || a.order - b.order)
})
const topArticles = computed(() => topArticlesAll.value.slice(0, visibleCounts.top))

// 最新文章（按 lastUpdated 降序 = 最近修改的排最前）
const latestArticlesAll = computed(() => {
  return [...articles].sort((a, b) => b.lastUpdated - a.lastUpdated)
})
const latestArticles = computed(() => latestArticlesAll.value.slice(0, visibleCounts.latest))

function expandSection(key, total) {
  visibleCounts[key] = Math.min(visibleCounts[key] + PAGE_STEP, total)
}

// 统计
const tagStats = computed(() => {
  const map = {}
  for (const a of articles) {
    for (const t of a.tags) {
      map[t] = (map[t] || 0) + 1
    }
  }
  return Object.entries(map)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
})
</script>

<div class="home-section">

<!-- 统计概览 -->
<div class="stats-grid">
  <div class="stat-card">
    <span class="stat-number">{{ articles.length }}</span>
    <span class="stat-label">篇文章</span>
  </div>
  <div class="stat-card">
    <span class="stat-number">{{ tagStats.length }}+</span>
    <span class="stat-label">个标签分类</span>
  </div>
  <div class="stat-card">
    <span class="stat-number">4</span>
    <span class="stat-label">大知识板块</span>
  </div>
</div>

<!-- 最近浏览 -->
<div class="section-block" v-if="recentlyViewedRaw.length">
  <div class="section-header">
    <h2>🕐 最近浏览</h2>
    <span class="section-desc">继续上次的阅读</span>
  </div>
  <div class="article-grid compact-grid">
    <ArticleCard
      v-for="item in recentlyViewed"
      :key="item.link"
      :title="item.title"
      :link="item.link"
      :star="item.star"
      :category="item.category"
      :tags="item.tags"
      :series="item.series"
      compact
      :max-tags="3"
    />
  </div>
  <div class="section-footer">
    <button
      v-if="visibleCounts.recent < recentlyViewedRaw.length"
      class="expand-btn"
      @click="expandSection('recent', recentlyViewedRaw.length)"
    >
      展开更多 ↓
    </button>
  </div>
</div>

<!-- 高星推荐 -->
<div class="section-block">
  <div class="section-header">
    <h2>⭐ 高分推荐</h2>
    <span class="section-desc">引用量高、顶会发表、顶级机构出品</span>
  </div>
  <div class="article-grid compact-grid">
    <ArticleCard
      v-for="article in topArticles"
      :key="article.link"
      :title="article.title"
      :link="article.link"
      :star="article.star"
      :category="article.category"
      :tags="article.tags"
      :series="article.series"
      compact
      :max-tags="3"
    />
  </div>
  <div class="section-footer">
    <button
      v-if="visibleCounts.top < topArticlesAll.length"
      class="expand-btn"
      @click="expandSection('top', topArticlesAll.length)"
    >
      展开更多 ↓
    </button>
    <a class="view-all-btn" :href="withBase('/tags?sort=star')">查看全部 →</a>
  </div>
</div>

<!-- 最新文章 -->
<div class="section-block">
  <div class="section-header">
    <h2>🆕 最新收录</h2>
    <span class="section-desc">新鲜出炉的论文和笔记</span>
  </div>
  <div class="article-grid compact-grid">
    <ArticleCard
      v-for="article in latestArticles"
      :key="article.link"
      :title="article.title"
      :link="article.link"
      :star="article.star"
      :category="article.category"
      :tags="article.tags"
      :series="article.series"
      compact
      :max-tags="3"
    />
  </div>
  <div class="section-footer">
    <button
      v-if="visibleCounts.latest < latestArticlesAll.length"
      class="expand-btn"
      @click="expandSection('latest', latestArticlesAll.length)"
    >
      展开更多 ↓
    </button>
    <a class="view-all-btn" :href="withBase('/tags?sort=new')">查看全部 →</a>
  </div>
</div>

<!-- 热门标签 -->
<div class="section-block">
  <div class="section-header">
    <h2>🏷️ 热门标签</h2>
  </div>
  <div class="tag-cloud-home">
    <a v-for="[tag, count] in tagStats" :key="tag" :href="withBase('/tags?tag=' + encodeURIComponent(tag))" class="tag-btn-home">
      # {{ tag }} <span class="tag-count-home">{{ count }}</span>
    </a>
  </div>
</div>

<div class="view-all-section">
  <a :href="withBase('/tags')" class="view-all-link">🧭 找文章：按主题浏览全部 →</a>
</div>

</div>

<style>
.home-section {
  max-width: 1152px;
  margin: 0 auto;
  padding: 32px 24px 64px;
}

/* 统计卡片 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 48px;
}

.stat-card {
  text-align: center;
  padding: 28px 16px;
  border-radius: 16px;
  background: linear-gradient(135deg, var(--vp-c-bg-soft), var(--vp-c-bg));
  border: 1px solid var(--vp-c-border);
  transition: transform 0.2s, box-shadow 0.2s;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.stat-number {
  display: block;
  font-size: 36px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--vp-c-brand-1), var(--vp-c-brand-2, #6366f1));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.stat-label {
  display: block;
  margin-top: 4px;
  font-size: 14px;
  color: var(--vp-c-text-2);
}

/* 板块通用 */
.section-block {
  margin-bottom: 48px;
}

.section-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--vp-c-border);
}

.section-header h2 {
  font-size: 20px;
  font-weight: 600;
  margin: 0;
  border: none;
  padding: 0;
}

.section-desc {
  font-size: 13px;
  color: var(--vp-c-text-3);
}

/* 文章卡片网格 (首页布局) */
.article-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}

@media (min-width: 640px) {
  .article-grid {
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  }
}

/* 首页板块用的紧凑网格：一行最多 4 张卡片，卡片更小更浓缩 */
.compact-grid {
  gap: 10px;
}

@media (min-width: 480px) {
  .compact-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 900px) {
  .compact-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1152px) {
  .compact-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

/* 板块底部：展开更多 / 查看全部 */
.section-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-top: 16px;
}

.expand-btn {
  padding: 8px 20px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  transition: all 0.2s;
}

.expand-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.view-all-btn {
  padding: 8px 20px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 500;
  text-decoration: none;
  color: var(--vp-c-brand-1);
  border: 1px solid transparent;
  transition: all 0.2s;
}

.view-all-btn:hover {
  background: var(--vp-c-brand-soft);
}

/* 热门标签 */
.tag-cloud-home {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.tag-btn-home {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 14px;
  text-decoration: none;
  border: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  transition: all 0.2s ease;
  font-weight: 500;
}

.tag-btn-home:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  transform: translateY(-1px);
}

.tag-count-home {
  font-size: 11px;
  opacity: 0.6;
  margin-left: 2px;
}

/* 底部按钮 */
.view-all-section {
  text-align: center;
  padding-top: 16px;
}

.view-all-link {
  display: inline-block;
  padding: 12px 32px;
  border-radius: 24px;
  background: linear-gradient(135deg, var(--vp-c-brand-1), var(--vp-c-brand-2, #6366f1));
  color: white !important;
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(var(--vp-c-brand-1-rgb, 100, 108, 255), 0.3);
}

.view-all-link:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(var(--vp-c-brand-1-rgb, 100, 108, 255), 0.4);
}

/* 响应式 */
@media (max-width: 640px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
  .section-header {
    flex-direction: column;
    gap: 4px;
  }
}

/* 暗色模式 */
.dark .stat-card {
  background: linear-gradient(135deg, var(--vp-c-bg-soft), var(--vp-c-bg));
}
</style>
