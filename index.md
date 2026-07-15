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
      text: 按标签浏览
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
import { ref, computed, onMounted } from 'vue'
import { withBase } from 'vitepress'
import { data as articles } from './.vitepress/theme/articles.data.mts'
import ArticleCard from './.vitepress/theme/components/ArticleCard.vue'

// 按星级排序的推荐文章
const topArticles = computed(() => {
  return [...articles]
    .sort((a, b) => b.star - a.star || a.order - b.order)
    .slice(0, 6)
})

// 最新文章（按 order 降序 = 编号越大越新）
const latestArticles = computed(() => {
  return [...articles]
    .sort((a, b) => b.order - a.order)
    .slice(0, 6)
})

// 最近浏览
const recentlyViewed = ref([])

onMounted(() => {
  try {
    const raw = localStorage.getItem('recently-viewed-articles')
    if (raw) {
      const items = JSON.parse(raw).slice(0, 6)
      // 将 recently-viewed 的 item 与 articles 数据匹配，补全 star/category/tags
      recentlyViewed.value = items.map(item => {
        const match = articles.find(a => a.link === item.link)
        return match
          ? { ...match }
          : { title: item.title, link: item.link, star: 0, category: '', tags: [] }
      })
    }
  } catch {}
})

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
<div class="section-block" v-if="recentlyViewed.length">
  <div class="section-header">
    <h2>🕐 最近浏览</h2>
    <span class="section-desc">继续上次的阅读</span>
  </div>
  <div class="article-grid">
    <ArticleCard
      v-for="item in recentlyViewed"
      :key="item.link"
      :title="item.title"
      :link="item.link"
      :star="item.star"
      :category="item.category"
      :tags="item.tags"
      :series="item.series"
    />
  </div>
</div>

<!-- 高星推荐 -->
<div class="section-block">
  <div class="section-header">
    <h2>⭐ 高分推荐</h2>
    <span class="section-desc">引用量高、顶会发表、顶级机构出品</span>
  </div>
  <div class="article-grid">
    <ArticleCard
      v-for="article in topArticles"
      :key="article.link"
      :title="article.title"
      :link="article.link"
      :star="article.star"
      :category="article.category"
      :tags="article.tags"
      :series="article.series"
    />
  </div>
</div>

<!-- 最新文章 -->
<div class="section-block">
  <div class="section-header">
    <h2>🆕 最新收录</h2>
    <span class="section-desc">新鲜出炉的论文和笔记</span>
  </div>
  <div class="article-grid">
    <ArticleCard
      v-for="article in latestArticles"
      :key="article.link"
      :title="article.title"
      :link="article.link"
      :star="article.star"
      :category="article.category"
      :tags="article.tags"
      :series="article.series"
    />
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
  <a :href="withBase('/tags')" class="view-all-link">查看全部文章 →</a>
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
