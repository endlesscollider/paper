<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'
import { useRefPanel, allowNextNavigation } from '../composables/useRefPanel'

const { isOpen, isLoading, loadError, currentSection, currentArticleTitle, currentArticleUrl, currentAnchor, closeRefPanel } = useRefPanel()
const { site } = useData()

// 拼出"跳转阅读全文"的真实链接：站点 base + 文章路径 + 锚点
const jumpHref = computed(() => {
  const base = site.value.base || '/'
  const path = `${base}${currentArticleUrl.value}`.replace(/\/+/g, '/')
  return `${path}#${currentAnchor.value}`
})

function handleBackdropClick(e) {
  // 只有点击遮罩本身（不是面板内容）才关闭，避免误触
  if (e.target === e.currentTarget) closeRefPanel()
}

// 「跳转阅读全文」的目标 href 和触发本面板的 ref link 长得一样
// （pathname + #锚点），必须放行这次导航，否则会被再次拦截打开面板。
//
// 关键细节：VitePress 自己的路由点击监听器注册在 window 的 capture 阶段，
// 这个阶段的执行顺序早于任何后代元素上的（无论 capture 还是 bubble）Vue
// @click 处理函数。也就是说，如果在 @click 里才调用 allowNextNavigation()，
// 这时 VitePress 已经在更早的 capture 阶段读取过标记位、做完拦截判断了，
// 设置已经来不及生效。mousedown 是一个独立的、更早发生的事件（先 mousedown
// 才有 click），在这里设置标记位可以确保它在 click 事件真正触发导航判断
//之前就已经生效。
function handleJumpMouseDown() {
  allowNextNavigation()
}

function handleJumpClick() {
  closeRefPanel()
}
</script>

<template>
  <Teleport to="body">
    <div v-if="isOpen" class="ref-panel-backdrop" @click="handleBackdropClick">
      <aside class="ref-panel" role="complementary" aria-label="引用内容预览">
        <header class="ref-panel-header">
          <div class="ref-panel-header-text">
            <div class="ref-panel-article-title">{{ currentArticleTitle || '正在加载…' }}</div>
            <div v-if="currentSection" class="ref-panel-breadcrumb">
              <span v-for="(b, i) in currentSection.breadcrumb" :key="i">{{ b }} <span class="sep">›</span> </span>
              <strong>{{ currentSection.title }}</strong>
            </div>
          </div>
          <button class="ref-panel-close" title="关闭 (Esc)" @click="closeRefPanel">✕</button>
        </header>

        <div class="ref-panel-body">
          <div v-if="isLoading" class="ref-panel-loading">加载中…</div>
          <div v-else-if="loadError" class="ref-panel-error">{{ loadError }}</div>
          <div v-else-if="currentSection" class="ref-panel-content vp-doc" v-html="currentSection.html" />
        </div>

        <footer class="ref-panel-footer">
          <a :href="jumpHref" class="ref-panel-jump-btn" @mousedown="handleJumpMouseDown" @click="handleJumpClick">跳转阅读全文 →</a>
        </footer>
      </aside>
    </div>
  </Teleport>
</template>

<style scoped>
.ref-panel-backdrop {
  position: fixed;
  inset: 0;
  z-index: 150;
  background: transparent;
  pointer-events: none;
}

/* 桌面端：右侧固定分栏面板，不遮挡左侧内容，遮罩本身不拦截点击 */
.ref-panel {
  position: fixed;
  top: var(--vp-nav-height, 64px);
  right: 0;
  bottom: 0;
  width: 42vw;
  min-width: 380px;
  max-width: 640px;
  background: var(--vp-c-bg);
  border-left: 1px solid var(--vp-c-border);
  box-shadow: -8px 0 24px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
  pointer-events: auto;
  animation: ref-panel-slide-in 0.22s ease-out;
}

@keyframes ref-panel-slide-in {
  from { transform: translateX(24px); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

.ref-panel-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--vp-c-border);
  flex-shrink: 0;
}

.ref-panel-article-title {
  font-size: 13px;
  color: var(--vp-c-text-3);
  margin-bottom: 4px;
}

.ref-panel-breadcrumb {
  font-size: 14px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}

.ref-panel-breadcrumb .sep {
  color: var(--vp-c-text-3);
  margin: 0 2px;
}

.ref-panel-close {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  border-radius: 6px;
  border: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  cursor: pointer;
  font-size: 14px;
}

.ref-panel-close:hover {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-brand-1);
}

.ref-panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
}

.ref-panel-loading,
.ref-panel-error {
  color: var(--vp-c-text-2);
  font-size: 14px;
  padding: 24px 0;
}

.ref-panel-error {
  color: var(--vp-c-warning-1, #e5a100);
}

.ref-panel-content {
  padding: 0;
  font-size: 15px;
}

.ref-panel-content :deep(.ref-panel-notice) {
  padding: 12px 16px;
  background: var(--vp-c-bg-soft);
  border: 1px dashed var(--vp-c-border);
  border-radius: 8px;
  color: var(--vp-c-text-2);
  font-size: 14px;
}

.ref-panel-footer {
  flex-shrink: 0;
  padding: 14px 20px;
  border-top: 1px solid var(--vp-c-border);
}

.ref-panel-jump-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: 8px;
  background: var(--vp-c-brand-1);
  color: white;
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
}

.ref-panel-jump-btn:hover {
  background: var(--vp-c-brand-2);
}

/* 小屏（平板/手机）：面板变成从底部升起的全宽抽屉，避免挤压到无法阅读 */
@media (max-width: 960px) {
  .ref-panel-backdrop {
    background: rgba(0, 0, 0, 0.35);
    pointer-events: auto;
  }

  .ref-panel {
    top: auto;
    left: 0;
    right: 0;
    width: 100%;
    max-width: none;
    height: 82vh;
    border-left: none;
    border-top: 1px solid var(--vp-c-border);
    border-radius: 16px 16px 0 0;
    animation: ref-panel-slide-up 0.22s ease-out;
  }

  @keyframes ref-panel-slide-up {
    from { transform: translateY(24px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
}
</style>
