import DefaultTheme from 'vitepress/theme'
import TagCloud from './components/TagCloud.vue'
import ArticleList from './components/ArticleList.vue'
import ArticleCard from './components/ArticleCard.vue'
import ReadProgressBar from './components/ReadProgressBar.vue'
import ReadProgressBadge from './components/ReadProgressBadge.vue'
import RecordVisit from './components/RecordVisit.vue'
import FocusModeToggle from './components/FocusModeToggle.vue'
import RefPanel from './components/RefPanel.vue'
import './custom.css'
import { h, onMounted, onUnmounted } from 'vue'
import { setupMathCopy } from './composables/useMathCopy'
import { setupMermaidZoom } from './composables/useMermaidZoom'
import { setupImageZoom } from './composables/useImageZoom'
import { setupRefLinkIntercept } from './composables/useRefLinkIntercept'
import { handleBeforeRouteChange, closeRefPanel, useRefPanel } from './composables/useRefPanel'

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    app.component('TagCloud', TagCloud)
    app.component('ArticleList', ArticleList)
    app.component('ArticleCard', ArticleCard)
    app.component('ReadProgressBadge', ReadProgressBadge)

    if (typeof window !== 'undefined') {
      // 路由切换后重新初始化公式复制功能
      router.onAfterRouteChanged = () => {
        setupMathCopy()
      }
      // “分栏引用”功能的核心拦截点：见 useRefPanel.ts 顶部注释
      const prevBefore = router.onBeforeRouteChange
      router.onBeforeRouteChange = async (to: string) => {
        const result = handleBeforeRouteChange(to)
        if (result === false) return false
        return prevBefore ? prevBefore(to) : undefined
      }
    }
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'doc-before': () => [h(ReadProgressBar), h(RecordVisit)],
      'layout-bottom': () => [h(FocusModeToggle), h(RefPanel)],
    })
  },
  setup() {
    onMounted(() => {
      setupMathCopy()
      setupImageZoom()
      setupMermaidZoom()
      setupRefLinkIntercept()

      const { isOpen } = useRefPanel()
      const onKeydown = (e: KeyboardEvent) => {
        if (e.key === 'Escape' && isOpen.value) closeRefPanel()
      }
      window.addEventListener('keydown', onKeydown)
      onUnmounted(() => window.removeEventListener('keydown', onKeydown))
    })
  }
}
