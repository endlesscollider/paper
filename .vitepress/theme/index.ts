import DefaultTheme from 'vitepress/theme'
import TagCloud from './components/TagCloud.vue'
import ArticleList from './components/ArticleList.vue'
import ArticleCard from './components/ArticleCard.vue'
import ReadProgressBar from './components/ReadProgressBar.vue'
import ReadProgressBadge from './components/ReadProgressBadge.vue'
import RecordVisit from './components/RecordVisit.vue'
import FocusModeToggle from './components/FocusModeToggle.vue'
import './custom.css'
import { h, onMounted } from 'vue'
import { setupMathCopy } from './composables/useMathCopy'
import { setupMermaidZoom } from './composables/useMermaidZoom'
import { setupImageZoom } from './composables/useImageZoom'

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    app.component('TagCloud', TagCloud)
    app.component('ArticleList', ArticleList)
    app.component('ArticleCard', ArticleCard)
    app.component('ReadProgressBadge', ReadProgressBadge)

    // 路由切换后重新初始化公式复制功能
    if (typeof window !== 'undefined') {
      router.onAfterRouteChanged = () => {
        setupMathCopy()
      }
    }
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'doc-before': () => [h(ReadProgressBar), h(RecordVisit)],
      'layout-bottom': () => h(FocusModeToggle),
    })
  },
  setup() {
    onMounted(() => {
      setupMathCopy()
      setupImageZoom()
      setupMermaidZoom()
    })
  }
}
