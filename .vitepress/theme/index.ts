import DefaultTheme from 'vitepress/theme'
import TagCloud from './components/TagCloud.vue'
import ArticleList from './components/ArticleList.vue'
import ArticleCard from './components/ArticleCard.vue'
import ReadProgressBar from './components/ReadProgressBar.vue'
import ReadProgressBadge from './components/ReadProgressBadge.vue'
import RecordVisit from './components/RecordVisit.vue'
import FocusModeToggle from './components/FocusModeToggle.vue'
import RefPanel from './components/RefPanel.vue'
import SidebarToggle from './components/SidebarToggle.vue'
import PrereqSearch from './components/PrereqSearch.vue'
import './custom.css'
import { defineAsyncComponent, h, onMounted, onUnmounted } from 'vue'

const PhysicsPlayground = defineAsyncComponent(() => import('./components/PhysicsPlayground.vue'))
const GaussianCovariancePlayground = defineAsyncComponent(() => import('./components/GaussianCovariancePlayground.vue'))
const GaussianCovariance1DPlayground = defineAsyncComponent(() => import('./components/GaussianCovariance1DPlayground.vue'))
const GaussianCovariance2DPlayground = defineAsyncComponent(() => import('./components/GaussianCovariance2DPlayground.vue'))
const GaussianSplatPlayground = defineAsyncComponent(() => import('./components/GaussianSplatPlayground.vue'))
import { setupMathCopy } from './composables/useMathCopy'
import { setupMermaidZoom } from './composables/useMermaidZoom'
import { setupImageZoom } from './composables/useImageZoom'
import { setupRefLinkIntercept } from './composables/useRefLinkIntercept'
import { handleBeforeRouteChange, closeRefPanel, useRefPanel } from './composables/useRefPanel'
import { useFocusHoverReveal } from './composables/useFocusHoverReveal'

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    app.component('TagCloud', TagCloud)
    app.component('ArticleList', ArticleList)
    app.component('ArticleCard', ArticleCard)
    app.component('ReadProgressBadge', ReadProgressBadge)
    app.component('PrereqSearch', PrereqSearch)
    app.component('PhysicsPlayground', PhysicsPlayground)
    app.component('GaussianCovariancePlayground', GaussianCovariancePlayground)
    app.component('GaussianCovariance1DPlayground', GaussianCovariance1DPlayground)
    app.component('GaussianCovariance2DPlayground', GaussianCovariance2DPlayground)
    app.component('GaussianSplatPlayground', GaussianSplatPlayground)

    if (typeof window !== 'undefined') {
      // 路由切换后重新初始化公式复制功能
      router.onAfterRouteChanged = () => {
        setupMathCopy()
      }
      // "分栏引用"功能的核心拦截点：见 useRefPanel.ts 顶部注释
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
      'layout-bottom': () => [h(FocusModeToggle), h(RefPanel), h(SidebarToggle)],
    })
  },
  setup() {
    onMounted(() => {
      setupMathCopy()
      setupImageZoom()
      setupMermaidZoom()
      setupRefLinkIntercept()
      useFocusHoverReveal()

      const { isOpen } = useRefPanel()
      const onKeydown = (e: KeyboardEvent) => {
        if (e.key === 'Escape' && isOpen.value) closeRefPanel()
      }
      window.addEventListener('keydown', onKeydown)
      onUnmounted(() => window.removeEventListener('keydown', onKeydown))
    })
  }
}
