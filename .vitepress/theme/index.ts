import DefaultTheme from 'vitepress/theme'
import TagCloud from './components/TagCloud.vue'
import ArticleList from './components/ArticleList.vue'
import ArticleCard from './components/ArticleCard.vue'
import ReadProgressBar from './components/ReadProgressBar.vue'
import ReadProgressBadge from './components/ReadProgressBadge.vue'
import RecordVisit from './components/RecordVisit.vue'
import './custom.css'
import { h } from 'vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('TagCloud', TagCloud)
    app.component('ArticleList', ArticleList)
    app.component('ArticleCard', ArticleCard)
    app.component('ReadProgressBadge', ReadProgressBadge)
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'doc-before': () => [h(ReadProgressBar), h(RecordVisit)],
    })
  }
}
