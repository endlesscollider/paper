/**
 * "分栏引用"功能：点击文章里引用另一篇文章某个标题的链接时，不整页跳转，
 * 而是把页面切成左右两栏——左边是当前文章（正常滚动），右边弹出一个面板，
 * 展示被引用那一节的渲染内容。用户可以直接在右侧读完，也可以点"跳转阅读
 * 全文"做真正的路由跳转，或者关闭面板恢复原来的单栏排版。
 *
 * 拦截点：VitePress 的路由器提供了官方扩展点 `router.onBeforeRouteChange`
 * ——每次点击站内链接触发页面切换前都会调用它，返回 false 可以取消这次
 * 导航。这比自己在 DOM 上加 click 监听器再抢跑 preventDefault 稳妥得多，
 * 因为 VitePress 自己的路由点击监听器是在 window 上以捕获阶段注册的，
 * 任何后加的捕获/冒泡监听器都跑在它之后，抢不到先手。
 *
 * 识别规则：只要目标 href 带有 "#锚点" 且指向的文章和当前文章不是同一篇，
 * 就认为这是一次"分节引用"点击。已有文章里大量形如
 *   [回顾 Q-Chunking 精读 第 4.2 节](/论文综述/071_xxx#_4-2-xxx)
 * 的引用天然满足这个条件，不需要改任何写法。
 *
 * 数据来源：每篇文章的"按标题分节 JSON"由 .vitepress/refSectionsPlugin.mts
 * 在 dev/build 阶段生成，前端只需要 fetch 对应文件名即可，文件名计算规则
 * 见 .vitepress/shared/refLink.mts（前后端共用同一份逻辑，保证对得上）。
 */
import { ref, shallowRef } from 'vue'
import { REF_SECTIONS_DIR, articleKeyFromHref, keyToSectionsFilename } from '../../shared/refLink.mts'

export interface RefSection {
  id: string
  level: number
  title: string
  breadcrumb: string[]
  html: string
}

export interface RefArticlePayload {
  articleUrl: string
  articleTitle: string
  sections: RefSection[]
}

const isOpen = ref(false)
const isLoading = ref(false)
const loadError = ref<string | null>(null)
const currentSection = shallowRef<RefSection | null>(null)
const currentArticleTitle = ref('')
const currentArticleUrl = ref('')
const currentAnchor = ref('')

// “跳转阅读全文”按钮点击后，导航目标 href 和 ref link 长得一模一样
// （都是 pathname + #锚点），必须靠这个一次性标记位跳过拦截，否则点了
// 跳转按钮又会立刻重新打开面板，形成死循环。
let bypassNextIntercept = false

/** 供 RefPanel.vue 的"跳转阅读全文"按钮调用：放行下一次导航 */
export function allowNextNavigation() {
  bypassNextIntercept = true
}

// 同一份 JSON 在一次会话里只需要拉取一次
const cache = new Map<string, Promise<RefArticlePayload>>()

function applyPanelClass(open: boolean) {
  if (typeof document === 'undefined') return
  document.documentElement.classList.toggle('ref-panel-open', open)
}

function siteBase(): string {
  return (import.meta as any).env?.BASE_URL || '/'
}

async function fetchArticleSections(filename: string): Promise<RefArticlePayload> {
  if (cache.has(filename)) return cache.get(filename)!
  const base = siteBase()
  const url = `${base}${REF_SECTIONS_DIR}/${filename}`.replace(/([^:])\/\/+/g, '$1/')
  const promise = fetch(url).then((res) => {
    if (!res.ok) throw new Error(`分节索引加载失败 (${res.status})`)
    return res.json()
  })
  cache.set(filename, promise)
  return promise
}

/**
 * 根据点击的链接 pathname + hash 打开分栏预览面板。
 * @param pathname 不含 base 的 pathname，如 "/论文综述/072_x.html"
 * @param hash 含 "#" 的锚点，如 "#_4-2-xxx"
 */
export async function openRefPanel(pathname: string, hash: string) {
  const anchor = decodeURIComponent(hash.replace(/^#/, ''))
  if (!anchor) return

  isOpen.value = true
  applyPanelClass(true)
  isLoading.value = true
  loadError.value = null
  currentSection.value = null
  currentAnchor.value = anchor

  const key = articleKeyFromHref(pathname)
  const filename = keyToSectionsFilename(key)
  currentArticleUrl.value = '/' + key

  try {
    const payload = await fetchArticleSections(filename)
    currentArticleTitle.value = payload.articleTitle
    const section = payload.sections.find((s) => s.id === anchor)
    if (!section) {
      loadError.value = `没有找到对应小节（锚点：${anchor}），可能文章已更新，请点击下方按钮跳转查看完整内容。`
    } else {
      currentSection.value = section
    }
  } catch (err: any) {
    loadError.value = err?.message || '加载失败'
  } finally {
    isLoading.value = false
  }
}

export function closeRefPanel() {
  isOpen.value = false
  applyPanelClass(false)
  currentSection.value = null
  loadError.value = null
}

/**
 * 挂在 VitePress router.onBeforeRouteChange 上的判定函数。
 * 返回 false 表示"拦截这次导航，我自己处理了"；返回/不返回值(undefined)
 * 表示"放行，交给 VitePress 正常跳转"。
 */
export function handleBeforeRouteChange(to: string): boolean | void {
  if (bypassNextIntercept) {
    bypassNextIntercept = false
    return
  }
  if (typeof window === 'undefined') return

  let url: URL
  try {
    url = new URL(to, window.location.href)
  } catch {
    return
  }
  if (!url.hash) return // 没有锚点，走正常整页跳转（如"下一篇"之类的链接）

  const normalize = (p: string) => p.replace(/\/index\.html$/, '/').replace(/\.html$/, '')
  if (normalize(url.pathname) === normalize(window.location.pathname)) return // 本页内锚点跳转，交给默认行为

  openRefPanel(url.pathname, url.hash)
  return false
}

export function useRefPanel() {
  return {
    isOpen,
    isLoading,
    loadError,
    currentSection,
    currentArticleTitle,
    currentArticleUrl,
    currentAnchor,
    openRefPanel,
    closeRefPanel
  }
}
