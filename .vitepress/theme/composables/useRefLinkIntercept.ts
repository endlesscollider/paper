/**
 * 给"引用另一篇文章某个标题"的链接打上视觉标记（追加一个小图标 class），
 * 让读者能一眼看出"点这个链接不会离开当前页面，而是弹出预览"。
 *
 * 注意：真正的点击拦截逻辑不在这里，而是挂在 VitePress 路由器的官方扩展点
 * `router.onBeforeRouteChange` 上（见 .vitepress/theme/index.ts 里的
 * enhanceApp，以及 useRefPanel.ts 里的 handleBeforeRouteChange）。这里
 * 只负责扫描 DOM、加样式，不做事件绑定，避免和 VitePress 自己的路由点击
 * 监听器抢事件。
 */

const MARK_ATTR = 'data-ref-link-init'

function isRefLink(a: HTMLAnchorElement): boolean {
  if (a.classList.contains('header-anchor')) return false // 标题旁的 "#" 永久链接图标，跳过
  if (a.target && a.target !== '') return false
  if (a.hasAttribute('download')) return false

  let url: URL
  try {
    url = new URL(a.href, window.location.href)
  } catch {
    return false
  }
  if (url.origin !== window.location.origin) return false
  if (!url.hash) return false

  const normalize = (p: string) => p.replace(/\/index\.html$/, '/').replace(/\.html$/, '')
  if (normalize(url.pathname) === normalize(window.location.pathname)) return false

  return true
}

export function setupRefLinkIntercept() {
  if (typeof window === 'undefined') return

  const init = () => {
    const links = document.querySelectorAll<HTMLAnchorElement>('.vp-doc a[href]')
    links.forEach((a) => {
      if (a.hasAttribute(MARK_ATTR)) return
      a.setAttribute(MARK_ATTR, '1')
      if (isRefLink(a)) a.classList.add('ref-link')
    })
  }

  setTimeout(init, 300)

  const observer = new MutationObserver(() => {
    setTimeout(init, 300)
  })
  observer.observe(document.body, { childList: true, subtree: true })
}
