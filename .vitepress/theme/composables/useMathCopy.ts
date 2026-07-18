/**
 * 给页面中的数学公式添加"点击复制 LaTeX 源码"功能
 * 
 * 问题背景：本站用 MathJax（markdown-it-mathjax3）渲染公式，公式最终变成纯 SVG
 * 路径（<mjx-container><svg>...</svg></mjx-container>），渲染结果里没有任何
 * 可选中的文本节点——鼠标拖选公式选不中任何字符，是 SVG 渲染的必然结果，不是
 * bug，但也没法通过"允许原生选中"来解决。
 * 解决方案：
 *   1. 在 .vitepress/config.mts 里，构建时把每个公式的原始 LaTeX 源码写入
 *      <mjx-container> 的 data-tex 属性
 *   2. 块级公式（display="true"）：右上角显示复制按钮 + 点击整个公式即可复制
 *   3. 行内公式：点击即可复制，鼠标悬停时有提示
 *   4. 复制的内容就是 data-tex 里保存的原始 LaTeX 源码
 */

export function setupMathCopy() {
  if (typeof window === 'undefined') return

  const init = () => {
    const containers = document.querySelectorAll('mjx-container[data-tex]')
    containers.forEach((el) => {
      if (el.getAttribute('data-math-copy-init')) return
      el.setAttribute('data-math-copy-init', '1')

      const latex = el.getAttribute('data-tex') || ''
      if (!latex) return

      const container = el as HTMLElement
      const isBlock = container.getAttribute('display') === 'true'

      if (isBlock) {
        // ===== 块级公式：右上角复制按钮 + 整块可点击 =====
        container.classList.add('math-block-copyable')
        container.style.position = 'relative'

        const btn = document.createElement('button')
        btn.className = 'math-copy-btn'
        btn.title = '复制 LaTeX 公式'
        btn.textContent = '📋'
        btn.addEventListener('click', (e) => {
          e.stopPropagation()
          e.preventDefault()
          copyAndFeedback(latex, btn, container)
        })
        container.appendChild(btn)

        container.style.cursor = 'pointer'
        container.addEventListener('click', (e) => {
          if ((e.target as HTMLElement).closest('.math-copy-btn')) return
          copyAndFeedback(latex, btn, container)
        })
        container.setAttribute('title', '点击复制 LaTeX 公式')
      } else {
        // ===== 行内公式：点击整体即可复制 =====
        container.classList.add('math-inline-copyable')
        container.style.cursor = 'pointer'
        container.setAttribute('title', '点击复制公式')

        container.addEventListener('click', (e) => {
          e.preventDefault()
          e.stopPropagation()
          navigator.clipboard.writeText(latex).then(() => {
            container.classList.add('math-copied-flash')
            showToast('已复制: ' + (latex.length > 40 ? latex.slice(0, 40) + '...' : latex))
            setTimeout(() => container.classList.remove('math-copied-flash'), 800)
          }).catch(() => {
            fallbackCopy(latex)
          })
        })
      }
    })
  }

  function copyAndFeedback(latex: string, btn: HTMLElement, container: HTMLElement) {
    navigator.clipboard.writeText(latex).then(() => {
      btn.textContent = '✓'
      btn.classList.add('copied')
      container.classList.add('math-block-copied')
      showToast('已复制 LaTeX 公式')
      setTimeout(() => {
        btn.textContent = '📋'
        btn.classList.remove('copied')
        container.classList.remove('math-block-copied')
      }, 2000)
    }).catch(() => {
      fallbackCopy(latex)
    })
  }

  function fallbackCopy(text: string) {
    // clipboard API 不可用时的 fallback
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    document.body.removeChild(textarea)
    showToast('已复制 LaTeX 公式')
  }

  function showToast(message: string) {
    // 移除旧 toast
    const old = document.querySelector('.math-copy-toast')
    if (old) old.remove()

    const toast = document.createElement('div')
    toast.className = 'math-copy-toast'
    toast.textContent = message
    document.body.appendChild(toast)

    // 触发动画
    requestAnimationFrame(() => toast.classList.add('show'))
    setTimeout(() => {
      toast.classList.remove('show')
      setTimeout(() => toast.remove(), 300)
    }, 2000)
  }

  // 页面加载完后初始化
  setTimeout(init, 500)

  // 监听 DOM 变化（VitePress SPA 路由切换时内容动态替换）
  const observer = new MutationObserver(() => {
    setTimeout(init, 300)
  })
  observer.observe(document.body, { childList: true, subtree: true })
}
