/**
 * 给页面中的 KaTeX 公式添加"点击复制 LaTeX 源码"功能
 * 
 * 问题背景：KaTeX 将公式渲染为大量嵌套 <span>，导致鼠标选中后复制得到的是乱码。
 * 解决方案：
 *   1. 块级公式（$$...$$）：右上角显示复制按钮 + 点击整个公式即可复制
 *   2. 行内公式（$...$）：点击即可复制，鼠标悬停时有提示
 *   3. 复制的内容是原始 LaTeX 源码（从 KaTeX 的 annotation 标签中提取）
 */

export function setupMathCopy() {
  if (typeof window === 'undefined') return

  const init = () => {
    // ===== 块级公式（.katex-display）=====
    const mathBlocks = document.querySelectorAll('.katex-display')
    mathBlocks.forEach((block) => {
      if (block.getAttribute('data-math-copy-init')) return
      block.setAttribute('data-math-copy-init', '1')

      const annotation = block.querySelector('annotation[encoding="application/x-tex"]')
      if (!annotation) return
      const latex = annotation.textContent || ''

      // 复制按钮
      const btn = document.createElement('button')
      btn.className = 'math-copy-btn'
      btn.title = '复制 LaTeX 公式'
      btn.textContent = '📋'
      btn.addEventListener('click', (e) => {
        e.stopPropagation()
        e.preventDefault()
        copyAndFeedback(latex, btn, block as HTMLElement)
      })

      const container = block as HTMLElement
      container.style.position = 'relative'
      container.appendChild(btn)

      // 整个块点击也可以复制
      container.style.cursor = 'pointer'
      container.addEventListener('click', (e) => {
        // 不要拦截复制按钮的点击
        if ((e.target as HTMLElement).closest('.math-copy-btn')) return
        copyAndFeedback(latex, btn, container)
      })

      // 添加 tooltip 提示
      container.setAttribute('title', '点击复制 LaTeX 公式')
    })

    // ===== 行内公式（span.katex，但不在 .katex-display 内）=====
    const inlineMaths = document.querySelectorAll('.katex:not(.katex-display .katex)')
    inlineMaths.forEach((span) => {
      if (span.getAttribute('data-math-copy-init')) return
      span.setAttribute('data-math-copy-init', '1')

      const annotation = span.querySelector('annotation[encoding="application/x-tex"]')
      if (!annotation) return
      const latex = annotation.textContent || ''

      const el = span as HTMLElement
      el.style.cursor = 'pointer'
      el.setAttribute('title', '点击复制公式')
      el.classList.add('math-inline-copyable')

      el.addEventListener('click', (e) => {
        e.preventDefault()
        e.stopPropagation()
        navigator.clipboard.writeText(latex).then(() => {
          // 短暂显示反馈
          el.classList.add('math-copied-flash')
          showToast('已复制: ' + (latex.length > 40 ? latex.slice(0, 40) + '...' : latex))
          setTimeout(() => el.classList.remove('math-copied-flash'), 800)
        }).catch(() => {
          // fallback: 选中文本方式
          fallbackCopy(latex)
        })
      })
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
