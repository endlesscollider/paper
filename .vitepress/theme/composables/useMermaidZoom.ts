/**
 * 给 Mermaid 渲染出的架构图/流程图（SVG）添加点击放大功能
 *
 * 问题背景：Mermaid 图是运行时异步渲染成 <svg>，不是 <img>，
 * 常规的图片放大库（如 medium-zoom）只处理 <img>，对它完全无效。
 *
 * 方案：
 *   1. 监听 DOM 变化，找到渲染完成的 .mermaid > svg
 *   2. 给容器加点击事件，点击后克隆该 SVG 到全屏遮罩层里放大展示
 *   3. 遮罩层内支持滚轮缩放、拖动，点击遮罩或 Esc 关闭
 */

let overlayEl: HTMLDivElement | null = null

function ensureOverlay(): HTMLDivElement {
  if (overlayEl) return overlayEl

  const overlay = document.createElement('div')
  overlay.className = 'mermaid-zoom-overlay'

  const stage = document.createElement('div')
  stage.className = 'mermaid-zoom-stage'
  overlay.appendChild(stage)

  const closeBtn = document.createElement('button')
  closeBtn.className = 'mermaid-zoom-close'
  closeBtn.title = '关闭 (Esc)'
  closeBtn.textContent = '✕'
  overlay.appendChild(closeBtn)

  const hint = document.createElement('div')
  hint.className = 'mermaid-zoom-hint'
  hint.textContent = '滚轮缩放 · 拖动平移 · Esc 关闭'
  overlay.appendChild(hint)

  document.body.appendChild(overlay)
  overlayEl = overlay

  let scale = 1
  let translateX = 0
  let translateY = 0
  let isDragging = false
  let dragStartX = 0
  let dragStartY = 0

  const applyTransform = () => {
    stage.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`
  }

  const resetTransform = () => {
    scale = 1
    translateX = 0
    translateY = 0
    applyTransform()
  }

  const close = () => {
    overlay.classList.remove('open')
    setTimeout(() => {
      stage.innerHTML = ''
    }, 200)
  }

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close()
  })
  closeBtn.addEventListener('click', close)

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.classList.contains('open')) close()
  })

  overlay.addEventListener('wheel', (e) => {
    if (!overlay.classList.contains('open')) return
    e.preventDefault()
    const delta = e.deltaY > 0 ? -0.1 : 0.1
    scale = Math.min(Math.max(scale + delta, 0.5), 6)
    applyTransform()
  }, { passive: false })

  stage.addEventListener('mousedown', (e) => {
    isDragging = true
    dragStartX = e.clientX - translateX
    dragStartY = e.clientY - translateY
    stage.style.cursor = 'grabbing'
  })

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return
    translateX = e.clientX - dragStartX
    translateY = e.clientY - dragStartY
    applyTransform()
  })

  window.addEventListener('mouseup', () => {
    isDragging = false
    stage.style.cursor = 'grab'
  })

  ;(overlay as any).__openWithSvg = (svg: SVGSVGElement) => {
    resetTransform()
    stage.innerHTML = ''
    const clone = svg.cloneNode(true) as SVGSVGElement
    clone.removeAttribute('style')
    clone.style.maxWidth = 'none'
    clone.style.maxHeight = 'none'
    stage.appendChild(clone)
    stage.style.cursor = 'grab'
    overlay.classList.add('open')
  }

  return overlay
}

export function setupMermaidZoom() {
  if (typeof window === 'undefined') return

  const init = () => {
    const containers = document.querySelectorAll<HTMLElement>('.mermaid, .vp-doc svg[id^="mermaid-"]')
    containers.forEach((el) => {
      // 有些版本 mermaid 直接把 svg 挂在 .mermaid 容器里，找到最终的 svg
      const container = el.tagName === 'svg' ? (el.closest('.mermaid') as HTMLElement) || el.parentElement! : el
      if (!container || container.getAttribute('data-mermaid-zoom-init')) return

      const svg = container.querySelector('svg')
      if (!svg) return

      container.setAttribute('data-mermaid-zoom-init', '1')
      container.classList.add('mermaid-zoomable')
      container.title = '点击放大'

      container.addEventListener('click', () => {
        const currentSvg = container.querySelector('svg')
        if (!currentSvg) return
        const overlay = ensureOverlay()
        ;(overlay as any).__openWithSvg(currentSvg)
      })
    })
  }

  setTimeout(init, 500)

  const observer = new MutationObserver(() => {
    setTimeout(init, 300)
  })
  observer.observe(document.body, { childList: true, subtree: true })
}
