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
  hint.innerHTML = '<span>滚轮缩放 · 拖动平移 · Esc 关闭 · 双击重置</span>'
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

  const calculateInitialScale = (svgWidth: number, svgHeight: number): number => {
    // 计算合适的初始缩放比例，让图表在视口中占据合适的空间
    const viewportWidth = window.innerWidth * 0.9  // 留出一些边距
    const viewportHeight = window.innerHeight * 0.9
    
    // 计算水平方向和垂直方向的比例
    const widthRatio = viewportWidth / svgWidth
    const heightRatio = viewportHeight / svgHeight
    
    // 取最小值，确保图表完全在视口中
    const fitScale = Math.min(widthRatio, heightRatio)
    
    // 限制最小和最大缩放比例
    return Math.max(0.8, Math.min(fitScale, 2))
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

  stage.addEventListener('dblclick', () => {
    // 双击重置缩放和位置
    scale = calculateInitialScale(
      stage.clientWidth,
      stage.clientHeight
    )
    translateX = 0
    translateY = 0
    applyTransform()
  })

  ;(overlay as any).__openWithSvg = (svg: SVGSVGElement) => {
    resetTransform()
    stage.innerHTML = ''
    const clone = svg.cloneNode(true) as SVGSVGElement
    clone.removeAttribute('style')
    clone.style.maxWidth = 'none'
    clone.style.maxHeight = 'none'
    stage.appendChild(clone)
    
    // 等待一帧让DOM更新，然后计算SVG的实际尺寸
    requestAnimationFrame(() => {
      // 尝试获取SVG的实际渲染尺寸
      let actualWidth = clone.clientWidth || clone.getBoundingClientRect().width
      let actualHeight = clone.clientHeight || clone.getBoundingClientRect().height
      
      // 如果无法获取渲染尺寸，尝试从属性获取
      if (actualWidth === 0 || actualHeight === 0) {
        const attrWidth = parseInt(clone.getAttribute('width') || '0')
        const attrHeight = parseInt(clone.getAttribute('height') || '0')
        if (attrWidth > 0 && attrHeight > 0) {
          actualWidth = attrWidth
          actualHeight = attrHeight
        }
      }
      
      // 如果还是没有尺寸，尝试从viewBox获取
      if ((actualWidth === 0 || actualHeight === 0) && clone.viewBox.baseVal) {
        actualWidth = clone.viewBox.baseVal.width
        actualHeight = clone.viewBox.baseVal.height
      }
      
      // 如果还是无法获取尺寸，使用默认值
      if (actualWidth === 0 || actualHeight === 0) {
        actualWidth = 800
        actualHeight = 600
      }
      
      // 设置初始缩放比例
      scale = calculateInitialScale(actualWidth, actualHeight)
      applyTransform()
      
      stage.style.cursor = 'grab'
      overlay.classList.add('open')
    })
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
