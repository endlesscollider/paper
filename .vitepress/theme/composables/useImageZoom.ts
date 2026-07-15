/**
 * 给文章正文中的 <img> 添加点击放大功能
 *
 * 问题背景：之前用的 medium-zoom 在放大时会被图片的原始分辨率
 * (naturalWidth / naturalHeight) 限制 —— 如果原图本身保存得比较小
 * （比如截图），点开后放大倍数依然有限，看起来还是很小，需要用户自己
 * 再手动放大。
 *
 * 方案：自定义全屏遮罩层，把图片用 object-fit: contain 缩放到
 * 视口的合适大小展示（不受原始分辨率限制，小图也能撑满），并支持
 * 滚轮进一步放大、拖动平移、点击遮罩或 Esc 关闭，交互上与 Mermaid
 * 图的放大保持一致。
 */

let overlayEl: HTMLDivElement | null = null

function ensureOverlay(): HTMLDivElement {
  if (overlayEl) return overlayEl

  const overlay = document.createElement('div')
  overlay.className = 'image-zoom-overlay'

  const stage = document.createElement('div')
  stage.className = 'image-zoom-stage'
  overlay.appendChild(stage)

  const img = document.createElement('img')
  img.className = 'image-zoom-img'
  stage.appendChild(img)

  const closeBtn = document.createElement('button')
  closeBtn.className = 'image-zoom-close'
  closeBtn.title = '关闭 (Esc)'
  closeBtn.textContent = '✕'
  overlay.appendChild(closeBtn)

  const hint = document.createElement('div')
  hint.className = 'image-zoom-hint'
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
    document.body.classList.remove('image-zoom--opened')
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
    scale = Math.min(Math.max(scale + delta, 1), 6)
    applyTransform()
  }, { passive: false })

  stage.addEventListener('mousedown', (e) => {
    if (scale <= 1) return
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
    stage.style.cursor = scale > 1 ? 'grab' : 'default'
  })

  ;(overlay as any).__openWithSrc = (src: string, alt: string) => {
    resetTransform()
    img.src = src
    img.alt = alt || ''
    stage.style.cursor = 'default'
    overlay.classList.add('open')
    document.body.classList.add('image-zoom--opened')
  }

  return overlay
}

export function setupImageZoom() {
  if (typeof window === 'undefined') return

  const init = () => {
    const images = document.querySelectorAll<HTMLImageElement>('.vp-doc img:not(.no-zoom)')
    images.forEach((img) => {
      if (img.getAttribute('data-image-zoom-init')) return
      img.setAttribute('data-image-zoom-init', '1')
      img.addEventListener('click', () => {
        const overlay = ensureOverlay()
        ;(overlay as any).__openWithSrc(img.currentSrc || img.src, img.alt)
      })
    })
  }

  setTimeout(init, 300)

  const observer = new MutationObserver(() => {
    setTimeout(init, 300)
  })
  observer.observe(document.body, { childList: true, subtree: true })
}
