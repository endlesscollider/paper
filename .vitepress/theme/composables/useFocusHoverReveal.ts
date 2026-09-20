import { watch } from 'vue'
import { useFocusMode } from './useFocusMode'

// 全屏阅读模式下，顶栏 / 左侧栏 / 右侧栏默认隐藏，鼠标移到屏幕边缘的
// "感应条"或面板本身时才滑出，移开后延迟隐藏（避免鼠标略微抖动就闪烁）。

const HIDE_DELAY = 250

type ZoneKey = 'top' | 'left' | 'right'

const zoneConfig: Record<ZoneKey, { revealClass: string; hostSelector: string }> = {
  top: { revealClass: 'focus-reveal-top', hostSelector: '.VPNav' },
  left: { revealClass: 'focus-reveal-left', hostSelector: '.VPSidebar' },
  right: { revealClass: 'focus-reveal-right', hostSelector: '.VPDoc .aside' },
}

const ZONE_KEYS: ZoneKey[] = ['top', 'left', 'right']

let composableInitialized = false
let active = false

const hideTimers: Partial<Record<ZoneKey, number>> = {}
const zoneEls: Partial<Record<ZoneKey, HTMLDivElement>> = {}
let onMouseOver: ((e: MouseEvent) => void) | null = null
let onMouseOut: ((e: MouseEvent) => void) | null = null

function reveal(key: ZoneKey) {
  document.documentElement.classList.add(zoneConfig[key].revealClass)
  const timer = hideTimers[key]
  if (timer) {
    window.clearTimeout(timer)
    hideTimers[key] = undefined
  }
}

function scheduleHide(key: ZoneKey) {
  const existing = hideTimers[key]
  if (existing) window.clearTimeout(existing)
  hideTimers[key] = window.setTimeout(() => {
    document.documentElement.classList.remove(zoneConfig[key].revealClass)
    hideTimers[key] = undefined
  }, HIDE_DELAY)
}

function createZoneElement(key: ZoneKey): HTMLDivElement {
  const el = document.createElement('div')
  el.className = `focus-hover-zone focus-hover-zone-${key}`
  el.setAttribute('aria-hidden', 'true')
  document.body.appendChild(el)
  el.addEventListener('mouseenter', () => reveal(key))
  el.addEventListener('mouseleave', () => scheduleHide(key))
  return el
}

function setup() {
  if (active || typeof document === 'undefined') return
  active = true

  ZONE_KEYS.forEach((key) => {
    zoneEls[key] = createZoneElement(key)
  })

  // 面板本身（顶栏 / 左侧栏 / 右侧栏）用事件委托监听，
  // 因为 SPA 路由切换会重建这些元素，委托到 document 上不需要重新绑定。
  onMouseOver = (e: MouseEvent) => {
    const target = e.target as HTMLElement | null
    if (!target) return
    ZONE_KEYS.forEach((key) => {
      if (target.closest(zoneConfig[key].hostSelector)) {
        reveal(key)
      }
    })
  }
  onMouseOut = (e: MouseEvent) => {
    const target = e.target as HTMLElement | null
    if (!target) return
    ZONE_KEYS.forEach((key) => {
      if (target.closest(zoneConfig[key].hostSelector)) {
        scheduleHide(key)
      }
    })
  }
  document.addEventListener('mouseover', onMouseOver)
  document.addEventListener('mouseout', onMouseOut)
}

function teardown() {
  if (!active) return
  active = false

  if (onMouseOver) document.removeEventListener('mouseover', onMouseOver)
  if (onMouseOut) document.removeEventListener('mouseout', onMouseOut)
  onMouseOver = null
  onMouseOut = null

  ZONE_KEYS.forEach((key) => {
    zoneEls[key]?.remove()
    zoneEls[key] = undefined
    document.documentElement.classList.remove(zoneConfig[key].revealClass)
    const timer = hideTimers[key]
    if (timer) {
      window.clearTimeout(timer)
      hideTimers[key] = undefined
    }
  })
}

export function useFocusHoverReveal() {
  if (composableInitialized || typeof window === 'undefined') return
  composableInitialized = true

  const { isFocusMode } = useFocusMode()
  watch(
    isFocusMode,
    (isActive) => {
      if (isActive) setup()
      else teardown()
    },
    { immediate: true }
  )
}
