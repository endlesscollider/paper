<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useData, useRoute } from 'vitepress'

const { frontmatter } = useData()
const route = useRoute()

const isDocPage = computed(() => {
  const layout = frontmatter.value?.layout
  return !layout || layout === 'doc'
})

const leftCollapsed = ref(false)
const rightCollapsed = ref(false)

// ===== 常量 =====
const MIN_WIDTH = 220
const MAX_WIDTH = 600
const DEFAULT_WIDTH = 340
const COLLAPSE_THRESHOLD = 180 // 拖到这个值以下就收起

// ===== 宽度管理 =====
let lastWidth = DEFAULT_WIDTH // 记住收起前的宽度

function getSavedWidth(): number {
  const saved = localStorage.getItem('sidebar-width')
  if (saved) {
    const n = parseInt(saved, 10)
    if (n >= MIN_WIDTH && n <= MAX_WIDTH) return n
  }
  return DEFAULT_WIDTH
}

function applySidebarWidth(width: number) {
  document.documentElement.style.setProperty('--vp-sidebar-width', `${width}px`)
}

// ===== 左侧栏收起/展开 =====
function collapseLeft() {
  leftCollapsed.value = true
  document.documentElement.classList.add('sidebar-left-collapsed')
  localStorage.setItem('sidebar-left-collapsed', 'true')
}

function expandLeft() {
  leftCollapsed.value = false
  document.documentElement.classList.remove('sidebar-left-collapsed')
  localStorage.setItem('sidebar-left-collapsed', 'false')
  // 恢复宽度
  applySidebarWidth(lastWidth)
}

function toggleRightAndSave() {
  rightCollapsed.value = !rightCollapsed.value
  document.documentElement.classList.toggle('sidebar-right-collapsed', rightCollapsed.value)
  localStorage.setItem('sidebar-right-collapsed', String(rightCollapsed.value))
}

// ===== 拖拽逻辑 =====
let isDragging = false
let startX = 0
let startWidth = 0

function injectResizeHandle() {
  const sidebar = document.querySelector('.VPSidebar') as HTMLElement | null
  if (!sidebar) return
  if (sidebar.querySelector('.sidebar-resize-handle')) return

  const handle = document.createElement('div')
  handle.className = 'sidebar-resize-handle'
  handle.title = '拖拽调整宽度 · 双击恢复默认'
  sidebar.appendChild(handle)

  handle.addEventListener('mousedown', onDragStart)
  handle.addEventListener('touchstart', onTouchStart, { passive: false })
  handle.addEventListener('dblclick', onDoubleClick)
}

function onDragStart(e: MouseEvent) {
  e.preventDefault()
  isDragging = true
  startX = e.clientX
  startWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--vp-sidebar-width'), 10) || lastWidth
  document.documentElement.classList.add('sidebar-resizing')
  document.addEventListener('mousemove', onDragMove)
  document.addEventListener('mouseup', onDragEnd)
}

function onTouchStart(e: TouchEvent) {
  e.preventDefault()
  isDragging = true
  startX = e.touches[0].clientX
  startWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--vp-sidebar-width'), 10) || lastWidth
  document.documentElement.classList.add('sidebar-resizing')
  document.addEventListener('touchmove', onTouchMove)
  document.addEventListener('touchend', onTouchEnd)
}

function onDragMove(e: MouseEvent) {
  if (!isDragging) return
  const delta = e.clientX - startX
  const raw = startWidth + delta
  if (raw < COLLAPSE_THRESHOLD) {
    // 拖到阈值以下 → 视觉预览收起
    applySidebarWidth(0)
  } else {
    const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, raw))
    applySidebarWidth(newWidth)
  }
}

function onTouchMove(e: TouchEvent) {
  if (!isDragging) return
  const delta = e.touches[0].clientX - startX
  const raw = startWidth + delta
  if (raw < COLLAPSE_THRESHOLD) {
    applySidebarWidth(0)
  } else {
    const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, raw))
    applySidebarWidth(newWidth)
  }
}

function onDragEnd() {
  if (!isDragging) return
  isDragging = false
  document.documentElement.classList.remove('sidebar-resizing')
  document.removeEventListener('mousemove', onDragMove)
  document.removeEventListener('mouseup', onDragEnd)

  const current = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--vp-sidebar-width'), 10)
  if (current < COLLAPSE_THRESHOLD) {
    // 收起
    collapseLeft()
  } else {
    // 保存新宽度
    const clamped = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, current))
    lastWidth = clamped
    applySidebarWidth(clamped)
    localStorage.setItem('sidebar-width', String(clamped))
    // 确保非收起状态
    if (leftCollapsed.value) expandLeft()
  }
}

function onTouchEnd() {
  if (!isDragging) return
  isDragging = false
  document.documentElement.classList.remove('sidebar-resizing')
  document.removeEventListener('touchmove', onTouchMove)
  document.removeEventListener('touchend', onTouchEnd)

  const current = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--vp-sidebar-width'), 10)
  if (current < COLLAPSE_THRESHOLD) {
    collapseLeft()
  } else {
    const clamped = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, current))
    lastWidth = clamped
    applySidebarWidth(clamped)
    localStorage.setItem('sidebar-width', String(clamped))
    if (leftCollapsed.value) expandLeft()
  }
}

function onDoubleClick() {
  applySidebarWidth(DEFAULT_WIDTH)
  lastWidth = DEFAULT_WIDTH
  localStorage.setItem('sidebar-width', String(DEFAULT_WIDTH))
  if (leftCollapsed.value) expandLeft()
}

// ===== 生命周期 =====
onMounted(() => {
  // 恢复折叠状态
  const savedLeft = localStorage.getItem('sidebar-left-collapsed')
  const savedRight = localStorage.getItem('sidebar-right-collapsed')

  // 恢复宽度
  lastWidth = getSavedWidth()

  if (savedLeft === 'true') {
    leftCollapsed.value = true
    document.documentElement.classList.add('sidebar-left-collapsed')
  } else {
    applySidebarWidth(lastWidth)
  }

  if (savedRight === 'true') {
    rightCollapsed.value = true
    document.documentElement.classList.add('sidebar-right-collapsed')
  }

  nextTick(() => injectResizeHandle())
})

// 路由变化后重新注入手柄
watch(() => route.path, () => {
  nextTick(() => {
    setTimeout(injectResizeHandle, 100)
  })
})

onUnmounted(() => {
  document.removeEventListener('mousemove', onDragMove)
  document.removeEventListener('mouseup', onDragEnd)
  document.removeEventListener('touchmove', onTouchMove)
  document.removeEventListener('touchend', onTouchEnd)
})
</script>

<template>
  <template v-if="isDocPage">
    <!-- 左侧栏收起后的展开按钮 -->
    <button
      v-if="leftCollapsed"
      class="sidebar-expand-btn sidebar-expand-left"
      @click="expandLeft"
      title="展开左侧栏"
      aria-label="Expand left sidebar"
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5.5 3L10.5 8l-5 5"/>
      </svg>
    </button>

    <!-- 右侧栏折叠后的展开按钮 -->
    <button
      v-if="rightCollapsed"
      class="sidebar-expand-btn sidebar-expand-right"
      @click="toggleRightAndSave"
      title="展开右侧栏"
      aria-label="Expand right sidebar"
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.5 3L5.5 8l5 5"/>
      </svg>
    </button>

    <!-- 右侧栏未折叠时的收起按钮 -->
    <button
      v-if="!rightCollapsed"
      class="sidebar-collapse-right-btn"
      @click="toggleRightAndSave"
      title="收起右侧栏"
      aria-label="Collapse right sidebar"
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5.5 3L10.5 8l-5 5"/>
      </svg>
    </button>
  </template>
</template>

<style scoped>
.sidebar-expand-btn {
  position: fixed;
  z-index: 26;
  top: calc(var(--vp-nav-height, 64px) + 12px);
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-3);
  cursor: pointer;
  opacity: 0.5;
  transition: all 0.2s ease;
}

.sidebar-expand-btn:hover {
  opacity: 1;
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.sidebar-expand-left {
  left: 8px;
}

.sidebar-expand-right {
  right: 8px;
}

.sidebar-collapse-right-btn {
  position: fixed;
  z-index: 26;
  top: calc(var(--vp-nav-height, 64px) + 12px);
  right: 280px;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid transparent;
  border-radius: 4px;
  background: transparent;
  color: var(--vp-c-text-3);
  cursor: pointer;
  opacity: 0.3;
  transition: all 0.2s ease;
}

.sidebar-collapse-right-btn:hover {
  opacity: 1;
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

@media (max-width: 960px) {
  .sidebar-expand-left {
    display: none;
  }
}

@media (max-width: 1280px) {
  .sidebar-expand-right,
  .sidebar-collapse-right-btn {
    display: none;
  }
}
</style>
