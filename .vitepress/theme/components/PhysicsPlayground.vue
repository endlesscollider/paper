<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { LoaderCircle, Play, RotateCcw, Square, Terminal } from '@lucide/vue'
import EditorWorker from 'monaco-editor/editor/editor.worker.js?worker'
import PyodideWorker from '../workers/pyodide.worker?worker'

const props = withDefaults(defineProps<{
  preset: string
  title?: string
}>(), {
  title: '交互实验',
})

const exampleModules = import.meta.glob('../playgrounds/newton-solver/*.py', {
  eager: true,
  query: '?raw',
  import: 'default',
}) as Record<string, string>

const originalCode = computed(() => {
  const entry = Object.entries(exampleModules).find(([path]) => path.endsWith(`/${props.preset}.py`))
  return entry?.[1] ?? ''
})

const editorHost = ref<HTMLElement | null>(null)
const canvas = ref<HTMLCanvasElement | null>(null)
const status = ref<'idle' | 'loading' | 'ready' | 'running' | 'done' | 'paused' | 'error'>('idle')
const workerActive = ref(false)
const isAnimating = ref(false)
const output = ref('')
const activePane = ref<'preview' | 'output'>('preview')
const simulation = ref<Simulation | null>(null)
const currentFrame = ref(0)

let editor: any = null
let monaco: any = null
let worker: Worker | null = null
let resizeObserver: ResizeObserver | null = null
let themeObserver: MutationObserver | null = null
let animationId = 0
let animationStartedAt = 0

type Drawable = {
  type: 'circle' | 'segment' | 'polygon'
  x?: number
  y?: number
  radius?: number
  points?: number[][]
  from?: number[]
  to?: number[]
  fill?: string
  stroke?: string
  width?: number
  label?: string
}

type SimulationFrame = {
  time?: number
  objects: Drawable[]
  metrics?: Record<string, string | number>
}

type Simulation = {
  bounds?: number[]
  fps?: number
  loop?: boolean
  frames: SimulationFrame[]
}

const isBusy = computed(() => workerActive.value)
const canStop = computed(() => workerActive.value || isAnimating.value)
const statusText = computed(() => ({
  idle: '未运行',
  loading: '正在加载 Python',
  ready: 'Python 已就绪',
  running: '正在运行',
  done: '运行完成',
  paused: '已暂停',
  error: '运行失败',
}[status.value]))

function storageKey() {
  return `physics-playground:${props.preset}`
}

function getCode() {
  return editor?.getValue() ?? originalCode.value
}

type StoredCode = {
  source: string
  code: string
}

function loadCode() {
  const raw = localStorage.getItem(storageKey())
  if (!raw) return originalCode.value

  try {
    const saved = JSON.parse(raw) as Partial<StoredCode>
    if (saved.source === originalCode.value && typeof saved.code === 'string') {
      return saved.code
    }
  } catch {
    // Legacy entries stored only the raw code and cannot detect source updates.
  }

  localStorage.removeItem(storageKey())
  return originalCode.value
}

function saveCode() {
  const value: StoredCode = {
    source: originalCode.value,
    code: getCode(),
  }
  localStorage.setItem(storageKey(), JSON.stringify(value))
}

function createWorker() {
  if (worker) return worker
  const instance = new PyodideWorker()
  worker = instance
  workerActive.value = true
  instance.onmessage = ({ data }) => {
    // A terminated worker can still have a queued message in the event loop.
    if (worker !== instance) return
    if (data.type === 'status') {
      status.value = data.status
      return
    }
    if (data.type === 'result') {
      workerActive.value = false
      status.value = 'done'
      output.value = [data.stdout, data.stderr].filter(Boolean).join('\n') || '运行完成，没有文本输出。'
      simulation.value = normalizeSimulation(data.simulation)
      currentFrame.value = 0
      startAnimation()
      return
    }
    if (data.type === 'error') {
      workerActive.value = false
      status.value = 'error'
      output.value = data.message
      activePane.value = 'output'
    }
  }
  instance.onerror = (event) => {
    if (worker !== instance) return
    workerActive.value = false
    status.value = 'error'
    output.value = event.message || 'Python Worker 启动失败。'
    activePane.value = 'output'
  }
  return worker
}

function normalizeSimulation(value: unknown): Simulation | null {
  if (!value || typeof value !== 'object') return null
  const candidate = value as Simulation
  if (!Array.isArray(candidate.frames) || candidate.frames.length === 0) return null
  return candidate
}

function runCode() {
  if (worker) {
    worker.terminate()
    worker = null
  }
  stopAnimation()
  simulation.value = null
  output.value = ''
  status.value = 'loading'
  activePane.value = 'preview'
  saveCode()
  workerActive.value = true
  createWorker().postMessage({ type: 'run', code: getCode() })
}

function stopCode() {
  if (workerActive.value) {
    worker?.terminate()
    worker = null
    workerActive.value = false
    stopAnimation()
    status.value = 'idle'
    output.value = '运行已停止。'
    activePane.value = 'output'
    return
  }

  if (isAnimating.value) {
    stopAnimation()
    status.value = 'paused'
  }
}

function resetCode() {
  worker?.terminate()
  worker = null
  workerActive.value = false
  editor?.setValue(originalCode.value)
  localStorage.removeItem(storageKey())
  output.value = ''
  simulation.value = null
  currentFrame.value = 0
  status.value = 'idle'
  stopAnimation()
  drawEmptyCanvas()
}

function stopAnimation() {
  cancelAnimationFrame(animationId)
  animationId = 0
  isAnimating.value = false
}

function startAnimation() {
  stopAnimation()
  if (!simulation.value) return
  isAnimating.value = true
  animationStartedAt = performance.now()
  const tick = (now: number) => {
    const data = simulation.value
    if (!data) {
      isAnimating.value = false
      return
    }
    const fps = Math.max(1, data.fps ?? 30)
    const elapsedFrame = Math.floor((now - animationStartedAt) / (1000 / fps))
    const last = data.frames.length - 1
    currentFrame.value = data.loop === false
      ? Math.min(elapsedFrame, last)
      : elapsedFrame % data.frames.length
    drawFrame(data.frames[currentFrame.value], data.bounds)
    if (data.loop !== false || currentFrame.value < last) {
      animationId = requestAnimationFrame(tick)
    } else {
      animationId = 0
      isAnimating.value = false
    }
  }
  animationId = requestAnimationFrame(tick)
}

function canvasContext() {
  const element = canvas.value
  if (!element) return null
  const rect = element.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  const width = Math.max(1, Math.round(rect.width * dpr))
  const height = Math.max(1, Math.round(rect.height * dpr))
  if (element.width !== width || element.height !== height) {
    element.width = width
    element.height = height
  }
  const context = element.getContext('2d')
  context?.setTransform(dpr, 0, 0, dpr, 0, 0)
  return context ? { context, width: rect.width, height: rect.height } : null
}

function drawEmptyCanvas() {
  const target = canvasContext()
  if (!target) return
  const { context, width, height } = target
  context.clearRect(0, 0, width, height)
  context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-bg-soft') || '#f6f6f7'
  context.fillRect(0, 0, width, height)
  context.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-divider') || '#dedede'
  context.lineWidth = 1
  context.beginPath()
  context.moveTo(24, height - 32)
  context.lineTo(width - 24, height - 32)
  context.stroke()
}

function drawFrame(frame: SimulationFrame, rawBounds?: number[]) {
  const target = canvasContext()
  if (!target) return
  const { context, width, height } = target
  const bounds = rawBounds?.length === 4 ? rawBounds : [-6, -1, 6, 8]
  const [minX, minY, maxX, maxY] = bounds
  const padding = 28
  const sx = (width - padding * 2) / Math.max(0.001, maxX - minX)
  const sy = (height - padding * 2) / Math.max(0.001, maxY - minY)
  const scale = Math.min(sx, sy)
  const offsetX = (width - (maxX - minX) * scale) / 2
  const offsetY = (height - (maxY - minY) * scale) / 2
  const map = (point: number[]) => [
    offsetX + (point[0] - minX) * scale,
    height - offsetY - (point[1] - minY) * scale,
  ]

  context.clearRect(0, 0, width, height)
  context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-bg-soft') || '#f6f6f7'
  context.fillRect(0, 0, width, height)

  for (const object of frame.objects ?? []) {
    context.lineWidth = object.width ?? 2
    context.strokeStyle = object.stroke ?? '#334155'
    context.fillStyle = object.fill ?? 'transparent'

    if (object.type === 'circle') {
      const [x, y] = map([object.x ?? 0, object.y ?? 0])
      context.beginPath()
      context.arc(x, y, Math.max(2, (object.radius ?? 0.2) * scale), 0, Math.PI * 2)
      context.fill()
      context.stroke()
      drawLabel(context, object.label, x, y - (object.radius ?? 0.2) * scale - 8)
    } else if (object.type === 'segment' && object.from && object.to) {
      const [x1, y1] = map(object.from)
      const [x2, y2] = map(object.to)
      context.beginPath()
      context.moveTo(x1, y1)
      context.lineTo(x2, y2)
      context.stroke()
      drawLabel(context, object.label, (x1 + x2) / 2, (y1 + y2) / 2 - 8)
    } else if (object.type === 'polygon' && object.points?.length) {
      context.beginPath()
      object.points.forEach((point, index) => {
        const [x, y] = map(point)
        index === 0 ? context.moveTo(x, y) : context.lineTo(x, y)
      })
      context.closePath()
      context.fill()
      context.stroke()
      const center = object.points.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0])
      const [x, y] = map([center[0] / object.points.length, center[1] / object.points.length])
      drawLabel(context, object.label, x, y)
    }
  }
}

function drawLabel(context: CanvasRenderingContext2D, label: string | undefined, x: number, y: number) {
  if (!label) return
  context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-text-1') || '#111827'
  context.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace'
  context.textAlign = 'center'
  context.fillText(label, x, y)
}

async function mountEditor() {
  if (!editorHost.value) return
  monaco = await import('monaco-editor/editor/editor.api.js')
  self.MonacoEnvironment = {
    getWorker: () => new EditorWorker(),
  }
  editor = monaco.editor.create(editorHost.value, {
    value: loadCode(),
    language: 'python',
    theme: document.documentElement.classList.contains('dark') ? 'vs-dark' : 'vs',
    automaticLayout: false,
    fontSize: 13,
    lineHeight: 20,
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    wordWrap: 'on',
    tabSize: 4,
    padding: { top: 12, bottom: 12 },
    ariaLabel: `${props.title} Python 编辑器`,
  })
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, runCode)
  resizeObserver = new ResizeObserver(() => {
    editor?.layout()
    if (simulation.value) {
      drawFrame(simulation.value.frames[currentFrame.value], simulation.value.bounds)
    } else {
      drawEmptyCanvas()
    }
  })
  resizeObserver.observe(editorHost.value)
}

onMounted(async () => {
  await nextTick()
  await mountEditor()
  drawEmptyCanvas()
  themeObserver = new MutationObserver(() => {
    editor && monaco.editor.setTheme(document.documentElement.classList.contains('dark') ? 'vs-dark' : 'vs')
    if (simulation.value) drawFrame(simulation.value.frames[currentFrame.value], simulation.value.bounds)
    else drawEmptyCanvas()
  })
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
})

watch(() => props.preset, () => {
  worker?.terminate()
  worker = null
  workerActive.value = false
  editor?.setValue(loadCode())
  output.value = ''
  simulation.value = null
  currentFrame.value = 0
  status.value = 'idle'
  stopAnimation()
  drawEmptyCanvas()
})

onBeforeUnmount(() => {
  saveCode()
  stopAnimation()
  worker?.terminate()
  resizeObserver?.disconnect()
  themeObserver?.disconnect()
  editor?.dispose()
})
</script>

<template>
  <section class="physics-playground" :aria-label="title">
    <header class="playground-toolbar">
      <strong class="playground-title">{{ title }}</strong>
      <span class="playground-status" :class="`is-${status}`">
        <LoaderCircle v-if="isBusy" :size="14" class="spin" aria-hidden="true" />
        {{ statusText }}
      </span>
      <div class="playground-actions">
        <button class="tool-button primary" type="button" :disabled="isBusy" title="运行代码 (Ctrl/Command + Enter)" @click="runCode">
          <Play :size="16" fill="currentColor" aria-hidden="true" />
          <span>运行</span>
        </button>
        <button class="tool-button" type="button" :disabled="!canStop" title="停止运行或动画" @click="stopCode">
          <Square :size="15" fill="currentColor" aria-hidden="true" />
          <span>停止</span>
        </button>
        <button class="tool-button icon-only" type="button" title="恢复初始代码" aria-label="恢复初始代码" @click="resetCode">
          <RotateCcw :size="16" aria-hidden="true" />
        </button>
      </div>
    </header>

    <div class="playground-grid">
      <div ref="editorHost" class="playground-editor" />
      <div class="playground-result">
        <div class="result-tabs" role="tablist" aria-label="运行结果">
          <button type="button" :class="{ active: activePane === 'preview' }" role="tab" @click="activePane = 'preview'">动画</button>
          <button type="button" :class="{ active: activePane === 'output' }" role="tab" @click="activePane = 'output'">
            <Terminal :size="14" aria-hidden="true" />输出
          </button>
        </div>
        <div v-show="activePane === 'preview'" class="canvas-wrap">
          <canvas ref="canvas" class="simulation-canvas" />
          <dl v-if="simulation?.frames[currentFrame]?.metrics" class="frame-metrics">
            <template v-for="(value, key) in simulation.frames[currentFrame].metrics" :key="key">
              <dt>{{ key }}</dt><dd>{{ value }}</dd>
            </template>
          </dl>
        </div>
        <pre v-show="activePane === 'output'" class="playground-output" :class="{ error: status === 'error' }">{{ output || '等待运行…' }}</pre>
      </div>
    </div>
  </section>
</template>

<style scoped>
.physics-playground {
  margin: 24px 0 32px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  background: var(--vp-c-bg);
}

.playground-toolbar {
  min-height: 48px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 7px 10px 7px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
}

.playground-title {
  font-size: 14px;
  letter-spacing: 0;
}

.playground-status {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--vp-c-text-3);
  font-size: 12px;
  white-space: nowrap;
}

.playground-status.is-done,
.playground-status.is-ready { color: #15803d; }
.playground-status.is-paused { color: var(--vp-c-brand-1); }
.playground-status.is-error { color: var(--vp-c-danger-1); }

.playground-actions {
  margin-left: auto;
  display: flex;
  gap: 6px;
}

.tool-button {
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 13px;
  cursor: pointer;
}

.tool-button:hover:not(:disabled) { background: var(--vp-c-default-soft); }
.tool-button.primary { border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-1); color: white; }
.tool-button.primary:hover:not(:disabled) { background: var(--vp-c-brand-2); }
.tool-button.icon-only { width: 32px; padding: 0; }
.tool-button:disabled { cursor: not-allowed; opacity: 0.45; }

.playground-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr);
  min-height: 480px;
}

.playground-editor {
  min-width: 0;
  height: 480px;
  border-right: 1px solid var(--vp-c-divider);
}

.playground-result {
  min-width: 0;
  display: grid;
  grid-template-rows: 40px minmax(0, 1fr);
  height: 480px;
}

.result-tabs {
  display: flex;
  align-items: stretch;
  border-bottom: 1px solid var(--vp-c-divider);
}

.result-tabs button {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 0 14px;
  border: 0;
  border-bottom: 2px solid transparent;
  background: transparent;
  color: var(--vp-c-text-2);
  font-size: 13px;
  cursor: pointer;
}

.result-tabs button.active {
  border-bottom-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.canvas-wrap { position: relative; min-height: 0; }
.simulation-canvas { display: block; width: 100%; height: 100%; }

.frame-metrics {
  position: absolute;
  top: 10px;
  right: 10px;
  display: grid;
  grid-template-columns: auto auto;
  gap: 2px 10px;
  margin: 0;
  padding: 7px 9px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: color-mix(in srgb, var(--vp-c-bg) 90%, transparent);
  font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
}

.frame-metrics dt { color: var(--vp-c-text-3); }
.frame-metrics dd { margin: 0; color: var(--vp-c-text-1); text-align: right; }

.playground-output {
  box-sizing: border-box;
  height: 100%;
  margin: 0;
  padding: 14px;
  overflow: auto;
  background: var(--vp-code-block-bg);
  color: var(--vp-code-block-color);
  font: 12px/1.65 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap;
  word-break: break-word;
}

.playground-output.error { color: var(--vp-c-danger-1); }
.spin { animation: spin 0.9s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 760px) {
  .playground-toolbar { flex-wrap: wrap; }
  .playground-status { order: 3; width: 100%; }
  .playground-grid { grid-template-columns: 1fr; min-height: 0; }
  .playground-editor { height: 400px; border-right: 0; border-bottom: 1px solid var(--vp-c-divider); }
  .playground-result { height: 360px; }
  .canvas-wrap {
    display: grid;
    grid-template-rows: minmax(0, 1fr) auto;
  }
  .simulation-canvas { min-height: 0; }
  .frame-metrics {
    position: static;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 2px 10px;
    padding: 7px 10px;
    border: 0;
    border-top: 1px solid var(--vp-c-divider);
    border-radius: 0;
    background: var(--vp-c-bg-soft);
  }
  .frame-metrics dt,
  .frame-metrics dd { white-space: nowrap; }
  .tool-button span { display: none; }
  .tool-button { width: 32px; padding: 0; }
}
</style>
