<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { LoaderCircle, Play, RotateCcw, Square, Terminal } from '@lucide/vue'
import EditorWorker from 'monaco-editor/editor/editor.worker.js?worker'
import PyodideWorker from '../workers/pyodide.worker?worker'

// 本组件是 PhysicsPlayground.vue 的姊妹组件，专门用于渲染 3D Gaussian Splatting
// 系列需要的画面：带旋转角的椭圆（2D 高斯投影）、按深度排序的 alpha blending 合成、
// 每个高斯自己的不透明度和颜色。PhysicsPlayground 的 Drawable 只支持
// circle/segment/polygon，画不出"旋转椭圆 + 半透明叠色"，所以单独建一个组件，
// 而不是往共享组件里加分支，避免影响其余系列已经在用的渲染路径。
const props = withDefaults(defineProps<{
  preset: string
  title?: string
}>(), {
  title: '高斯溅射实验',
})

const exampleModules = import.meta.glob('../playgrounds/gaussian-splatting/*.py', {
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

// 一个 Splat 就是屏幕上的一个 2D 高斯投影：中心 (x,y)，沿两条主轴的半径
// (radiusA, radiusB)，主轴相对水平方向的夹角 angle（弧度），以及自己的颜色
// 和不透明度 alpha。depth 只用于按"从远到近"排序，不直接参与画面坐标。
type Splat = {
  type: 'splat'
  x: number
  y: number
  radiusA: number
  radiusB?: number
  angle?: number
  depth?: number
  color?: string
  alpha?: number
  label?: string
}

type Drawable =
  | Splat
  | { type: 'circle'; x?: number; y?: number; radius?: number; fill?: string; stroke?: string; width?: number; label?: string }
  | { type: 'segment'; from?: number[]; to?: number[]; stroke?: string; width?: number; label?: string }
  | { type: 'point'; x?: number; y?: number; fill?: string; radius?: number; label?: string }

type PixelReadout = {
  x: number
  y: number
  color: string
  layers: { label?: string; color: string; alpha: number; contribution: number; transmittance: number }[]
}

type SimulationFrame = {
  time?: number
  objects: Drawable[]
  metrics?: Record<string, string | number>
  pixel?: PixelReadout
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
  return `gaussian-splat-playground:${props.preset}`
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
    // 旧版本只存了代码字符串，无法判断源码是否已更新，直接丢弃重新加载。
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
}

function drawLabel(context: CanvasRenderingContext2D, label: string | undefined, x: number, y: number) {
  if (!label) return
  context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-text-1') || '#111827'
  context.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace'
  context.textAlign = 'center'
  context.fillText(label, x, y)
}

function drawFrame(frame: SimulationFrame, rawBounds?: number[]) {
  const target = canvasContext()
  if (!target) return
  const { context, width, height } = target
  const bounds = rawBounds?.length === 4 ? rawBounds : [-4, -4, 4, 4]
  const [minX, minY, maxX, maxY] = bounds
  const padding = 28
  const sx = (width - padding * 2) / Math.max(0.001, maxX - minX)
  const sy = (height - padding * 2) / Math.max(0.001, maxY - minY)
  const scale = Math.min(sx, sy)
  const offsetX = (width - (maxX - minX) * scale) / 2
  const offsetY = (height - (maxY - minY) * scale) / 2
  const map = (x: number, y: number) => [
    offsetX + (x - minX) * scale,
    height - offsetY - (y - minY) * scale,
  ]

  context.clearRect(0, 0, width, height)
  context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-bg-soft') || '#f6f6f7'
  context.fillRect(0, 0, width, height)

  // splat 类型需要按 depth 从远到近画（远的先画在下面），这样近处的
  // 高斯才能正确地"盖住"远处高斯的一部分颜色——这正是 alpha blending
  // 公式 C = Σ cᵢαᵢ∏(1-αⱼ) 里"从远到近累乘透过率"的可视化对应。
  const objects = [...(frame.objects ?? [])]
  objects.sort((a, b) => {
    const da = 'depth' in a ? (a.depth ?? 0) : 0
    const db = 'depth' in b ? (b.depth ?? 0) : 0
    return db - da
  })

  for (const object of objects) {
    if (object.type === 'splat') {
      const [x, y] = map(object.x, object.y)
      const radiusA = Math.max(1.5, object.radiusA * scale)
      const radiusB = Math.max(1.5, (object.radiusB ?? object.radiusA) * scale)
      const angle = object.angle ?? 0
      const alpha = object.alpha ?? 1
      context.save()
      context.globalAlpha = Math.min(1, Math.max(0, alpha))
      context.translate(x, y)
      // canvas 的 y 轴朝下、angle 是"世界坐标里逆时针"的角度，翻转一次符号
      // 才能让椭圆在屏幕上转到用户在 Python 里指定的方向。
      context.rotate(-angle)
      context.beginPath()
      context.ellipse(0, 0, radiusA, radiusB, 0, 0, Math.PI * 2)
      context.fillStyle = object.color ?? '#2196f3'
      context.fill()
      context.restore()
      context.save()
      context.globalAlpha = 1
      context.strokeStyle = 'rgba(0,0,0,0.18)'
      context.lineWidth = 1
      context.translate(x, y)
      context.rotate(-angle)
      context.beginPath()
      context.ellipse(0, 0, radiusA, radiusB, 0, 0, Math.PI * 2)
      context.stroke()
      context.restore()
      drawLabel(context, object.label, x, y - radiusB - 8)
    } else if (object.type === 'circle') {
      const [x, y] = map(object.x ?? 0, object.y ?? 0)
      context.lineWidth = object.width ?? 2
      context.strokeStyle = object.stroke ?? '#334155'
      context.fillStyle = object.fill ?? 'transparent'
      context.beginPath()
      context.arc(x, y, Math.max(2, (object.radius ?? 0.1) * scale), 0, Math.PI * 2)
      context.fill()
      context.stroke()
      drawLabel(context, object.label, x, y - (object.radius ?? 0.1) * scale - 8)
    } else if (object.type === 'point') {
      const [x, y] = map(object.x ?? 0, object.y ?? 0)
      context.fillStyle = object.fill ?? '#111827'
      context.beginPath()
      context.arc(x, y, Math.max(2, (object.radius ?? 0.04) * scale), 0, Math.PI * 2)
      context.fill()
      drawLabel(context, object.label, x, y - 10)
    } else if (object.type === 'segment' && object.from && object.to) {
      const [x1, y1] = map(object.from[0], object.from[1])
      const [x2, y2] = map(object.to[0], object.to[1])
      context.lineWidth = object.width ?? 2
      context.strokeStyle = object.stroke ?? '#94a3b8'
      context.beginPath()
      context.moveTo(x1, y1)
      context.lineTo(x2, y2)
      context.stroke()
      drawLabel(context, object.label, (x1 + x2) / 2, (y1 + y2) / 2 - 8)
    }
  }

  // pixel 是可选字段：Python 侧可以标出"正在追踪的那个像素"，并给出它
  // 逐层累积透过率的明细，方便配合 alpha blending 公式逐层核对。
  if (frame.pixel) {
    const [px, py] = map(frame.pixel.x, frame.pixel.y)
    context.save()
    context.strokeStyle = '#111827'
    context.lineWidth = 1.5
    context.beginPath()
    context.moveTo(px - 7, py)
    context.lineTo(px + 7, py)
    context.moveTo(px, py - 7)
    context.lineTo(px, py + 7)
    context.stroke()
    context.restore()
  }
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
}
</style>
