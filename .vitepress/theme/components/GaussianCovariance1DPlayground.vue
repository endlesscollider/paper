<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

// 一维高斯交互演示：只有一个参数 σ（标准差 / 宽度）。
// 目的是让读者在最少的参数下看清「方差就是宽度」这件最基础的事——
// 拖动 σ，钟形曲线变高变窄 / 变矮变胖，并标出 ±σ 的位置和曲线下的 68% 面积。

const canvasRef = ref<HTMLCanvasElement | null>(null)

// 只有一个自由度：标准差 σ。均值固定在 0（位置和宽度解耦，位置不影响形状）。
const sigma = ref(1.0)

const presets = [
  { label: '窄 (σ=0.5)', sigma: 0.5 },
  { label: '标准 (σ=1)', sigma: 1.0 },
  { label: '宽 (σ=1.8)', sigma: 1.8 },
]

// 方差 = σ²，读者常听到「方差」这个词，这里显式算出来对照。
const variance = computed(() => sigma.value ** 2)
// 峰值高度 = 归一化高斯在中心的值 1/(σ√(2π))，用来印证「越窄越高」。
const peak = computed(() => 1 / (sigma.value * Math.sqrt(2 * Math.PI)))

let animationId = 0
let dpr = 1

function draw() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const cssW = canvas.clientWidth || 460
  const cssH = canvas.clientHeight || 300
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr)
    canvas.height = Math.round(cssH * dpr)
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssW, cssH)

  const pad = 34 // 底部留白给坐标轴标签
  const xRange = 4 // 横轴显示 [-4, 4]
  // 纵轴固定上限，让读者看到「σ 变小 → 曲线冲得越高」的对比，而不是每次自动缩放。
  const yMax = 0.85
  const toPx = (x: number, y: number): [number, number] => [
    ((x + xRange) / (2 * xRange)) * cssW,
    cssH - pad - (y / yMax) * (cssH - pad - 10),
  ]

  const s = sigma.value
  const g = (x: number) => (1 / (s * Math.sqrt(2 * Math.PI))) * Math.exp(-(x * x) / (2 * s * s))

  // 1) 曲线下 [-σ, σ] 区间填充（约 68% 面积），直观展示「一个 σ 覆盖多宽」。
  ctx.beginPath()
  const [bx0, by0] = toPx(-s, 0)
  ctx.moveTo(bx0, by0)
  for (let x = -s; x <= s; x += (2 * s) / 120) {
    const [px, py] = toPx(x, g(x))
    ctx.lineTo(px, py)
  }
  const [bx1, by1] = toPx(s, 0)
  ctx.lineTo(bx1, by1)
  ctx.closePath()
  ctx.fillStyle = 'rgba(33, 150, 243, 0.18)'
  ctx.fill()

  // 2) 坐标轴（横轴 y=0）
  ctx.strokeStyle = 'rgba(96, 125, 139, 0.6)'
  ctx.lineWidth = 1
  ctx.beginPath()
  const [axL, ay] = toPx(-xRange, 0)
  const [axR] = toPx(xRange, 0)
  ctx.moveTo(axL, ay)
  ctx.lineTo(axR, ay)
  ctx.stroke()

  // 3) ±σ 竖直标注线
  ctx.strokeStyle = 'rgba(244, 67, 54, 0.7)'
  ctx.setLineDash([5, 4])
  ctx.lineWidth = 1.5
  for (const xv of [-s, s]) {
    const [vx, vyTop] = toPx(xv, g(xv))
    const [, vyBot] = toPx(xv, 0)
    ctx.beginPath()
    ctx.moveTo(vx, vyTop)
    ctx.lineTo(vx, vyBot)
    ctx.stroke()
  }
  ctx.setLineDash([])

  // 4) 高斯曲线本体
  ctx.strokeStyle = '#1565c0'
  ctx.lineWidth = 2.5
  ctx.beginPath()
  let first = true
  for (let x = -xRange; x <= xRange; x += (2 * xRange) / 400) {
    const [px, py] = toPx(x, g(x))
    if (first) { ctx.moveTo(px, py); first = false } else ctx.lineTo(px, py)
  }
  ctx.stroke()

  // 5) 中心峰值点
  const [cpx, cpy] = toPx(0, g(0))
  ctx.fillStyle = '#212121'
  ctx.beginPath()
  ctx.arc(cpx, cpy, 3, 0, Math.PI * 2)
  ctx.fill()

  // 6) 文字标注：横轴刻度 -σ / 0 / +σ
  ctx.fillStyle = 'rgba(96, 125, 139, 0.95)'
  ctx.font = '12px ui-monospace, monospace'
  ctx.textAlign = 'center'
  const [lsx] = toPx(-s, 0)
  const [rsx] = toPx(s, 0)
  const [zx] = toPx(0, 0)
  ctx.fillText('-σ', lsx, cssH - pad + 16)
  ctx.fillText('0', zx, cssH - pad + 16)
  ctx.fillText('+σ', rsx, cssH - pad + 16)
}

function scheduleDraw() {
  cancelAnimationFrame(animationId)
  animationId = requestAnimationFrame(draw)
}

watch(sigma, scheduleDraw)

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  dpr = Math.min(window.devicePixelRatio || 1, 2)
  draw()
  if (canvasRef.value) {
    resizeObserver = new ResizeObserver(() => scheduleDraw())
    resizeObserver.observe(canvasRef.value)
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(animationId)
  resizeObserver?.disconnect()
})
</script>

<template>
  <section class="gaussian1d-playground" aria-label="一维高斯交互演示">
    <div class="pg-head">
      <div>
        <h3>一维版：一个参数 σ，看清「方差就是宽度」</h3>
        <p>拖动唯一的参数 σ（标准差），观察钟形曲线怎么变高变窄 / 变矮变胖。阴影区是 [-σ, σ]，约占总面积的 68%。</p>
      </div>
      <div class="preset-row" role="group" aria-label="预设宽度">
        <button v-for="p in presets" :key="p.label" type="button" @click="sigma = p.sigma">{{ p.label }}</button>
      </div>
    </div>
    <div class="pg-grid">
      <div class="canvas-wrap">
        <canvas ref="canvasRef" class="gaussian1d-canvas" aria-label="一维高斯钟形曲线"></canvas>
      </div>
      <div class="controls-panel">
        <div class="control-group">
          <h4>唯一的参数：标准差 σ</h4>
          <label>σ <input v-model.number="sigma" type="range" min="0.3" max="2.2" step="0.01" /><output>{{ sigma.toFixed(2) }}</output></label>
        </div>
        <div class="readout-block">
          <div class="row"><span>方差 σ²</span><b>{{ variance.toFixed(3) }}</b></div>
          <div class="row"><span>峰值高度</span><b>{{ peak.toFixed(3) }}</b></div>
          <p class="readout-note">
            σ 越小 → 方差 σ² 越小 → 曲线越<b>窄</b>、峰值越<b>高</b>；σ 越大 → 曲线越<b>胖</b>、峰值越<b>矮</b>。总面积始终等于 1，被挤高就一定变窄。
          </p>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.gaussian1d-playground{margin:1.5rem 0 2rem;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg-soft);padding:1.1rem}
.pg-head{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-bottom:1rem}
.pg-head h3{margin:0 0 .35rem;font-size:1.08rem}
.pg-head p{margin:0;color:var(--vp-c-text-2);font-size:.9rem}
.preset-row{display:flex;flex-wrap:wrap;gap:.45rem;justify-content:flex-end}
button{border:1px solid var(--vp-c-brand-1);background:var(--vp-c-bg);color:var(--vp-c-brand-1);padding:.38rem .62rem;border-radius:5px;cursor:pointer;white-space:nowrap;font-size:.82rem}
button:hover{background:var(--vp-c-brand-soft)}
.pg-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(220px,.8fr);gap:1rem}
.gaussian1d-canvas{width:100%;aspect-ratio:3/2;max-width:460px;border:1px solid var(--vp-c-divider);background:#f7f9fc;display:block}
.controls-panel{display:grid;gap:.8rem;align-content:start}
.control-group,.readout-block{border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);padding:.75rem .8rem}
.control-group h4{margin:0 0 .6rem;font-size:.92rem}
label{display:grid;grid-template-columns:24px minmax(0,1fr) 48px;align-items:center;gap:.45rem;font-size:.84rem;margin:.48rem 0}
input[type=range]{width:100%;accent-color:var(--vp-c-brand-1)}
output{text-align:right;font-variant-numeric:tabular-nums;color:var(--vp-c-text-2)}
.readout-block .row{display:flex;justify-content:space-between;align-items:center;font-size:.86rem;padding:.28rem 0;border-bottom:1px dashed var(--vp-c-divider)}
.readout-block .row b{font-variant-numeric:tabular-nums;color:var(--vp-c-brand-1)}
.readout-note{margin:.6rem 0 0;color:var(--vp-c-text-2);font-size:.78rem;line-height:1.5}
.readout-note b{color:var(--vp-c-text-1)}
@media(max-width:720px){.pg-head{display:block}.preset-row{justify-content:flex-start;margin-top:.75rem}.pg-grid{grid-template-columns:1fr}}
</style>
