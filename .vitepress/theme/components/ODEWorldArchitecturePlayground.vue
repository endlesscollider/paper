<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'

// ODEWorld 架构互动演示：
//  标签 1 架构速览：逐节点讲清 PT-Flow 的三层结构
//  标签 2 物理时间积分：拖动/播放 τ，看速度场如何被 ODE 积分成未来
// 目标：把"预测未来 = 沿物理时间积分速度场"这件事画出来。

type Tab = 'walk' | 'ode'
const tab = ref<Tab>('walk')

interface Step {
  emoji: string
  short: string
  title: string
  frozen: boolean
  desc: string
  key: string
  adv: string
}

const steps: Step[] = [
  {
    emoji: '🖼️',
    short: '输入观测',
    title: '① 输入：一张真实观测 x₀',
    frozen: false,
    desc: '起点是真实世界的某一帧图像。PT-Flow 的初始状态 s₀ 可以取演示里的任意一帧，不必是首帧。',
    key: '时间 t 定义为"从 s₀ 起经过的物理时间"，是一个真实的量。',
    adv: '起点来自真实观测，因此预测的是真实世界的演化，而不是从噪声生成。',
  },
  {
    emoji: '🧊',
    short: '冻结 DINOv2',
    title: '② 冻结的 DINOv2：像素 → 特征状态 s',
    frozen: true,
    desc: '用预训练 DINOv2 把 480×640 的图像压成 16×16×768 的特征状态。它全程冻结，可训练参数极少。',
    key: '在特征空间里估计速度比在像素空间数值更稳。',
    adv: '冻结强骨干让训练稳定、参数小。',
  },
  {
    emoji: '🧩',
    short: '动力学编码器',
    title: '③ 动力学编码器 f_dyn：抽出"随时间变化"的部分',
    frozen: false,
    desc: '编码器以 s₀ 为条件，把当前特征 s_t 压成一个 1×768 的单 token 潜表示 z_t；解码器再以 s₀ 为条件重建。',
    key: '编码器/解码器都带 s₀ 条件，静态背景可直接从 s₀ 取，逼迫 z_t 只存"变化"。',
    adv: '单 token 就够（压缩 256 倍），潜规划极其轻量。',
  },
  {
    emoji: '⚡',
    short: '速度场 v_θ',
    title: '④ 速度场 v_θ：3 层 MLP + FiLM(τ)',
    frozen: false,
    desc: '一个极轻的网络，输入 z_t、z₀、目标条件 c 和时间 τ，输出"此刻潜状态该往哪走、走多快"的瞬时速度。',
    key: '用 JVP 从真实数据算出潜速度，直接监督它——这条路避开了 JEPA 的表示坍塌。',
    adv: '直接一阶监督不需要额外的坍塌防止正则，还能保住潜空间结构。',
  },
  {
    emoji: '🧮',
    short: 'ODE 积分',
    title: '⑤ ODE 积分：预测 = 沿物理时间累加速度',
    frozen: false,
    desc: '用四阶 Runge-Kutta（RK4）从 z₀ 积分到任意 τ∈[0,1]，得到未来的潜状态 z_τ。全程在紧凑潜空间进行。',
    key: '因为是连续速度场，可以在任意时刻求值——摆脱训练帧率限制。',
    adv: '速度场只有浅 MLP，积分轻量，支持实时生成与快速规划。',
  },
  {
    emoji: '🔮',
    short: '解码/子目标',
    title: '⑥ 两路输出：生成图像 or 当作子目标',
    frozen: true,
    desc: '一路经 g_dyn → g_obs 解码成图像帧；另一路把 z_τ 直接当作紧凑的子目标，喂给下游策略学习。',
    key: '潜状态 z_τ 是动力学上有意义的低维目标，绕开高维视觉规划。',
    adv: '同一套表示同时服务"看"和"决策"。',
  },
  {
    emoji: '🔁',
    short: '连续时间能力',
    title: '⑦ 连续时间独有：反向预测 + 任意分辨率',
    frozen: false,
    desc: '把积分方向取反（v→-v）就能倒放；在任意 τ 求值就能做时间超分辨率、填补丢帧。',
    key: '这两项能力离散时间模型天然没有——它们没有"方向"和"任意时刻"这两回事。',
    adv: '证明模型内化的是连续物理，而非记住前向模式。',
  },
]

const current = ref(0)
const step = computed(() => steps[current.value])
function go(i: number) {
  current.value = Math.max(0, Math.min(steps.length - 1, i))
}

// ---------------- 物理时间积分 ----------------
const W = 520
const H = 190
const padL = 36
const padB = 26
const padT = 14
const tau = ref(0)
const direction = ref<1 | -1>(1)

function profile(t: number): number {
  const approach = 1.15 * Math.exp(-((t - 0.22) ** 2) / 0.0035)
  const pull = 0.85 * Math.exp(-((t - 0.68) ** 2) / 0.01)
  const decay = 1 / (1 + Math.exp(-40 * (0.92 - t)))
  return (approach + pull) * decay
}

const samples = 140
const vmax = 1.25
const xOf = (t: number) => padL + t * (W - padL - 10)
const yOf = (v: number) => H - padB - (v / vmax) * (H - padB - padT)

const curvePath = computed(() => {
  let d = ''
  for (let i = 0; i <= samples; i++) {
    const t = i / samples
    const x = xOf(t)
    const y = yOf(profile(t))
    d += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1) + ' '
  }
  return d
})

// 累计位移（数值积分 ∫v dt），用于显示"已经走了多远"
const cumulative = computed(() => {
  const arr: number[] = [0]
  for (let i = 1; i <= samples; i++) {
    const t0 = (i - 1) / samples
    const t1 = i / samples
    arr.push(arr[i - 1] + ((profile(t0) + profile(t1)) / 2) * (t1 - t0))
  }
  return arr
})
const totalArea = computed(() => cumulative.value[samples])

const vNow = computed(() => profile(tau.value) * direction.value)
const progress = computed(() => {
  const idx = Math.round(tau.value * samples)
  return cumulative.value[idx] / totalArea.value
})

const playheadX = computed(() => xOf(tau.value))
const tokenY = computed(() => yOf(Math.abs(vNow.value)))

let timer: number | null = null
function play() {
  stop()
  timer = window.setInterval(() => {
    tau.value = Math.min(1, Math.max(0, tau.value + 0.012 * direction.value))
    if (tau.value >= 1 || tau.value <= 0) stop()
  }, 40)
}
function stop() {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}
function reset() {
  stop()
  tau.value = 0
  direction.value = 1
}
function flip() {
  direction.value = direction.value === 1 ? -1 : 1
}
onBeforeUnmount(stop)
</script>

<template>
  <section class="ode-pg" aria-label="ODEWorld 架构互动演示">
    <div class="pg-head">
      <h3>🌊 ODEWorld 长什么样：让时间重新变成"连续的"</h3>
      <p>ODEWorld 把"预测下一帧"换成"沿物理时间积分一个速度场"。点节点看架构，或切到第二个标签页，亲手拖动物理时间 τ，看预测是怎么被一步步积分出来的。</p>
      <div class="tabs">
        <button :class="{ on: tab === 'walk' }" @click="tab = 'walk'">① 架构速览</button>
        <button :class="{ on: tab === 'ode' }" @click="tab = 'ode'">② 物理时间积分</button>
      </div>
    </div>

    <div v-if="tab === 'walk'">
      <div class="pipeline">
        <div
          v-for="(s, i) in steps"
          :key="i"
          class="node"
          :class="{ on: i === current, frozen: s.frozen }"
          @click="go(i)"
        >
          <div class="emoji">{{ s.emoji }}</div>
          <div class="label">{{ s.short }}</div>
          <span class="tag">{{ s.frozen ? '冻结' : '可训练' }}</span>
        </div>
      </div>
      <div class="flow-hint">👉 虚线是"冻结"的模块（有预训练权重、不更新），实线是可训练部分</div>

      <div class="detail">
        <h4>{{ step.title }}</h4>
        <p class="desc">{{ step.desc }}</p>
        <div class="kv"><span class="k">关键点</span><span class="v">{{ step.key }}</span></div>
        <div class="kv"><span class="k adv">优势</span><span class="v">{{ step.adv }}</span></div>
      </div>

      <div class="stepper">
        <button :disabled="current === 0" @click="go(current - 1)">← 上一步</button>
        <div class="dots">
          <span v-for="(s, i) in steps" :key="i" :class="{ on: i === current }" @click="go(i)"></span>
        </div>
        <button :disabled="current === steps.length - 1" @click="go(current + 1)">下一步 →</button>
      </div>
    </div>

    <div v-else class="ode">
      <p class="lead">
        横轴是归一化物理时间 τ，曲线是速度场的瞬时幅值 |v|。预测未来 = 把这条曲线下的面积（速度 × 时间）累加起来。
        <b>{{ direction === 1 ? '当前：正向积分' : '当前：反向积分（v → -v）' }}</b>
      </p>

      <svg class="ode-svg" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="速度场与物理时间积分示意">
        <!-- 网格与坐标轴 -->
        <line :x1="padL" :y1="H - padB" :x2="W - 10" :y2="H - padB" stroke="var(--vp-c-divider)" />
        <text :x="padL" :y="H - 8" class="ax">τ=0</text>
        <text :x="W - 30" :y="H - 8" class="ax">τ=1</text>
        <text :x="6" :y="padT + 6" class="ax">|v|</text>

        <!-- 速度曲线 -->
        <path :d="curvePath" fill="none" stroke="#2196F3" stroke-width="2.6" />

        <!-- 阶段标注 -->
        <text :x="xOf(0.30)" :y="yOf(0.30)-8" class="note">靠近把手</text>
        <text :x="xOf(0.62)" :y="yOf(0.62)-10" class="note">拉动抽屉</text>

        <!-- playhead -->
        <line :x1="playheadX" :y1="padT" :x2="playheadX" :y2="H - padB" stroke="#F44336" stroke-width="2" />
        <circle :cx="playheadX" :cy="tokenY" r="6" fill="#F44336" />
        <text :x="playheadX + 8" :y="tokenY - 8" class="ax">潜状态 z_τ</text>
      </svg>

      <div class="readouts">
        <div class="ro"><span>物理时间 τ</span><b>{{ tau.toFixed(2) }}</b></div>
        <div class="ro"><span>瞬时速度 |v|</span><b>{{ Math.abs(vNow).toFixed(2) }}</b></div>
        <div class="ro"><span>已积分位移</span><b>∫v dτ = {{ (progress * totalArea).toFixed(2) }}</b></div>
      </div>
      <div class="prog"><i :style="{ width: (progress * 100) + '%' }"></i></div>

      <div class="controls">
        <label>拖动物理时间 τ <input v-model.number="tau" type="range" min="0" max="1" step="0.01" /><output>{{ tau.toFixed(2) }}</output></label>
        <div class="btn-row">
          <button @click="play">▶ 播放积分</button>
          <button @click="stop">⏸ 暂停</button>
          <button @click="flip">⇄ 切换方向</button>
          <button @click="reset">重置</button>
        </div>
      </div>
      <p class="takeaway">
        注意曲线是"分阶段"的：靠近把手时速度先升后降、几乎停住，进入拉动阶段再上升。这说明速度场不只是指向目标，还编码了每个阶段的正确瞬时速度。把方向切到反向，同一条曲线就能用来倒放——这是离散时间模型给不了的。
      </p>
    </div>
  </section>
</template>

<style scoped>
.ode-pg{margin:1.5rem 0 2rem;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg-soft);border-radius:10px;padding:1.1rem 1.2rem}
.pg-head h3{margin:0 0 .4rem;font-size:1.1rem}
.pg-head p{margin:0 0 .8rem;color:var(--vp-c-text-2);font-size:.9rem;line-height:1.6}
.tabs{display:flex;gap:.5rem;flex-wrap:wrap}
.tabs button{border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);color:var(--vp-c-text-1);padding:.42rem .8rem;border-radius:8px;cursor:pointer;font-size:.85rem}
.tabs button.on{border-color:var(--vp-c-brand-1);background:var(--vp-c-brand-soft);color:var(--vp-c-brand-1);font-weight:600}

.pipeline{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}
.node{flex:1 1 110px;min-width:104px;border:2px solid var(--vp-c-divider);background:var(--vp-c-bg);border-radius:12px;padding:.6rem .5rem;text-align:center;cursor:pointer;transition:.15s}
.node:hover{transform:translateY(-2px)}
.node.on{border-color:var(--vp-c-brand-1);background:var(--vp-c-brand-soft);box-shadow:0 0 0 3px rgba(33,150,243,.15)}
.node.frozen{border-style:dashed;background:repeating-linear-gradient(45deg,var(--vp-c-bg) 0 6px,var(--vp-c-bg-soft) 6px 12px)}
.node .emoji{font-size:1.5rem;line-height:1.2}
.node .label{font-size:.78rem;font-weight:600;margin-top:.2rem}
.node .tag{display:inline-block;margin-top:.35rem;font-size:.62rem;color:var(--vp-c-text-3);border:1px solid var(--vp-c-divider);border-radius:999px;padding:0 .4rem}
.flow-hint{margin:.7rem 0 0;font-size:.78rem;color:var(--vp-c-text-3)}

.detail{margin-top:1rem;border:1px solid var(--vp-c-divider);border-left:4px solid var(--vp-c-brand-1);background:var(--vp-c-bg);border-radius:8px;padding:.85rem 1rem}
.detail h4{margin:0 0 .45rem;font-size:1rem}
.detail .desc{margin:0 0 .6rem;font-size:.88rem;line-height:1.65}
.kv{display:flex;gap:.5rem;font-size:.85rem;line-height:1.6;margin:.3rem 0}
.kv .k{flex:0 0 auto;font-weight:700;color:#F44336;background:rgba(244,67,54,.1);border-radius:5px;padding:0 .45rem;height:fit-content}
.kv .k.adv{color:#2E7D32;background:rgba(76,175,80,.14)}

.stepper{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-top:1rem}
.stepper button{border:1px solid var(--vp-c-brand-1);background:var(--vp-c-bg);color:var(--vp-c-brand-1);padding:.4rem .8rem;border-radius:8px;cursor:pointer;font-size:.85rem}
.stepper button:disabled{opacity:.4;cursor:not-allowed}
.dots{display:flex;gap:.4rem}
.dots span{width:10px;height:10px;border-radius:50%;background:var(--vp-c-divider);cursor:pointer}
.dots span.on{background:var(--vp-c-brand-1)}

.ode .lead{font-size:.9rem;line-height:1.65}
.ode .lead b{color:var(--vp-c-brand-1)}
.ode-svg{width:100%;max-width:560px;background:var(--vp-c-bg);border:1px solid var(--vp-c-divider);border-radius:8px;margin:.6rem 0}
.ode-svg .ax{font-size:9px;fill:var(--vp-c-text-3)}
.ode-svg .note{font-size:9px;fill:var(--vp-c-text-2)}
.readouts{display:flex;flex-wrap:wrap;gap:.6rem;margin:.4rem 0}
.ro{flex:1 1 150px;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);border-radius:8px;padding:.45rem .6rem;display:flex;justify-content:space-between;font-size:.82rem}
.ro b{color:var(--vp-c-brand-1);font-variant-numeric:tabular-nums}
.prog{height:8px;background:var(--vp-c-bg);border:1px solid var(--vp-c-divider);border-radius:999px;overflow:hidden}
.prog i{display:block;height:100%;background:linear-gradient(90deg,#2196F3,#9C27B0);border-radius:999px;transition:width .1s}
.controls{display:grid;gap:.6rem;margin-top:1rem}
.controls label{display:grid;grid-template-columns:130px minmax(0,1fr) 48px;align-items:center;gap:.6rem;font-size:.85rem}
.controls input[type=range]{width:100%;accent-color:var(--vp-c-brand-1)}
.controls output{text-align:right;font-variant-numeric:tabular-nums;color:var(--vp-c-brand-1)}
.btn-row{display:flex;gap:.5rem;flex-wrap:wrap}
.btn-row button{border:1px solid var(--vp-c-brand-1);background:var(--vp-c-bg);color:var(--vp-c-brand-1);padding:.4rem .8rem;border-radius:8px;cursor:pointer;font-size:.83rem}
.takeaway{margin-top:.9rem;font-size:.82rem;line-height:1.7;color:var(--vp-c-text-2);background:var(--vp-c-bg);border-radius:8px;padding:.7rem .85rem;border:1px solid var(--vp-c-divider)}
@media(max-width:720px){.stepper{flex-direction:column;gap:.6rem}}
</style>
