<script setup lang="ts">
import { computed, ref } from 'vue'

// ACT 架构互动演示：两个标签页
//  1) 架构速览：逐节点高亮，讲清每个模块是什么、关键点、优势
//  2) 动作分块 & 时间集成：可调 k 与 m，直观看到"块重叠 + 对同一时刻加权平均"如何运转
// 目标：让完全没有机器人背景的读者也能记住 ACT 的骨架。

type Tab = 'walk' | 'ensemble'
const tab = ref<Tab>('walk')

interface Step {
  emoji: string
  short: string
  title: string
  phase: 'train' | 'both'
  desc: string
  key: string
  adv: string
}

const steps: Step[] = [
  {
    emoji: '🕹️',
    short: '遥操作采集',
    title: '① ALOHA 遥操作：把人手的动作变成数据',
    phase: 'train',
    desc: '操作者反向拖动较小的 leader 臂，较大的 follower 臂同步镜像运动。系统以 50Hz 记录 follower 的观测和 leader 的关节角。',
    key: '记录的是 leader 关节角，不是 follower 的——leader 与 follower 的角度差隐含了人当时使了多大的劲。',
    adv: '整套硬件不到 2 万美元，却能采集到穿扎带、插电池这类高精度演示。',
  },
  {
    emoji: '📷',
    short: '观测编码',
    title: '② 观测：4 路图像 + 14 维关节角',
    phase: 'both',
    desc: '两张腕部相机 + 前视 + 顶视，共 4 张 480×640 图像。每张经 ResNet18 变成 300×512 的 token，加 2D 位置编码后拼成 1200 个 token，再接上 14 维关节角。',
    key: '四路图像共享同一个 ResNet 骨干，落在同一特征空间里才能互相对齐。',
    adv: '腕部相机提供夹爪附近的特写，让策略看清接触状态。',
  },
  {
    emoji: '🎲',
    short: 'CVAE 编码器',
    title: '③ CVAE 编码器：推断"这次是哪种做法"',
    phase: 'train',
    desc: '编码器只看关节角 + 真实动作块（不含图像），输出风格变量 z 的对角高斯参数。它只在训练时存在。',
    key: '风格变量 z 承载演示的随机性——同一次抓取，人可以做出一百种都正确的轨迹。',
    adv: '测试时整块丢弃，模型仍能稳定工作，却已把多模态"编译"进了解码器参数。',
  },
  {
    emoji: '🧠',
    short: '解码器/策略',
    title: '④ 解码器 = 策略：一次吐出整段动作块',
    phase: 'both',
    desc: '解码器吃"图像 + 关节角 + z"，经 Transformer 编码器综合、解码器并行生成，再经 MLP 得到 k×14 的目标关节角。',
    key: '解码器输入是固定的位置嵌入，不是自回归——一次并行输出整个 k 步块。',
    adv: '推理约 0.01 秒，足够实时，也不会像自回归那样误差逐 token 累积。',
  },
  {
    emoji: '✂️',
    short: '分块+时间集成',
    title: '⑤ 动作分块 + 时间集成：不滚雪球、不抖动',
    phase: 'both',
    desc: '策略每一步都查询一次，产生的动作块互相重叠；对同一个目标时刻的多个预测做指数加权平均。',
    key: '权重 w_i = exp(-m·i)：越旧的预测权重越高（当锚），m 越小越偏向新预测。',
    adv: '有效视野缩短 k 倍，复合误差不再逐帧累积；集成把动作磨平滑。',
  },
  {
    emoji: '🤖',
    short: '真机执行',
    title: '⑥ 真机执行：50Hz 底层 PID 闭环',
    phase: 'both',
    desc: '集成后的目标关节角交给 Dynamixel 电机内的 PID 控制器高频跟踪，相机持续提供闭环视觉反馈。',
    key: '精细任务需要高频控制，低频会把撬盖、插电池的关键调整窗口抹平。',
    adv: '6 个真机任务达到 80-90% 成功率，只需约 10 分钟演示。',
  },
]

const current = ref(0)
const step = computed(() => steps[current.value])
function go(i: number) {
  current.value = Math.max(0, Math.min(steps.length - 1, i))
}

// ---------------- 动作分块 & 时间集成 ----------------
const k = ref(5)
const m = ref(0.1)
const target = ref(9)

const cols = 16 // 时间轴 0..15
const cellW = 40

// 覆盖目标时刻 target 的所有"发起预测的时刻"（从旧到新）
const issues = computed(() => {
  const lo = Math.max(0, target.value - k.value + 1)
  const list: number[] = []
  for (let q = lo; q <= target.value; q++) list.push(q)
  return list
})

// 每个候选预测：发起时刻 q、年龄 a（0=最旧）、权重、一个虚构的预测值
const candidates = computed(() =>
  issues.value
    .map((q, idx) => {
      const age = idx // 0 最旧
      const weight = Math.exp(-m.value * age)
      // 虚构的"某关节角预测值"：随发起时刻轻微起伏，模拟不同预测的差异
      const value = 42 + 18 * Math.sin(q * 1.05)
      return { q, age, weight, value }
    })
    .slice()
    .reverse(), // 显示时按"新 → 旧"排，阅读更直观
)

const weightSum = computed(() => candidates.value.reduce((s, c) => s + c.weight, 0))
const maxWeight = computed(() => Math.max(...candidates.value.map((c) => c.weight)))
const ensembleValue = computed(() => {
  const s = weightSum.value
  if (!s) return 0
  return candidates.value.reduce((acc, c) => acc + c.weight * c.value, 0) / s
})

function advance() {
  target.value = target.value >= cols - 1 ? 3 : target.value + 1
}
function reset() {
  k.value = 5
  m.value = 0.1
  target.value = 9
}
</script>

<template>
  <section class="act-pg" aria-label="ACT 架构互动演示">
    <div class="pg-head">
      <h3>🧩 ACT 长什么样：点一点，走一遍</h3>
      <p>ACT（Action Chunking with Transformers）是 2023 年 ALOHA 论文提出的双臂精细操作策略。它由"遥操作采数据 + CVAE + 动作分块"三块拼成。点节点看详情，或切到第二个标签页亲手调参。</p>
      <div class="tabs">
        <button :class="{ on: tab === 'walk' }" @click="tab = 'walk'">① 架构速览</button>
        <button :class="{ on: tab === 'ensemble' }" @click="tab = 'ensemble'">② 分块与时间集成</button>
      </div>
    </div>

    <!-- 标签页 1：架构速览 -->
    <div v-if="tab === 'walk'">
      <div class="pipeline">
        <div
          v-for="(s, i) in steps"
          :key="i"
          class="node"
          :class="{ on: i === current, train: s.phase === 'train' }"
          @click="go(i)"
        >
          <div class="emoji">{{ s.emoji }}</div>
          <div class="label">{{ s.short }}</div>
          <span class="tag">{{ s.phase === 'train' ? '仅训练' : '训练+推理' }}</span>
        </div>
      </div>
      <div class="flow-hint">👉 左边数据 → 中间建模 → 右边执行，虚线节点是"只在训练时存在"的部件</div>

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

    <!-- 标签页 2：分块与时间集成 -->
    <div v-else class="ensemble">
      <p class="lead">
        策略<b>每一步</b>都查询一次，于是相邻的动作块互相重叠。看第
        <b>{{ target }}</b> 步：有 {{ candidates.length }} 个历史预测都"说"了它该做什么，时间集成把它们加权合并。
      </p>

      <div class="timeline-wrap">
        <div class="tl" :style="{ width: cols * cellW + 'px' }">
          <!-- 目标时刻竖线 -->
          <div class="target-line" :style="{ left: target * cellW + cellW / 2 + 'px' }"></div>
          <!-- 每个发起时刻画一个动作块条 -->
          <div
            v-for="c in candidates"
            :key="c.q"
            class="chunk"
            :style="{
              left: c.q * cellW + 'px',
              width: k * cellW + 'px',
              top: candidates.indexOf(c) * 30 + 'px',
            }"
          >
            <span class="chunk-label">预测于 t={{ c.q }}</span>
          </div>
          <!-- 时间轴刻度 -->
          <div v-for="n in cols" :key="n" class="tick" :style="{ left: (n - 1) * cellW + 'px' }">
            <span v-if="(n - 1) % 2 === 0">{{ n - 1 }}</span>
          </div>
        </div>
      </div>

      <div class="weight-panel">
        <div class="wp-title">对第 {{ target }} 步的各预测权重（新 → 旧）</div>
        <div v-for="c in candidates" :key="c.q" class="wrow">
          <span class="wq">t={{ c.q }}</span>
          <span class="wbar"><i :style="{ width: (c.weight / maxWeight) * 100 + '%' }"></i></span>
          <span class="wv">{{ (c.weight / weightSum * 100).toFixed(0) }}%</span>
        </div>
        <div class="ensemble-out">
          加权平均动作 ≈ <b>{{ ensembleValue.toFixed(1) }}</b> 度　
          <span class="note">越旧的预测权重越高（{{ candidates.length ? candidates[candidates.length - 1].q : '-' }} 的预测是锚），m 越小越偏向新预测</span>
        </div>
      </div>

      <div class="controls">
        <label>分块长度 k <input v-model.number="k" type="range" min="3" max="8" step="1" /><output>{{ k }}</output></label>
        <label>纳入速度 m <input v-model.number="m" type="range" min="0" max="0.4" step="0.01" /><output>{{ m.toFixed(2) }}</output></label>
        <div class="btn-row">
          <button @click="advance">向未来推进一步 →</button>
          <button @click="reset">重置</button>
        </div>
      </div>
      <p class="takeaway">
        把 k 调大：每段动作更长、视野更短（复合误差更难累积），但块与块的重叠也更"重"。把 m 调大：只剩最旧的锚说了算，动作更稳但对新观测反应更慢。实际使用中，正是这一对旋钮的平衡让动作既不滚雪球也不抖。
      </p>
    </div>
  </section>
</template>

<style scoped>
.act-pg{margin:1.5rem 0 2rem;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg-soft);border-radius:10px;padding:1.1rem 1.2rem}
.pg-head h3{margin:0 0 .4rem;font-size:1.1rem}
.pg-head p{margin:0 0 .8rem;color:var(--vp-c-text-2);font-size:.9rem;line-height:1.6}
.tabs{display:flex;gap:.5rem;flex-wrap:wrap}
.tabs button{border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);color:var(--vp-c-text-1);padding:.42rem .8rem;border-radius:8px;cursor:pointer;font-size:.85rem}
.tabs button.on{border-color:var(--vp-c-brand-1);background:var(--vp-c-brand-soft);color:var(--vp-c-brand-1);font-weight:600}

.pipeline{display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;margin-top:1rem}
.node{flex:1 1 120px;min-width:110px;border:2px solid var(--vp-c-divider);background:var(--vp-c-bg);border-radius:12px;padding:.6rem .5rem;text-align:center;cursor:pointer;position:relative;transition:.15s}
.node:hover{transform:translateY(-2px)}
.node.on{border-color:var(--vp-c-brand-1);background:var(--vp-c-brand-soft);box-shadow:0 0 0 3px rgba(33,150,243,.15)}
.node.train{border-style:dashed}
.node .emoji{font-size:1.6rem;line-height:1.2}
.node .label{font-size:.8rem;font-weight:600;margin-top:.2rem}
.node .tag{display:inline-block;margin-top:.35rem;font-size:.62rem;color:var(--vp-c-text-3);border:1px solid var(--vp-c-divider);border-radius:999px;padding:0 .4rem}
.flow-hint{margin:.7rem 0 0;font-size:.78rem;color:var(--vp-c-text-3)}

.detail{margin-top:1rem;border:1px solid var(--vp-c-divider);border-left:4px solid var(--vp-c-brand-1);background:var(--vp-c-bg);border-radius:8px;padding:.85rem 1rem}
.detail h4{margin:0 0 .45rem;font-size:1rem}
.detail .desc{margin:0 0 .6rem;font-size:.88rem;line-height:1.65;color:var(--vp-c-text-1)}
.kv{display:flex;gap:.5rem;font-size:.85rem;line-height:1.6;margin:.3rem 0}
.kv .k{flex:0 0 auto;font-weight:700;color:#F44336;background:rgba(244,67,54,.1);border-radius:5px;padding:0 .45rem;height:fit-content}
.kv .k.adv{color:#2E7D32;background:rgba(76,175,80,.14)}
.kv .v{color:var(--vp-c-text-1)}

.stepper{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-top:1rem}
.stepper button{border:1px solid var(--vp-c-brand-1);background:var(--vp-c-bg);color:var(--vp-c-brand-1);padding:.4rem .8rem;border-radius:8px;cursor:pointer;font-size:.85rem}
.stepper button:disabled{opacity:.4;cursor:not-allowed}
.dots{display:flex;gap:.4rem}
.dots span{width:10px;height:10px;border-radius:50%;background:var(--vp-c-divider);cursor:pointer}
.dots span.on{background:var(--vp-c-brand-1)}

.ensemble .lead{font-size:.9rem;line-height:1.65;color:var(--vp-c-text-1)}
.ensemble .lead b{color:var(--vp-c-brand-1)}
.timeline-wrap{overflow-x:auto;padding:.5rem 0;border-bottom:1px dashed var(--vp-c-divider)}
.tl{position:relative;height:220px;background-image:repeating-linear-gradient(90deg,var(--vp-c-divider) 0 1px,transparent 1px 40px)}
.chunk{position:absolute;height:22px;background:linear-gradient(180deg,rgba(33,150,243,.85),rgba(33,150,243,.55));border-radius:6px;display:flex;align-items:center;padding:0 .5rem;color:#fff;font-size:.68rem;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,.15)}
.target-line{position:absolute;top:0;bottom:0;width:2px;background:#F44336;z-index:2}
.tick{position:absolute;bottom:-2px;font-size:.65rem;color:var(--vp-c-text-3)}
.weight-panel{margin-top:1rem;border:1px solid var(--vp-c-divider);background:var(--vp-c-bg);border-radius:8px;padding:.7rem .85rem}
.wp-title{font-size:.85rem;font-weight:600;margin-bottom:.5rem}
.wrow{display:grid;grid-template-columns:56px minmax(0,1fr) 42px;align-items:center;gap:.55rem;font-size:.78rem;margin:.28rem 0}
.wq{color:var(--vp-c-text-2);font-variant-numeric:tabular-nums}
.wbar{height:12px;background:var(--vp-c-bg-soft);border-radius:999px;overflow:hidden;border:1px solid var(--vp-c-divider)}
.wbar i{display:block;height:100%;background:linear-gradient(90deg,#4CAF50,#2196F3);border-radius:999px}
.wv{text-align:right;color:var(--vp-c-brand-1);font-variant-numeric:tabular-nums}
.ensemble-out{margin-top:.6rem;font-size:.86rem;border-top:1px dashed var(--vp-c-divider);padding-top:.5rem}
.ensemble-out b{color:#E91E63;font-size:1.05rem}
.ensemble-out .note{font-size:.74rem;color:var(--vp-c-text-3)}
.controls{display:grid;gap:.6rem;margin-top:1rem}
.controls label{display:grid;grid-template-columns:110px minmax(0,1fr) 44px;align-items:center;gap:.6rem;font-size:.85rem}
.controls input[type=range]{width:100%;accent-color:var(--vp-c-brand-1)}
.controls output{text-align:right;font-variant-numeric:tabular-nums;color:var(--vp-c-brand-1)}
.btn-row{display:flex;gap:.5rem;flex-wrap:wrap}
.btn-row button{border:1px solid var(--vp-c-brand-1);background:var(--vp-c-bg);color:var(--vp-c-brand-1);padding:.4rem .8rem;border-radius:8px;cursor:pointer;font-size:.83rem}
.takeaway{margin-top:.9rem;font-size:.82rem;line-height:1.7;color:var(--vp-c-text-2);background:var(--vp-c-bg);border-radius:8px;padding:.7rem .85rem;border:1px solid var(--vp-c-divider)}
@media(max-width:720px){
  .node{flex:1 1 90px;min-width:90px}
  .stepper{flex-direction:column;gap:.6rem}
}
</style>
