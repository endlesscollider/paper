# Repository Instructions

## Writing Tasks

- Before every writing task in this repository, Codex must inspect `/home/wahaha/paper/.kiro/` and read all applicable files there, including steering rules, skills, and hooks. If applicability is unclear, read all files under `.kiro` before drafting or editing.
- Treat the instructions in `.kiro` as required project writing guidance. For a new article, run `node scripts/prep_writing.mjs <keywords...>` before writing, as required by `.kiro/steering/writing-prep-tool.md`.

## Article Voice

- Explain the subject, not the author's teaching order. Remove meta-narration such as “后面再讲”, “现在再看”, “为了让读者先理解”, and “数值例子要等到公式定义后再代入”.
- Use content-based headings, transitions, and captions. Example: replace “公式建立后再代入数值” with “一维完全非弹性碰撞算例”.
- Scope, prerequisite, navigation, safety, and procedure statements are allowed only when operationally necessary. Before finishing, remove organizational uses of “后面”, “接下来”, “先讲/再讲”, and similar phrases.

## Model / Architecture Explanation Order

- **WHAT → HOW → WHY**. 先把每个模块是什么、接收什么、输出什么讲清楚，"为什么这样设计"留到模块讲完或全部讲完之后。禁止对读者尚未见过的设计提问"为什么"。
- 每个子模块讲解前，先用图标出它在全局的位置，再说输入/输出，再逐步拆内部。
- 架构文章的第一节必须给出完整数据流图（输入 → 中间处理 → 输出），后续每节在图上标注"当前讲到哪里"。
- 输入嵌入（数据怎么进网络）必须在讲内部处理之前出现，不能放到后面。

## Formula Design Rationale

- Every displayed formula, plus any inline formula that introduces or materially changes a method, must be explained as a design, not only translated into a symbol table. State the problem or invariant the formula is intended to satisfy, then explain why its structure follows from that goal.
- Define every nontrivial symbol operationally before its first use. Merely naming a symbol (for example, "constraint multiplier") is not a definition: connect it to quantities the reader already knows, say whether it is an input, state, output, coefficient, or increment, and distinguish old, current, incremental, and updated values explicitly.
- When discretization derives a solver coefficient from a physical/model parameter and a numerical setting such as the time step, use distinct notation and names for the source parameter and the derived coefficient. State which value the user may tune, which value the solver computes, derive the scaling, and explain what changes physically if the derived value is incorrectly held fixed.
- For every iterative or stateful method, present the complete state transition before showing code: initialization or reset, old state read, increment computation, primary-state update, and accumulator/history update. Implementation code must only translate updates already explained in the prose and equations; it must never introduce a required algorithm step for the first time.
- When a formula modifies an earlier formula or method, explicitly compare old and new forms. Identify every added, removed, reweighted, or accumulated term; explain what failure in the old form motivated each change and what behavior the change creates.
- Explain structural choices that a reader could reasonably question: why a term is in the numerator or denominator, why its sign is positive or negative, why it is squared, normalized, accumulated, clipped, averaged, or scaled by time. Do not stop at saying what the term represents.
- Show why the chosen form is preferable to plausible alternatives. When useful, remove or alter one new term and demonstrate the resulting failure with a minimal numerical example, limiting case, counterexample, or before/after comparison.
- Check boundary and reduction cases. A modified formula should be shown to reduce to the previous method under the appropriate parameter choice, and important extreme parameter regimes should be interpreted physically or algorithmically.
- Put the core design explanation immediately after the formula in visible prose. Folded symbol-by-symbol details may supplement this explanation but must not replace it. For modified formulas, prefer an old-vs-new table and a small worked example; add a local diagram when the design involves multiple interacting terms, stages, or state updates.
- Final self-check: after reading the explanation, a reader should be able to answer both "why is the formula written this way instead of another way?" and "what specifically breaks if this new term is removed or changed?"

## Interactive Code Demos

- Article code demos MUST use the `<PhysicsPlayground preset="name" title="标题" />` component. NEVER paste static Python code blocks or matplotlib scripts for the reader to run locally.
- PhysicsPlayground provides an in-browser Monaco code editor + Pyodide (Python in WebAssembly) + Canvas animation. Readers edit code, click run, and see the animation directly on the page.
- Preset Python files go in `.vitepress/theme/playgrounds/newton-solver/`. They must be **pure Python** (no numpy — Pyodide doesn't load it). Use `import math` and manual list operations.
- The Python code must define a `simulation` global dict: `{"bounds": [xmin,ymin,xmax,ymax], "fps": N, "loop": bool, "frames": [...]}`. Each frame has `"objects"` (circle/segment/polygon) and optional `"metrics"`.
- See existing presets (`free-fall.py`, `fem-jelly-drop.py`, etc.) in that directory for working examples.

## Visual-First Explanations

- For every article-writing or article-editing task, actively look for concepts that can be explained visually. If a relationship, process, lifecycle, state transition, memory layout, index mapping, architecture, comparison, timeline, or spatial structure can be drawn, it must be drawn instead of being left as prose alone.
- Put each diagram next to the first substantial explanation of the concept. A single overview diagram at the beginning or end does not replace local diagrams for major mechanisms.
- Use Mermaid for relationships, flows, architecture, and state transitions. Use generated plots or other suitable images for functions, numerical trends, geometry, and spatial intuition. Follow `.kiro/steering/visualization-rules.md` for detailed requirements.
- Diagrams must carry explanatory information, not decorate the page: label the important objects, arrows, stages, and before/after states, then add a short note telling the reader what to notice.
