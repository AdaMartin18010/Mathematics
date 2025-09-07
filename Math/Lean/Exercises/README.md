# Exercises 子项目 | Exercises Subproject

本子目录用于集中管理练习与参考解，并以独立 `lake` 项目形式提供一键构建与验证。

## 使用 | Usage

```bash
cd Exercises
lake build
```

提示：`lean-toolchain` 建议与主仓库一致；`lakefile.lean` 推荐将 `mathlib4` 固定到 `stable` 或 `版本同步索引.md` 记录的提交。

## 结构 | Structure

```text
Exercises/
├─ lean-toolchain
├─ lakefile.lean
├─ README.md
├─ Basics/
│   └─ AddComm.lean
├─ Algebra/
│   ├─ Group.lean
│   ├─ Ring.lean
│   └─ Field.lean
├─ Analysis/
│   ├─ Real.lean
│   └─ Complex.lean
├─ Geometry/
│   └─ Euclidean.lean
├─ NumberTheory/
│   └─ Basic.lean
├─ Probability/
│   └─ Basic.lean
├─ Topology/
│   └─ Basic.lean
├─ CategoryTheory/
│   └─ Basic.lean
└─ MeasureTheory/
    └─ Basic.lean
```

## 练习分类 | Exercise Categories

- **Basics/**: 基础语法与类型系统练习
- **Algebra/**: 代数结构练习（群、环、域）
- **Analysis/**: 分析学练习（实数分析、复分析）
- **Geometry/**: 几何学练习（欧几里得几何）
- **NumberTheory/**: 数论练习（基础数论）
- **Probability/**: 概率论练习（基础概率论）
- **Topology/**: 拓扑学练习（基础拓扑）
- **CategoryTheory/**: 范畴论练习（基础范畴论）
- **MeasureTheory/**: 测度论练习（基础测度论）

## 验证 | Verification

所有练习都经过 `lake build` 验证，确保代码可编译运行。

## 按课程路径练习清单 | Course-aligned Exercise Map

- 基础入门（对应 Stage 1，Xena 入门）
  - `Basics/AddComm.lean`、`Basics/` 其他入门题

- 证明系统（对应 Stage 3）
  - 在可构建的项目中运行：配合 `03-证明系统与策略/01-交互式证明环境.md`

- 代数方向（Xena/Algebra 对齐）
  - `Algebra/Group.lean`、`Algebra/Ring.lean`、`Algebra/Field.lean`

- 分析方向（Real/Complex 基础）
  - `Analysis/Real.lean`、`Analysis/Complex.lean`

- 拓扑/测度/概率/范畴与数论（扩展路径）
  - `Topology/Basic.lean`、`MeasureTheory/Basic.lean`、`Probability/Basic.lean`、`CategoryTheory/Basic.lean`、`NumberTheory/Basic.lean`

说明：如遇策略或 API 变动，请先参阅 `Lean/版本同步索引.md` 与各文档“版本与兼容性注记”。

## 参考条目与外部链接 | Reference Index

- 文档与手册：
  - Lean 4 手册：`https://leanprover.github.io/lean4/doc/`
  - Theorem Proving in Lean 4：`https://leanprover.github.io/theorem_proving_in_lean4/`
  - mathlib4 文档：`https://leanprover-community.github.io/mathlib4_docs/`

- 常用策略与决策：
  - tactics 速览：`https://leanprover-community.github.io/tactics.html`
  - 线性算术 `linarith`：mathlib4_docs 检索“linarith”
  - 多项式 `ring`：mathlib4_docs 检索“ring”
  - 判定 `decide`：mathlib4_docs 检索“decide”

- 课程与项目：
  - Xena Project（帝国理工）：`https://xenaproject.wordpress.com/`
  - Xena 代码仓库：`https://github.com/ImperialCollegeLondon/xena`

## docs 检索技巧 | Docs Search Tips

- 关键词定位：在 `mathlib4_docs` 站内直接搜索英文关键词/lemma 名，如 “linarith”, “add_comm”, “Nat.gcd_rec”。
- 模块路径：不清楚导入路径时，先搜索类型或定理名，进入页面顶部可见模块路径，按需补充 `import`。
- 组合检索：先搜策略（如 “ring”），再在结果页中用浏览器内查找目标符号（如 “pow”）。
- 示例查询：
  - 线性算术：搜索 “linarith” → 查看使用条件与示例
  - 多项式环运算：搜索 “ring” → 核对需要的代数结构实例
  - 互素与 gcd：搜索 “coprime” / “gcd_eq_one” → 对照数论练习

---

更新时间：2025-09（新增课程路径练习清单与版本对齐提示）
