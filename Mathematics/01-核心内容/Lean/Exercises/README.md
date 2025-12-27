# Exercises 子项目 | Exercises Subproject

**创建日期**: 2025-12-20
**版本**: v1.0
**状态**: 进行中
**适用范围**: Lean模块 - Exercises子目录

本子目录用于集中管理练习与参考解，并以独立 `lake` 项目形式提供一键构建与验证。

---

## 使用 | Usage

```bash
cd Exercises
lake build
```

提示：`lean-toolchain` 建议与主仓库一致；`lakefile.lean` 推荐将 `mathlib4` 固定到 `stable` 或 `版本同步索引.md` 记录的提交。

### Windows 环境准备 | Windows Setup

1. 安装 elan（Lean 工具链管理）：
   - 访问 `https://leanprover-community.github.io/get_started.html` 并按 Windows 指南安装
2. 安装 lake：
   - 安装后确保 `lake --version` 可用；若不可用，重启 PowerShell 或将用户本地包路径加入 `PATH`
3. 验证：
   - 在任意目录执行 `lean --version` 与 `lake --version`
4. 一键构建：
   - 运行 `./build.ps1`（首次需右键“以 PowerShell 运行”，或在 PowerShell 执行 `Set-ExecutionPolicy -Scope Process Bypass` 后 `./build.ps1`）

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

## 语义练习 | Semantic Exercises

- 入口文档：`Lean/Exercises-说明.md#语义练习--semantic-exercises`
- 代码位置：`Exercises/Semantics/`
  - `UniversePiSigma.lean`：Universe×Π/Σ 与 `Vec/append/get` 骨架
  - `Inference.lean`：极简 `TyVar/TyTerm/Subst` 与 `unify`
- 运行方式：
  - 构建全部：`cd Exercises && lake build`
  - 打开单文件：在编辑器中打开对应 `.lean` 文件进行交互检查

## 验证 | Verification

所有练习都经过 `lake build` 验证，确保代码可编译运行。

## 构建故障排查 | Troubleshooting

- 工具链不匹配：
  - 核对 `Exercises/lean-toolchain` 是否为最新稳定（当前：`v4.12.0`）
  - 运行 `lean --version` 与 `lake --version` 确认本地环境
- mathlib4 版本波动：
  - `lakefile.lean` 已固定到 `stable`，如遇 API 变更请参考 `Lean/版本同步索引.md`
  - 必要时将 `@ "stable"` 临时替换为索引页记录的具体提交哈希
- 导入路径错误：
  - 在 `mathlib4_docs` 搜索目标定理/类型，确认 `import` 路径
  - 常见：`Mathlib/Tactic/Linarith`、`Mathlib/Measure/Integral/IntervalIntegral`
- 缓存问题：
  - 执行 `lake clean` 后重试 `lake build`
  - 删除 `.lake/` 目录后重试（注意会重新下载依赖）

## 每月构建抽检计划 | Monthly Build Spot-check

- 2025-10：抽检 `Analysis/*` 与 `Topology/*`；步骤参见 `Lean/Exercises-说明.md#下月构建抽检计划（2025-10）--next-month-build-spot-check-plan`。

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

## 索引与参考解 | Indexes

- 跨章节练习索引：`Lean/Exercises/跨章节练习索引.md`
- 练习参考解索引：`Lean/Exercises/练习参考解索引.md`

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
