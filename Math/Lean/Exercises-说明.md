# Exercises 子项目说明 | Exercises Subproject

本子目录用于集中管理练习与参考解，并以独立 `lake` 项目形式提供一键构建与验证。

## 结构

```text
Exercises/
  ├─ lean-toolchain
  ├─ lakefile.lean
  ├─ README.md
  ├─ Basics/
  │   └─ AddComm.lean
  ├─ Algebra/
  │   └─ Group.lean
  ├─ Analysis/
  │   └─ Real.lean
  └─ Semantics/
      ├─ UniversePiSigma.lean
      └─ Inference.lean
```

## 最小文件内容示例

```text
Exercises/lean-toolchain      # 与主项目一致的工具链版本
Exercises/lakefile.lean       # Lake 配置（见下）
Exercises/README.md           # 使用说明
```

```lean
-- Exercises/lakefile.lean
import Lake
open Lake DSL

package exercises

@[default_target]
lean_lib Exercises

-- 固定 mathlib4 至已验证提交，避免 master 波动导致不稳定
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "stable"
```

```lean
-- Exercises/Basics/AddComm.lean
namespace Exercises.Basics

theorem add_comm_ex (a b : Nat) : a + b = b + a := by
  exact Nat.add_comm a b

end Exercises.Basics
```

```markdown
    # Exercises/README.md

    ## 使用
    ```bash
    cd Exercises
    lake build
    ```

```

## 运行指引 | How to Run

- 构建全部练习：
  - 在仓库根目录：`cd Exercises && lake build`
- 仅运行语义练习：
  - `lake build Exercises:Semantics`
- 查看单文件（VS Code）：
  - 打开 `Exercises/Semantics/UniversePiSigma.lean` 或 `Exercises/Semantics/Inference.lean`

---

## 版本与工具链对齐 | Version & Toolchain Alignment

- 工具链：`Exercises/lean-toolchain` 应与主仓库保持一致，建议跟随最新稳定版。
- mathlib4：在 `lakefile.lean` 固定到 `stable` 或 `版本同步索引.md` 中记录的已验证提交。
- 同步流程：每月对照 `Lean/版本同步索引.md`，如有 breaking changes，先在 `Exercises` 内验证再更新示例。

---

## 下月构建抽检计划（2025-10） | Next-month Build Spot-check Plan

- 抽检范围：`Exercises/Analysis/*` 与 `Exercises/Topology/*`
- 工具链：按 `Lean/版本同步索引.md` 的最新稳定版本
- 步骤：
  1) `cd Exercises && lake build` 全量构建
  2) 若失败，先固定 `mathlib4` 至索引页记录提交后重试
  3) 记录问题与修复在 `Lean/版本同步索引.md` 的“当月记录”
  4) 回归验证：最小示例与受影响文档片段

## 语义练习 | Semantic Exercises

- 依赖×宇宙层级交互（从入门到进阶）
  - 阅读：`Lean/08-Lean4语义分析论证系统/02-Lean4语言语义深度分析/02-类型系统语义分析.md#依赖类型宇宙层级交互--dependent-types--universe-levels`
  - 总览：`Lean/08-Lean4语义分析论证系统/01-语义分析基础理论/01-语义分析总览.md`
  - 实践：在 `Exercises/Semantics/UniversePiSigma.lean` 中实现：
    1) 定义 Π/Σ 的层级保持示例（使用 `universe u v` 与 `max u v`）
    2) 构造 `Vec`、`append`、`get` 并证明左段索引的语义保持（骨架可参考文档）
  - 回链：完成后在本页“完成打勾”并在实践指南中记录心得

- 类型推断语义与统一
  - 阅读：`Lean/08-Lean4语义分析论证系统/02-Lean4语言语义深度分析/02-类型系统语义分析.md#-类型推断语义分析--type-inference-semantic-analysis`
  - 实践：在 `Exercises/Semantics/Inference.lean` 中构造约束→替换的小型统一器接口与用例

- 语法↔语义对应关系
  - 阅读：`Lean/08-Lean4语义分析论证系统/02-Lean4语言语义深度分析/01-语法语义对应关系.md`
  - 实践：从 `表达式→类型→证明` 的三段式小练习，确保每步可 #check / #eval

- Coercions 与子类型
  - 阅读：`Lean/08-Lean4语义分析论证系统/02-Lean4语言语义深度分析/02-类型系统语义分析.md#15-coercions与子类型语义--coercions--subtyping-semantics`
  - 实践：`Exercises/Semantics/CoercionsSubtype.lean` 完成 Age/Fin 投影与等式保持练习

- 类型类解析与实例搜索
  - 阅读：同上文档 `1.4 类型类推导语义`
  - 实践：`Exercises/Semantics/TypeclassResolution.lean` 完成不同实例下的行为验证

### 完成打勾 | Completion Checklist

- [ ] Universe×Pi/Sigma 示例可编译
- [ ] `Vec.append`/`get` 证明通过（或有最小 `admit` 占位）
- [ ] 类型推断与统一用例通过
- [ ] 所有练习 `lake build` 成功
- [ ] （可选）统一器支持 occurs-check 并含示例

---

## 相关文档 | Related Docs

- 实践指南：`Lean/02-基础语法与类型系统/06-Lean4最新特性实践指南.md`
- 语义总览：`Lean/08-Lean4语义分析论证系统/01-语义分析基础理论/01-语义分析总览.md`
