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
  └─ Analysis/
      └─ Real.lean
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

---

## 版本与工具链对齐 | Version & Toolchain Alignment

- 工具链：`Exercises/lean-toolchain` 应与主仓库保持一致，建议跟随最新稳定版。
- mathlib4：在 `lakefile.lean` 固定到 `stable` 或 `版本同步索引.md` 中记录的已验证提交。
- 同步流程：每月对照 `Lean/版本同步索引.md`，如有 breaking changes，先在 `Exercises` 内验证再更新示例。
