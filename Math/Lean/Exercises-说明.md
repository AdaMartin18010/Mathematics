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

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"
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
