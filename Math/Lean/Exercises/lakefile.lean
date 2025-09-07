import Lake
open Lake DSL

package exercises

@[default_target]
lean_lib Exercises

-- 建议将 mathlib4 固定到稳定分支或已验证提交，避免 master 波动
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "stable"
