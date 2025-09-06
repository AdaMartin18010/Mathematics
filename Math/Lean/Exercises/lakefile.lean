import Lake
open Lake DSL

package exercises

@[default_target]
lean_lib Exercises

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"
