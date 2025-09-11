/-!
# Universe × Pi/Sigma 与依赖向量练习骨架

本文件提供与“依赖×宇宙层级交互”相关的最小可编译练习骨架：
- 显式宇宙与 Π/Σ 层级组合
- 自定义 `Vec`、`append`、`get` 与关键性质骨架

完成提示：将 `sorry` 替换为完整证明。
-/

namespace Exercises.Semantics

set_option autoImplicit true

/- 宇宙与 Π/Σ 层级 -/
universe u v

def PiLevel (A : Sort u) (B : A → Sort v) : Sort (max u v) :=
  (x : A) → B x

structure Sigma' {A : Sort u} (B : A → Sort v) : Sort (max u v) where
  fst : A
  snd : B fst

/- 依赖向量与安全索引 -/
inductive Vec (α : Type) : Nat → Type where
  | nil  : Vec α 0
  | cons : α → {n : Nat} → Vec α n → Vec α (n+1)

open Vec

def append {α} : {m n : Nat} → Vec α m → Vec α n → Vec α (m+n)
  | _, _, nil,        w => w
  | _, _, cons a v,   w => cons a (append v w)

def get {α} {n} : Vec α n → Fin n → α
  | cons a, ⟨0, _⟩      => a
  | cons _ v, ⟨i+1, h⟩  => get v ⟨i, Nat.lt_of_succ_lt_succ h⟩

/- 语义性质（骨架）：对左段索引，连接后的访问等于左向量访问 -/
theorem get_append_left_skeleton {α} {m n} (v : Vec α m) (w : Vec α n) (i : Fin m) :
  get (append v w) ⟨i.val, Nat.lt_trans i.isLt (Nat.lt_add_right _ _ _)⟩ = get v i := by
  -- 练习：使用对 v 的归纳与对 i 的分类讨论完成证明
  sorry

end Exercises.Semantics
