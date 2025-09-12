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
  -- 对 v 做归纳，并对 i 分情况讨论
  induction v with
  | nil =>
      -- m = 0，Fin 0 无元素，矛盾消解
      cases i with
      | mk _ h => cases h
  | @cons a m' v ih =>
      -- 此时 m = m' + 1
      cases i with
      | mk idx h =>
          cases idx with
          | zero =>
              -- i = 0 情况
              -- LHS 与 RHS 都应化约为 a
              simp [append, get]
          | succ k =>
              -- i = k+1 情况，构造子问题的索引 i'
              have hk : k < m' := Nat.lt_of_succ_lt_succ h
              have i' : Fin m' := ⟨k, hk⟩
              -- 两侧分别按定义化约，并套用归纳假设
              simp [append, get, ih, i']

end Exercises.Semantics
