/-!
# Coercions 与子类型练习骨架

目标：演示 `Subtype`/`Fin` 到宿主类型的安全提升（指称保持），并给出等式保持的练习。
-/

namespace Exercises.Semantics

structure Age where
  val : Nat
  prf : val ≤ 150

-- 简化 Coe：最小自定义接口（为避免与内建冲突，使用 Coee 名称）
class Coee (α : Type) (β : Type) where
  coe : α → β

instance : Coee Age Nat where
  coe a := a.val

def celebrate (n : Nat) := n + 1
def celebrateSafe (a : Age) := celebrate (Coee.coe a)

theorem coercion_ext {f : Nat → Nat} (a : Age) :
  f (Coee.coe a) = f a.val := rfl

-- Fin 到 Nat 的投影与语义保持
instance : Coee (Fin n) Nat where
  coe i := i.val

inductive Vec (α : Type) : Nat → Type where
  | nil : Vec α 0
  | cons : α → {n : Nat} → Vec α n → Vec α (n+1)

open Vec

def get {α} {n} : Vec α n → Fin n → α
  | cons a, ⟨0, _⟩      => a
  | cons _ v, ⟨i+1, h⟩  => get v ⟨i, Nat.lt_of_succ_lt_succ h⟩

theorem get_safe_eq (v : Vec α n) (i : Fin n) :
  get v i = get v ⟨(Coee.coe i), i.isLt⟩ := rfl

end Exercises.Semantics
