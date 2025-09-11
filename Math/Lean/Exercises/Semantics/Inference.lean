/-!
# 类型推断与统一最小骨架

目标：构造极简“类型项/变量/替换”与统一器接口骨架，并给出两个可编译示例。
-/

namespace Exercises.Semantics

inductive TyVar where
  | mk : String → TyVar
  deriving DecidableEq, Repr

inductive TyTerm where
  | var : TyVar → TyTerm
  | const : String → TyTerm
  | app : TyTerm → TyTerm → TyTerm
  deriving DecidableEq, Repr

structure Subst where
  map : TyVar → TyTerm

namespace Subst

def id : Subst := { map := fun v => TyTerm.var v }

def single (v : TyVar) (t : TyTerm) : Subst :=
  { map := fun v' => if v' = v then t else TyTerm.var v' }

end Subst

def apply (σ : Subst) : TyTerm → TyTerm
  | .var v     => σ.map v
  | .const c   => .const c
  | .app f a   => .app (apply σ f) (apply σ a)

partial def unify : TyTerm → TyTerm → Option Subst
  | .var v1, .var v2 =>
      if v1 = v2 then some Subst.id else some (Subst.single v1 (.var v2))
  | .const c1, .const c2 => if c1 = c2 then some Subst.id else none
  | .app f1 a1, .app f2 a2 => do
      let σ1 ← unify f1 f2
      let σ2 ← unify (apply σ1 a1) (apply σ1 a2)
      -- 简化：合成以右侧覆盖为主
      let σ : Subst := { map := fun v => apply σ2 (σ1.map v) }
      pure σ
  | _, _ => none

/- 示例：
   unify (app (const "List") (var "α")) (app (const "List") (const "Nat"))
   期望得到 { α ↦ Nat } -/

def α : TyVar := .mk "α"

def ex1 := unify (.app (.const "List") (.var α)) (.app (.const "List") (.const "Nat"))

#eval ex1

end Exercises.Semantics
