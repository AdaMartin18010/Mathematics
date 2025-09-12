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

def compose (σ τ : Subst) : Subst :=
  { map := fun v => (apply τ (σ.map v)) }

end Subst

def apply (σ : Subst) : TyTerm → TyTerm
  | .var v     => σ.map v
  | .const c   => .const c
  | .app f a   => .app (apply σ f) (apply σ a)

-- occurs-check：变量 v 是否出现在项 t 中（防止 v = t[v] 自指导致无限类型）
def occurs (v : TyVar) : TyTerm → Bool
  | .var v' => decide (v' = v)
  | .const _ => false
  | .app f a => occurs v f || occurs v a

partial def unify : TyTerm → TyTerm → Option Subst
  | .var v1, .var v2 =>
      if v1 = v2 then some Subst.id else some (Subst.single v1 (.var v2))
  | .const c1, .const c2 => if c1 = c2 then some Subst.id else none
  | .app f1 a1, .app f2 a2 => do
      let σ1 ← unify f1 f2
      let σ2 ← unify (apply σ1 a1) (apply σ1 a2)
      -- 合成：先应用 σ1 再应用 σ2
      let σ : Subst := Subst.compose σ1 σ2
      pure σ
  | .var v, t =>
      -- occurs-check：v 不得出现在 t 中
      if occurs v t then none else some (Subst.single v t)
  | t, .var v =>
      if occurs v t then none else some (Subst.single v t)
  | _, _ => none

/- 示例：
   unify (app (const "List") (var "α")) (app (const "List") (const "Nat"))
   期望得到 { α ↦ Nat } -/

def α : TyVar := .mk "α"
def β : TyVar := .mk "β"

def ex1 := unify (.app (.const "List") (.var α)) (.app (.const "List") (.const "Nat"))

#eval ex1

-- 快速可见性：是否统一成功
#eval Option.isSome ex1

-- 合成与应用示例
def σ : Subst := Subst.single α (.const "Nat")
def τ : Subst := Subst.single β (.const "Bool")
def term : TyTerm := .app (.app (.const "Pair") (.var α)) (.var β)

-- 期望：Pair Nat Bool
#eval apply (Subst.compose σ τ) term

-- 非可统一示例（常量不同）：应为 none
def exFail := unify (.const "List") (.const "Nat")
#eval Option.isNone exFail

-- occurs-check 示例：期望 none（v 出现在 app 中）
def exOccurs := unify (.var α) (.app (.const "F") (.var α))
#eval Option.isNone exOccurs

end Exercises.Semantics
