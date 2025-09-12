/-!
# 类型类解析与实例搜索练习骨架

目标：构造最小 `Monoid` 类与实例，使用实例搜索完成折叠函数，并验证行为。
-/

namespace Exercises.Semantics

class Monoid (α : Type) where
  ident : α
  op    : α → α → α

instance : Monoid Nat where
  ident := 0
  op := Nat.add

def foldMonoid {α} [m : Monoid α] : List α → α
  | []      => m.ident
  | a :: as => m.op a (foldMonoid as)

#eval foldMonoid [1,2,3,4]  -- 10

-- 布尔上的 Monoid（或）
instance : Monoid Bool where
  ident := false
  op := fun a b => a || b

#eval foldMonoid [true, false, true]  -- true

end Exercises.Semantics
