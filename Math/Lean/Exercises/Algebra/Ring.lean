-- 环论练习 | Ring Theory Exercises
-- 对齐国际标准：剑桥大学Part II代数课程
-- 更新时间：2025-01-15

/-!
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`，版本随 `lakefile.lean` 固定到 stable 或已验证提交
- 最小导入：`import Std`, `import Mathlib`
-/

import Std
import Mathlib

namespace Exercises.Algebra

-- 自定义环结构（避免与Mathlib冲突）
class MyRing (R : Type) extends Add R, Mul R, Zero R, One R, Neg R where
  add_assoc : ∀ a b c : R, (a + b) + c = a + (b + c)
  add_comm : ∀ a b : R, a + b = b + a
  add_zero : ∀ a : R, a + 0 = a
  add_left_neg : ∀ a : R, -a + a = 0
  mul_assoc : ∀ a b c : R, (a * b) * c = a * (b * c)
  mul_one : ∀ a : R, a * 1 = a
  one_mul : ∀ a : R, 1 * a = a
  left_distrib : ∀ a b c : R, a * (b + c) = a * b + a * c
  right_distrib : ∀ a b c : R, (a + b) * c = a * c + b * c

-- 环的基本性质练习
theorem ring_mul_zero (R : Type) [MyRing R] (a : R) : a * 0 = 0 := by
  have h1 := MyRing.left_distrib a 0 0
  have h2 := MyRing.add_zero (a * 0)
  have h3 := MyRing.add_zero 0
  -- HINT: 使用分配律将 a*(0+0) 与 a*0 + a*0 比较，再移项
  -- SOLUTION:
  -- a*(0+0) = a*0 + a*0（分配律）且 0+0=0，因此 a*0 = a*0 + a*0，移项得 a*0=0
  have : a * (0 + 0) = a * 0 + a * 0 := by simpa using h1
  simpa using this

-- 环的幂运算练习
theorem ring_pow_zero (R : Type) [MyRing R] (a : R) : a^0 = 1 := by
  rw [pow_zero]

-- 环的负元性质练习
theorem ring_neg_mul (R : Type) [MyRing R] (a b : R) : (-a) * b = -(a * b) := by
  -- HINT: 用 (-a)*b + a*b = 0 证明，并用加法消去得到等式
  -- SOLUTION: 经典做法是证明两边相加为 0；此处留作进阶
  admit

-- 环的分配律练习
theorem ring_distrib_left (R : Type) [MyRing R] (a b c : R) : a * (b + c) = a * b + a * c := by
  exact MyRing.left_distrib a b c

-- 环的分配律练习
theorem ring_distrib_right (R : Type) [MyRing R] (a b c : R) : (a + b) * c = a * c + b * c := by
  exact MyRing.right_distrib a b c

-- 环的幂运算结合律练习
theorem ring_pow_add (R : Type) [MyRing R] (a : R) (m n : ℕ) : a^(m + n) = a^m * a^n := by
  induction n with
  | zero => simp [pow_zero, mul_one]
  | succ k ih => simp [pow_succ, ih, mul_assoc]

end Exercises.Algebra
