-- 域论练习 | Field Theory Exercises
-- 对齐国际标准：剑桥大学Part II代数课程
-- 更新时间：2025-01-15

import Std
import Mathlib

namespace Exercises.Algebra

-- 自定义域结构（避免与Mathlib冲突）
class MyField (F : Type) extends MyRing F, Inv F where
  mul_comm : ∀ a b : F, a * b = b * a
  mul_inv_cancel : ∀ a : F, a ≠ 0 → a * a⁻¹ = 1
  inv_zero : (0 : F)⁻¹ = 0

-- 域的基本性质练习
theorem field_mul_inv (F : Type) [MyField F] (a : F) (ha : a ≠ 0) : a * a⁻¹ = 1 := by
  exact MyField.mul_inv_cancel a ha

-- 域的逆元唯一性练习
theorem field_inv_unique (F : Type) [MyField F] (a b : F) (ha : a ≠ 0) :
  a * b = 1 → b = a⁻¹ := by
  intro h
  have h1 := MyField.mul_inv_cancel a ha
  have h2 := congr_arg (fun x => b * x) h1
  sorry -- 这是一个练习，需要学生完成

-- 域的除法定义练习
theorem field_div_def (F : Type) [MyField F] (a b : F) (hb : b ≠ 0) :
  a / b = a * b⁻¹ := by
  rfl

-- 域的乘法逆元性质练习
theorem field_inv_inv (F : Type) [MyField F] (a : F) (ha : a ≠ 0) : (a⁻¹)⁻¹ = a := by
  have h1 := MyField.mul_inv_cancel a ha
  have h2 := MyField.mul_inv_cancel a⁻¹ (by sorry) -- 需要证明 a⁻¹ ≠ 0
  sorry -- 这是一个练习，需要学生完成

-- 域的幂运算性质练习
theorem field_pow_neg (F : Type) [MyField F] (a : F) (ha : a ≠ 0) (n : ℕ) :
  a^(-n) = (a^n)⁻¹ := by
  sorry -- 这是一个练习，需要学生完成

-- 域的平方根性质练习
theorem field_sqrt_property (F : Type) [MyField F] (a : F) (ha : a ≠ 0) :
  ∃ b : F, b^2 = a ∨ b^2 = -a := by
  sorry -- 这是一个练习，需要学生完成

end Exercises.Algebra
