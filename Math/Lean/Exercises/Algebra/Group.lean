/-!
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`，版本随 `lakefile.lean` 固定到 stable 或已验证提交
- 最小导入：`import Std`, `import Mathlib`
-/

import Std
import Mathlib

namespace Exercises.Algebra

-- 自定义群结构（避免与Mathlib冲突）
class MyGroup (G : Type) extends Mul G, One G, Inv G where
  mul_assoc : ∀ a b c : G, (a * b) * c = a * (b * c)
  one_mul : ∀ a : G, 1 * a = a
  mul_one : ∀ a : G, a * 1 = a
  mul_left_inv : ∀ a : G, a⁻¹ * a = 1

-- 群的基本性质练习
theorem group_mul_right_inv (G : Type) [MyGroup G] (a : G) : a * a⁻¹ = 1 := by
  have h := MyGroup.mul_left_inv a
  have h1 := MyGroup.mul_assoc a a⁻¹ a
  have h2 := MyGroup.mul_one a
  -- HINT: 从右侧乘以 a⁻¹，结合结合律与左逆元性质；再使用右单位元
  sorry -- 这是一个练习，需要学生完成

-- 群的幂运算练习
theorem group_pow_one (G : Type) [MyGroup G] (a : G) : a^1 = a := by
  rw [pow_one]

-- 群的逆元性质练习
theorem group_inv_inv (G : Type) [MyGroup G] (a : G) : (a⁻¹)⁻¹ = a := by
  -- HINT: 证明 b * a = 1 蕴含 b = a⁻¹ 的唯一性；取 b := (a⁻¹)⁻¹ 应用到上一个练习
  sorry -- 这是一个练习，需要学生完成

end Exercises.Algebra
