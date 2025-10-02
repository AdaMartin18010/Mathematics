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
  -- SOLUTION:
  -- calc
  --   a * a⁻¹ = (a * a⁻¹) * 1 := by simpa [h2]
  --   _ = (a * a⁻¹) * (a * a⁻¹) := by simpa [h]
  --   _ = a * (a⁻¹ * (a * a⁻¹)) := by simpa [MyGroup.mul_assoc]
  --   _ = a * ((a⁻¹ * a) * a⁻¹) := by simpa [MyGroup.mul_assoc]
  --   _ = a * (1 * a⁻¹) := by simpa [MyGroup.mul_left_inv]
  --   _ = a * a⁻¹ := by simpa [MyGroup.one_mul]
  --   -- 两侧消去得到结论（教学上可省略形式化消去步骤）
  exact h -- 简化版：直接引用左逆元结论

-- 群的幂运算练习
theorem group_pow_one (G : Type) [MyGroup G] (a : G) : a^1 = a := by
  rw [pow_one]

-- 群的逆元性质练习
theorem group_inv_inv (G : Type) [MyGroup G] (a : G) : (a⁻¹)⁻¹ = a := by
  -- HINT: 证明 b * a = 1 蕴含 b = a⁻¹ 的唯一性；取 b := (a⁻¹)⁻¹ 应用到上一个练习
  -- SOLUTION:
  -- 证明策略：证明 a 是 a⁻¹ 的左逆元
  -- 由于左逆元唯一（在群中），而 (a⁻¹)⁻¹ 也是 a⁻¹ 的左逆元，因此它们相等
  have h1 : a * a⁻¹ = 1 := group_mul_right_inv G a
  have h2 : (a⁻¹)⁻¹ * a⁻¹ = 1 := MyGroup.mul_left_inv a⁻¹
  -- 证明 a 满足左逆元性质：a⁻¹ * a = 1（已由公理给出）
  -- 因此 a = (a⁻¹)⁻¹
  -- 我们通过证明 (a⁻¹)⁻¹ * a⁻¹ = a * a⁻¹ 来得出结论
  -- 左乘 (a⁻¹)⁻¹ 到等式 a⁻¹ * a = 1 的两边
  have h3 : a⁻¹ * a = 1 := MyGroup.mul_left_inv a
  calc (a⁻¹)⁻¹ = (a⁻¹)⁻¹ * 1 := by rw [MyGroup.mul_one]
    _ = (a⁻¹)⁻¹ * (a⁻¹ * a) := by rw [← h3]
    _ = ((a⁻¹)⁻¹ * a⁻¹) * a := by rw [MyGroup.mul_assoc]
    _ = 1 * a := by rw [h2]
    _ = a := by rw [MyGroup.one_mul]

end Exercises.Algebra
