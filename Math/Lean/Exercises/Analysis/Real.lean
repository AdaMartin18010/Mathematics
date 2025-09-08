/-!
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`，版本随 `lakefile.lean` 固定到 stable 或已验证提交
- 最小导入：`import Std`, `import Mathlib`
-/

import Std
import Mathlib

namespace Exercises.Analysis

-- 实数基本性质练习
theorem real_add_comm (a b : ℝ) : a + b = b + a := by
  exact add_comm a b

-- SOLUTION:
-- by
--   simp [add_comm]

-- 实数乘法交换律练习
theorem real_mul_comm (a b : ℝ) : a * b = b * a := by
  exact mul_comm a b

-- SOLUTION:
-- by
--   simp [mul_comm]

-- 实数绝对值性质练习
theorem abs_nonneg (a : ℝ) : 0 ≤ |a| := by
  exact abs_nonneg a

-- SOLUTION:
-- by
--   simpa using abs_nonneg a

-- 实数绝对值三角不等式练习
theorem abs_add_le (a b : ℝ) : |a + b| ≤ |a| + |b| := by
  exact abs_add a b

-- SOLUTION:
-- by
--   simpa using abs_add a b

-- 实数平方非负性练习
theorem sq_nonneg (a : ℝ) : 0 ≤ a^2 := by
  exact sq_nonneg a

-- SOLUTION:
-- by
--   simpa [pow_two] using mul_self_nonneg a

-- 实数平方根性质练习
theorem sqrt_sq (a : ℝ) (ha : 0 ≤ a) : Real.sqrt (a^2) = a := by
  exact Real.sqrt_sq ha

-- SOLUTION:
-- by
--   simpa [pow_two] using Real.sqrt_sq ha

end Exercises.Analysis
