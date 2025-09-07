/-!
复分析练习 | Complex Analysis Exercises
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`（已在 lakefile 固定稳定版本或已验证提交）
-/

import Std
import Mathlib
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.Complex.Trigonometric

namespace ComplexAnalysisExercises

-- 复数基本性质练习
theorem complex_add_comm (z w : ℂ) : z + w = w + z := by
  exact add_comm z w

-- 复数乘法交换律练习
theorem complex_mul_comm (z w : ℂ) : z * w = w * z := by
  exact mul_comm z w

-- 复数模的性质练习
theorem complex_abs_mul (z w : ℂ) : |z * w| = |z| * |w| := by
  exact abs_mul z w

-- 复数共轭的性质练习
theorem complex_conj_mul (z w : ℂ) : (z * w).conj = z.conj * w.conj := by
  exact conj_mul z w

-- 欧拉公式练习
theorem euler_formula (θ : ℝ) : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.sin θ * Complex.I := by
  exact Complex.exp_mul_I θ

-- 复数的极坐标形式练习
theorem complex_polar_form (z : ℂ) (hz : z ≠ 0) :
  ∃ r : ℝ, ∃ θ : ℝ, z = r * Complex.exp (θ * Complex.I) := by
  -- HINT: 检索 `Complex.exists_polar`/`Complex.abs_arg` 相关定理；r 可取 `|z|`
  sorry

-- 复数的幂运算练习
theorem complex_pow_nat (z : ℂ) (n : ℕ) : z^n = (|z|^n) * Complex.exp (n * Complex.arg z * Complex.I) := by
  -- HINT: 使用极坐标与幂的性质；检索 `Complex.cpow`, `isROrC` 相关引理
  sorry

-- 复数的根运算练习
theorem complex_nth_root (z : ℂ) (n : ℕ) (hn : n ≠ 0) :
  ∃ w : ℂ, w^n = z := by
  -- HINT: 用代数闭包或根存在性结果；检索 `Complex.exists_mul_self`/`nthRoots`
  sorry

-- 复数的对数练习
theorem complex_log_property (z : ℂ) (hz : z ≠ 0) :
  Complex.exp (Complex.log z) = z := by
  -- HINT: 参考 `Complex.exp_log`（非零条件）
  sorry

-- 复数的三角函数练习
theorem complex_sin_cos (θ : ℝ) :
  Complex.sin θ = (Complex.exp (θ * Complex.I) - Complex.exp (-θ * Complex.I)) / (2 * Complex.I) := by
  -- HINT: 使用欧拉公式 `Complex.exp_mul_I`、三角函数定义与恒等式
  sorry

end ComplexAnalysisExercises
