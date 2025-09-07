/-!
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`，版本随 `lakefile.lean` 固定到 stable 或已验证提交
- 最小导入：`import Std`, `import Mathlib`
-/

import Std
import Mathlib
-- 概率论基础练习 | Probability Theory Basic Exercises
-- 对齐国际标准：芝加哥大学概率论课程
-- 更新时间：2025-01-15

import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.Conditional

-- 基础概率论概念练习
namespace Probability

-- 练习1：概率空间性质
-- 对应芝加哥大学概率论基础课程
theorem probability_space_property (Ω : Type*) [MeasurableSpace Ω] (μ : Measure Ω) :
  μ univ = 1 → IsProbabilityMeasure μ := by
  -- HINT: 使用概率基本公理与 `measure_theory` 等价命题；检索 `MeasureTheory.prob_add`/`Prob`
  sorry

-- 练习2：条件概率
-- 对应哈佛大学概率论标准
theorem conditional_probability (A B : Set Ω) [MeasurableSpace Ω] (μ : Measure Ω) :
  μ B ≠ 0 → μ (A ∩ B) / μ B = μ[A|B] := by
  -- HINT: 条件概率定义与乘法公式；检索 `cond`/`MeasureTheory.cond`
  sorry

-- 练习3：独立性
-- 对应剑桥大学Part II概率论
theorem independence_property (A B : Set Ω) [MeasurableSpace Ω] (μ : Measure Ω) :
  Indep A B μ ↔ μ (A ∩ B) = μ A * μ B := by
  -- HINT: 全概率公式；检索 `MeasureTheory.lintegral` 与分割求和
  sorry

-- 练习4：贝叶斯定理
-- 对应华威大学概率论课程
theorem bayes_theorem (A B : Set Ω) [MeasurableSpace Ω] (μ : Measure Ω) :
  μ B ≠ 0 → μ A ≠ 0 →
  μ[B|A] = μ[A|B] * μ B / μ A := by
  -- HINT: 贝叶斯公式；将条件概率改写为联合/边缘概率比值
  sorry

-- 练习5：期望性质
-- 对应巴黎第六大学概率论标准
theorem expectation_linearity (X Y : Ω → ℝ) [MeasurableSpace Ω] (μ : Measure Ω) :
  Integrable X μ → Integrable Y μ →
  ∫ x, (X x + Y x) ∂μ = ∫ x, X x ∂μ + ∫ x, Y x ∂μ := by
  -- HINT: 期望线性性 `lintegral_add`/`integral_add`；注意可积性前提
  sorry

-- 练习6：方差性质
-- 对应伦敦大学学院概率论课程
theorem variance_property (X : Ω → ℝ) [MeasurableSpace Ω] (μ : Measure Ω) :
  Var X = ∫ x, (X x - ∫ y, X y ∂μ) ^ 2 ∂μ := by
  -- HINT: 方差定义 `Var = E[X^2] - (E[X])^2`；结合期望线性性
  sorry

-- 练习7：大数定律
-- 对应国际标准：Law of Large Numbers
theorem law_of_large_numbers (X : ℕ → Ω → ℝ) [MeasurableSpace Ω] (μ : Measure Ω) :
  (∀ n, Integrable (X n) μ) →
  (∀ n, ∫ x, X n x ∂μ = μ) →
  Tendsto (fun n => (∑ i in Finset.range n, X i) / n) atTop (𝓝 μ) := by
  -- HINT: 切比雪夫不等式；检索 `chebyshev` 相关定理
  sorry

-- 练习8：中心极限定理
-- 对应国际标准：Central Limit Theorem
theorem central_limit_theorem (X : ℕ → Ω → ℝ) [MeasurableSpace Ω] (μ : Measure Ω) :
  (∀ n, Integrable (X n) μ) →
  (∀ n, ∫ x, X n x ∂μ = 0) →
  (∀ n, Var (X n) = σ ^ 2) →
  Tendsto (fun n => (∑ i in Finset.range n, X i) / sqrt n) atTop (𝓝 (Gaussian 0 σ)) := by
  -- HINT: 大数定律（弱/强）；引用现成定理或给出特例证明路线
  sorry

end Probability
