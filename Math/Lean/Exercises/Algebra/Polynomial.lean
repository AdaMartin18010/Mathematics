-- 多项式环理论练习 | Polynomial Ring Theory Exercises
-- 对齐国际标准：MIT 18.701、Stanford Math 120、Cambridge Part II
-- 更新时间：2025-10-01

/-!
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`，版本随 `lakefile.lean` 固定到 stable 或已验证提交
- 最小导入：`import Std`, `import Mathlib`
-/

import Std
import Mathlib

namespace Exercises.Algebra

-- ============================================
-- 多项式环基础定理（使用mathlib4标准定义）
-- ============================================

-- 带余除法（唯一性）
theorem div_mod_by_monic_unique {F : Type*} [Field F]
  (f g : Polynomial F) (hg : g.Monic) :
  ∃! (q r : Polynomial F), f = g * q + r ∧ r.degree < g.degree := by
  -- 使用mathlib4的Polynomial.divModByMonicUnique
  exact Polynomial.divModByMonicUnique f g hg

-- 余式定理
theorem eval_mod_by_monic_X_sub_C {F : Type*} [Field F]
  (f : Polynomial F) (a : F) :
  f.eval a = (f %ₘ (Polynomial.X - Polynomial.C a)).eval a := by
  -- 使用mathlib4的Polynomial.modByMonic_X_sub_C_eq_C_eval
  exact Polynomial.modByMonic_X_sub_C_eq_C_eval f a

-- Euclid算法（GCD存在性）
theorem gcd_exists {F : Type*} [Field F] (f g : Polynomial F) :
  ∃ d : Polynomial F, d.Monic ∧
    (∀ p : Polynomial F, p ∣ f → p ∣ g → p ∣ d) ∧
    (∃ u v : Polynomial F, d = u * f + v * g) := by
  -- 使用mathlib4的Polynomial.gcd和Polynomial.gcd_eq_gcd_ab
  -- 构造首一最大公因式
  let d := Polynomial.gcd f g
  -- 归一化d使其首一
  -- 如果d = 0，则d已经是首一的（零多项式是首一的）
  by_cases h_zero : d = 0
  · -- d = 0的情况
    use 0, Polynomial.monic_zero
    constructor
    · intro p hp_f hp_g
      exact Polynomial.dvd_zero p
    · use 0, 0
      simp [h_zero]
  · -- d ≠ 0的情况
    -- 归一化d使其首一：d / leadingCoeff d
    let d_normalized := Polynomial.C d.leadingCoeff⁻¹ * d
    have h_monic : d_normalized.Monic := by
      exact Polynomial.monic_mul_leadingCoeff_inv h_zero
    -- 证明归一化后的多项式仍然满足最大公因式的性质
    have h_dvd_normalized : d ∣ d_normalized := by
      -- d_normalized = (leadingCoeff d)⁻¹ * d
      -- 由于leadingCoeff d ≠ 0，d ∣ d_normalized
      use Polynomial.C d.leadingCoeff⁻¹
      ring
    have h_normalized_dvd : d_normalized ∣ d := by
      -- d = leadingCoeff d * d_normalized
      use Polynomial.C d.leadingCoeff
      ring
    -- 使用Polynomial.gcd_eq_gcd_ab
    have h_bezout : ∃ u v : Polynomial F, d = u * f + v * g := by
      exact Polynomial.gcd_eq_gcd_ab f g
    -- 从h_bezout推导出d_normalized的Bézout等式
    obtain ⟨u, v, h_bezout_d⟩ := h_bezout
    have h_bezout_normalized : ∃ u' v' : Polynomial F, d_normalized = u' * f + v' * g := by
      use Polynomial.C d.leadingCoeff⁻¹ * u, Polynomial.C d.leadingCoeff⁻¹ * v
      rw [← h_bezout_d]
      ring
    use d_normalized, h_monic
    constructor
    · -- 最大公因式的性质
      intro p hp_f hp_g
      -- 由gcd的定义，p ∣ gcd f g = d
      have h_p_dvd_d : p ∣ d := Polynomial.dvd_gcd hp_f hp_g
      -- 由于d ∣ d_normalized，因此p ∣ d_normalized
      exact dvd_trans h_p_dvd_d h_dvd_normalized
    · -- Bézout等式
      exact h_bezout_normalized

-- 唯一分解
theorem unique_factorization {F : Type*} [Field F] (f : Polynomial F) (hf : f ≠ 0) :
  ∃ (c : F) (factors : Multiset (Polynomial F)),
    (∀ p ∈ factors, Irreducible p ∧ p.Monic) ∧
    f = c • factors.prod := by
  -- F[x]是UFD，使用mathlib4的唯一分解定理
  -- 使用UniqueFactorizationMonoid.factors来获取不可约因子
  -- 首先，Polynomial F是UniqueFactorizationMonoid（因为F是域）
  have h_ufd : UniqueFactorizationMonoid (Polynomial F) := inferInstance
  -- 使用factors来分解f
  obtain ⟨factors, h_factors_prime, h_factors_prod⟩ := UniqueFactorizationMonoid.exists_prime_factors hf
  -- factors是素因子的多重集合，每个因子都是素元（在UFD中，素元等价于不可约元）
  -- 需要将factors转换为Monic不可约多项式
  -- 对于每个素因子p，我们可以将其归一化为首一多项式
  let factors_monic : Multiset (Polynomial F) := factors.map fun p =>
    (p * Polynomial.C p.leadingCoeff⁻¹)
  -- 证明factors_monic中的每个元素都是首一且不可约的
  have h_monic_irreducible : ∀ p ∈ factors_monic, Irreducible p ∧ p.Monic := by
    intro p hp
    obtain ⟨q, hq_mem, rfl⟩ := Multiset.mem_map.mp hp
    constructor
    · -- 不可约性：q是素元，因此不可约；乘以单位后仍不可约
      have h_prime : Prime q := h_factors_prime q hq_mem
      have h_irreducible : Irreducible q := Prime.irreducible h_prime
      -- q * C(q.leadingCoeff⁻¹)与q关联，因此不可约
      have h_associated : Associated q (q * Polynomial.C q.leadingCoeff⁻¹) := by
        use Polynomial.C q.leadingCoeff⁻¹
        constructor
        · exact isUnit_C.mpr (isUnit_iff_ne_zero.mpr (inv_ne_zero (leadingCoeff_ne_zero.mpr (Prime.ne_zero h_prime))))
        · ring
      exact h_associated.irreducible h_irreducible
    · -- 首一性
      exact Polynomial.monic_mul_leadingCoeff_inv (Prime.ne_zero (h_factors_prime q hq_mem))
  -- 计算常数因子c
  -- f = (factors.prod) = (factors_monic.prod) * (某个常数)
  -- 需要提取leadingCoeff
  use f.leadingCoeff
  use factors_monic
  constructor
  · exact h_monic_irreducible
  · -- 证明f = c • factors_monic.prod
    -- 这需要从h_factors_prod推导
    -- 简化处理：使用factors和factors_monic的关系
    -- 实际上，factors_monic.prod = factors.prod * (某个常数)
    -- 而f = factors.prod（在关联意义下）
    -- 因此f = c • factors_monic.prod，其中c = f.leadingCoeff / factors_monic.prod.leadingCoeff
    -- 由于factors_monic都是首一的，factors_monic.prod也是首一的
    -- 因此c = f.leadingCoeff
    -- 简化：直接使用h_factors_prod和factors_monic的定义
    -- 这里需要更仔细的证明，但基本思路是正确的
    -- 标记为TODO，需要完善细节
    sorry -- TODO: 完善常数因子的计算和等式证明

-- 根的个数
theorem card_roots_le_degree {F : Type*} [Field F] (f : Polynomial F) :
  (f.roots.toFinset.card : ℕ) ≤ f.natDegree := by
  -- 使用mathlib4的Polynomial.card_roots_le_natDegree
  exact Polynomial.card_roots_le_natDegree f

end Exercises.Algebra
