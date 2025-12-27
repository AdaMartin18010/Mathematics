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
    -- 关键思路：
    -- 1. factors.prod与f关联（h_factors_prod）
    -- 2. factors_monic中的每个元素p = q * C(q.leadingCoeff⁻¹)，其中q ∈ factors
    -- 3. factors_monic.prod = factors.prod * C(∏_{q ∈ factors} q.leadingCoeff⁻¹)
    -- 4. 由于factors_monic都是首一的，factors_monic.prod也是首一的
    -- 5. 因此f = f.leadingCoeff • factors_monic.prod
    -- 首先证明factors_monic.prod是首一的
    have h_monic_prod : factors_monic.prod.Monic := by
      -- 首一多项式的乘积是首一的
      apply Multiset.prod_mem
      intro p hp
      exact (h_monic_irreducible p hp).2
    -- 然后证明factors_monic.prod与factors.prod关联
    have h_associated_prod : Associated factors.prod factors_monic.prod := by
      -- factors_monic.prod = factors.prod * C(∏_{q ∈ factors} q.leadingCoeff⁻¹)
      -- 需要计算归一化常数
      -- factors_monic = factors.map (fun q => q * C(q.leadingCoeff⁻¹))
      -- factors_monic.prod = ∏_{q ∈ factors} (q * C(q.leadingCoeff⁻¹))
      -- = (∏_{q ∈ factors} q) * (∏_{q ∈ factors} C(q.leadingCoeff⁻¹))
      -- = factors.prod * C(∏_{q ∈ factors} q.leadingCoeff⁻¹)
      -- 因此factors_monic.prod与factors.prod关联（通过单位C(∏_{q ∈ factors} q.leadingCoeff⁻¹)）
      -- 使用Multiset.prod_map和Multiset.prod_hom
      have h_prod_map : factors_monic.prod = (factors.map fun q => q * Polynomial.C q.leadingCoeff⁻¹).prod := rfl
      -- 使用Multiset.prod_map_mul
      have h_prod_mul : (factors.map fun q => q * Polynomial.C q.leadingCoeff⁻¹).prod =
        factors.prod * (factors.map fun q => Polynomial.C q.leadingCoeff⁻¹).prod := by
        -- 证明思路：prod_map (fun q => q * C(q.leadingCoeff⁻¹)) = prod_map id * prod_map (fun q => C(q.leadingCoeff⁻¹))
        -- 关键：需要将乘积映射分解为两个映射的乘积
        -- 实施替代方案：使用Multiset.prod_map的定义和乘法性质
        -- 对于Multiset s，有prod_map (f * g) = prod_map f * prod_map g（在乘法意义下）
        -- 这需要证明：∏_{x ∈ s} (f x * g x) = (∏_{x ∈ s} f x) * (∏_{x ∈ s} g x)
        -- 方法1：使用Multiset.prod_map_mul（如果存在）
        -- 方法2：使用Multiset.prod_map和Multiset.prod_hom的组合
        -- 方法3：直接展开定义，使用prod_map的性质和乘法的交换律、结合律
        -- 方法4：使用归纳法或递归定义，对Multiset的大小进行归纳
        -- 简化：在mathlib4中，可能需要使用Multiset.prod_map_mul或类似定理
        -- 如果不存在，可以使用更基本的方法：
        -- 1. 使用Multiset.prod_map的定义展开：prod_map f = prod (map f s)
        -- 2. 使用Multiset.map的分配律：map (f * g) = map f * map g（在某种意义下）
        -- 3. 使用prod的乘法性质：prod (s * t) = prod s * prod t
        -- 4. 使用归纳法：对Multiset的大小进行归纳，证明基础情况和归纳步骤
        -- 可能的API：Multiset.prod_map_mul, Multiset.prod_map_mul',
        -- Multiset.prod_map_prod, Multiset.prod_hom, 或类似定理
        -- 需要查找正确的API名称
        sorry -- TODO: 使用Multiset.prod_map_mul或类似定理（需要查找正确的API）
        -- 替代方案：如果API不存在，可以考虑：
        -- 1. 使用归纳法：对Multiset的大小进行归纳
        -- 2. 直接展开定义，使用prod_map和乘法的性质
        -- 3. 使用Multiset的递归定义和乘法的交换律、结合律
      -- 计算(factors.map fun q => Polynomial.C q.leadingCoeff⁻¹).prod
      have h_prod_C : (factors.map fun q => Polynomial.C q.leadingCoeff⁻¹).prod =
        Polynomial.C (factors.map fun q => q.leadingCoeff⁻¹).prod := by
        -- 证明思路：∏_{q ∈ factors} C(q.leadingCoeff⁻¹) = C(∏_{q ∈ factors} q.leadingCoeff⁻¹)
        -- 关键：Polynomial.C是乘法同态，即C(a) * C(b) = C(a * b)
        -- 实施替代方案：使用Polynomial.C的乘法同态性质和Multiset.prod_hom
        -- 对于Multiset s，如果f是乘法同态，则prod_map f = f (prod_map id)
        -- 这里f = Polynomial.C，需要证明C是乘法同态
        -- 方法1：使用Multiset.prod_hom（如果Polynomial.C是乘法同态）
        -- 方法2：使用Polynomial.map_prod或类似定理
        -- 方法3：使用Polynomial.C_mul和归纳法
        -- 方法4：直接展开定义，使用C的乘法性质
        -- 简化：在mathlib4中，Polynomial.C应该是乘法同态
        -- 证明步骤：
        -- 1. 证明Polynomial.C是乘法同态：C(a) * C(b) = C(a * b)
        -- 2. 使用Multiset.prod_hom：如果f是乘法同态，则prod_map f = f (prod_map id)
        -- 3. 应用：prod_map C = C (prod_map id)
        -- 可能的API：Multiset.prod_hom, Polynomial.map_prod,
        -- Polynomial.C_mul, Polynomial.C_prod, Polynomial.C.map_mul, 或类似定理
        -- 需要查找正确的API名称
        sorry -- TODO: 使用Polynomial.C的乘法同态性质（需要查找Multiset.prod_hom或Polynomial.map_prod）
        -- 替代方案：如果API不存在，可以：
        -- 1. 使用Polynomial.C_mul和归纳法：对Multiset的大小进行归纳
        -- 2. 直接展开定义，使用C的乘法性质：C(a) * C(b) = C(a * b)
        -- 3. 使用Multiset的递归定义和乘法的交换律、结合律
      -- 因此factors_monic.prod = factors.prod * C(∏_{q ∈ factors} q.leadingCoeff⁻¹)
      rw [h_prod_map, h_prod_mul, h_prod_C]
      -- 证明factors.prod与factors.prod * C(...)关联
      -- 使用Associated.mul_unit_right
      have h_unit : IsUnit (Polynomial.C (factors.map fun q => q.leadingCoeff⁻¹).prod) := by
        -- C(∏_{q ∈ factors} q.leadingCoeff⁻¹)是单位（因为C将非零元素映射为单位）
        apply isUnit_C.mpr
        -- 需要证明∏_{q ∈ factors} q.leadingCoeff⁻¹ ≠ 0
        -- 由于每个q.leadingCoeff ≠ 0（因为q是素元，非零），因此q.leadingCoeff⁻¹ ≠ 0
        -- 因此乘积≠ 0
        -- 使用Multiset.prod_ne_zero：如果所有元素非零，则乘积非零
        -- 需要证明：∀ q ∈ factors, q.leadingCoeff⁻¹ ≠ 0
        -- 由于每个q.leadingCoeff ≠ 0（因为q是素元，非零），因此q.leadingCoeff⁻¹ ≠ 0
        -- 使用inv_ne_zero和leadingCoeff_ne_zero
        have h_all_nonzero : ∀ q ∈ factors, q.leadingCoeff⁻¹ ≠ 0 := by
          intro q hq
          apply inv_ne_zero
          exact leadingCoeff_ne_zero.mpr (Prime.ne_zero (h_factors_prime q hq))
        -- 使用Multiset.prod_ne_zero
        exact Multiset.prod_ne_zero h_all_nonzero
      exact Associated.mul_unit_right _ _ h_unit
    -- 由h_factors_prod和h_associated_prod，f与factors_monic.prod关联
    have h_f_associated_monic : Associated f factors_monic.prod := by
      exact Associated.trans h_factors_prod h_associated_prod
    -- 由于factors_monic.prod是首一的，且f与factors_monic.prod关联
    -- 因此f = f.leadingCoeff • factors_monic.prod
    -- 使用Associated的性质：如果p和q关联，且q是首一的，则p = p.leadingCoeff • q
    have h_eq : f = f.leadingCoeff • factors_monic.prod := by
      -- 使用Associated的性质
      -- 如果f与factors_monic.prod关联，且factors_monic.prod是首一的
      -- 则f = f.leadingCoeff • factors_monic.prod
      -- 关键：如果p和q关联，则存在单位u使得p = u * q
      -- 如果q是首一的，则u = p.leadingCoeff，因此p = p.leadingCoeff • q
      -- 使用Associated.dvd_dvd得到存在单位u使得f = u * factors_monic.prod
      obtain ⟨u, hu_unit, h_eq_unit⟩ := h_f_associated_monic.dvd_dvd
      -- 由于factors_monic.prod是首一的，且f = u * factors_monic.prod
      -- 因此u = f.leadingCoeff（因为首一多项式的leadingCoeff = 1）
      -- 因此f = f.leadingCoeff • factors_monic.prod
      -- 需要证明u = f.leadingCoeff
      have h_u_eq : u = Polynomial.C f.leadingCoeff := by
        -- 由于factors_monic.prod是首一的，且f = u * factors_monic.prod
        -- 因此f.leadingCoeff = u.leadingCoeff * factors_monic.prod.leadingCoeff
        -- = u.leadingCoeff * 1 = u.leadingCoeff
        -- 由于u是单位，u = C(u.leadingCoeff) = C(f.leadingCoeff)
        -- 简化：直接使用leadingCoeff的性质
        -- 使用leadingCoeff_mul：如果p和q都是非零的，则(p * q).leadingCoeff = p.leadingCoeff * q.leadingCoeff
        -- 由于factors_monic.prod是首一的，factors_monic.prod.leadingCoeff = 1
        -- 因此f.leadingCoeff = u.leadingCoeff * 1 = u.leadingCoeff
        -- 由于u是单位，u是常数多项式，因此u = C(u.leadingCoeff) = C(f.leadingCoeff)
        -- 使用Polynomial.isUnit_iff：单位是常数多项式
        -- 使用Polynomial.eq_C_of_degree_eq_zero或类似定理
        -- 简化：由于u是单位，u是常数多项式，因此u = C(u.coeff 0) = C(u.leadingCoeff) = C(f.leadingCoeff)
        -- 需要证明u.leadingCoeff = f.leadingCoeff
        have h_leadingCoeff_eq : u.leadingCoeff = f.leadingCoeff := by
          -- 从h_eq_unit: f = u * factors_monic.prod
          -- 使用leadingCoeff_mul
          rw [← h_eq_unit]
          -- 需要证明(u * factors_monic.prod).leadingCoeff = u.leadingCoeff * factors_monic.prod.leadingCoeff
          -- 使用leadingCoeff_mul
          -- 需要证明u ≠ 0和factors_monic.prod ≠ 0
          have h_u_ne_zero : u ≠ 0 := IsUnit.ne_zero hu_unit
          have h_prod_ne_zero : factors_monic.prod ≠ 0 := by
            -- 由于factors_monic中的每个元素都是不可约的，因此非零
            -- 因此它们的乘积非零
            apply Multiset.prod_ne_zero
            intro p hp
            exact Irreducible.ne_zero (h_monic_irreducible p hp).1
          -- 使用leadingCoeff_mul
          rw [leadingCoeff_mul h_u_ne_zero h_prod_ne_zero]
          -- 由于factors_monic.prod是首一的，factors_monic.prod.leadingCoeff = 1
          rw [Polynomial.Monic.leadingCoeff (h_monic_prod)]
          ring
        -- 由于u是单位，u是常数多项式，因此u = C(u.leadingCoeff)
        -- 证明思路：在多项式环中，单位是常数多项式（非零常数）
        -- 实施替代方案：使用degree和coeff的性质
        -- 步骤1：证明u.degree = 0（使用isUnit的性质）
        -- 步骤2：使用eq_C_of_degree_eq_zero得到u = C(u.coeff 0)
        -- 步骤3：证明u.coeff 0 = u.leadingCoeff（当degree = 0时）
        -- 方法1：使用Polynomial.isUnit_iff（如果存在）
        -- 方法2：使用Polynomial.eq_C_of_degree_eq_zero和isUnit的条件
        -- 方法3：使用Polynomial.degree_eq_zero_of_isUnit和Polynomial.eq_C_of_degree_eq_zero
        -- 方法4：直接使用isUnit的性质和多项式的结构
        -- 简化：在mathlib4中，多项式的单位应该是常数多项式
        -- 证明步骤：
        -- 1. 使用isUnit的性质证明u.degree = 0
        --    - 如果u是单位，则u ≠ 0，且存在v使得u * v = 1
        --    - 由于deg(u * v) = deg(u) + deg(v) = deg(1) = 0
        --    - 因此deg(u) = 0和deg(v) = 0
        -- 2. 使用eq_C_of_degree_eq_zero：如果deg(u) = 0，则u = C(u.coeff 0)
        -- 3. 证明u.coeff 0 = u.leadingCoeff：当degree = 0时，leadingCoeff = coeff 0
        -- 可能的API：Polynomial.isUnit_iff, Polynomial.isUnit_iff_C,
        -- Polynomial.eq_C_of_degree_eq_zero, Polynomial.degree_eq_zero_of_isUnit,
        -- Polynomial.isUnit_iff_degree_eq_zero, Polynomial.degree_eq_zero_iff_eq_C, 或类似定理
        -- 需要查找正确的API名称
        sorry -- TODO: 使用isUnit_iff_C或类似定理证明u = C(u.leadingCoeff)
        -- 替代方案：如果API不存在，可以使用degree和coeff的性质：
        -- 1. 证明u.degree = 0（使用isUnit的性质和degree_mul）
        -- 2. 使用eq_C_of_degree_eq_zero得到u = C(u.coeff 0)
        -- 3. 证明u.coeff 0 = u.leadingCoeff（当degree = 0时，使用leadingCoeff的定义）
      rw [h_u_eq] at h_eq_unit
      -- 现在h_eq_unit: f = C(f.leadingCoeff) * factors_monic.prod
      -- 需要证明C(f.leadingCoeff) * factors_monic.prod = f.leadingCoeff • factors_monic.prod
      -- 这需要证明C(c) * p = c • p（对于多项式p）
      have h_smul_eq : Polynomial.C f.leadingCoeff * factors_monic.prod = f.leadingCoeff • factors_monic.prod := by
        -- 使用Polynomial.C_mul或类似定理
        -- C(c) * p = c • p（对于多项式p）
        -- 使用Polynomial.C_mul：C(c) * p = c • p
        -- 或者使用Polynomial.smul_eq_C_mul：c • p = C(c) * p
        -- 简化：直接使用Polynomial.smul_eq_C_mul
        rw [Polynomial.smul_eq_C_mul]
      rw [h_smul_eq] at h_eq_unit
      exact h_eq_unit
    exact h_eq

-- 根的个数
theorem card_roots_le_degree {F : Type*} [Field F] (f : Polynomial F) :
  (f.roots.toFinset.card : ℕ) ≤ f.natDegree := by
  -- 使用mathlib4的Polynomial.card_roots_le_natDegree
  exact Polynomial.card_roots_le_natDegree f

end Exercises.Algebra
