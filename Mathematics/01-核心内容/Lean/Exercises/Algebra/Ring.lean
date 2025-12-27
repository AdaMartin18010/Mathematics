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
  -- SOLUTION: 证明 (-a)*b + a*b = 0，然后由负元唯一性得出结论
  -- 首先计算 (-a)*b + a*b
  have h1 : (-a) * b + a * b = (-a + a) * b := by rw [← MyRing.right_distrib]
  -- 由环的加法负元性质：-a + a = 0
  have h2 : -a + a = 0 := MyRing.add_left_neg a
  -- 因此 (-a)*b + a*b = 0 * b
  have h3 : (-a) * b + a * b = 0 * b := by rw [h2] at h1; exact h1
  -- 而 0 * b = 0（需要证明）
  have h4 : (0 : R) * b = 0 := by
    have : (0 : R) * b + 0 * b = (0 + 0) * b := by rw [← MyRing.right_distrib]
    have : (0 : R) * b + 0 * b = 0 * b := by simpa using this
    have : 0 * b + (0 * b + -(0 * b)) = 0 * b + -(0 * b) := by rw [this]
    have : 0 * b + (0 * b + -(0 * b)) = 0 * b + 0 := by
      rw [MyRing.add_assoc]; rw [MyRing.add_left_neg]; rw [MyRing.add_zero]
    simpa [MyRing.add_left_neg, MyRing.add_zero] using this
  -- 因此 (-a)*b + a*b = 0
  have h5 : (-a) * b + a * b = 0 := by rw [h4] at h3; exact h3
  -- 由加法消去律（两边加 -(a*b)）得 (-a)*b = -(a*b)
  have h6 : (-a) * b + a * b + (-(a * b)) = 0 + (-(a * b)) := by rw [h5]
  have h7 : (-a) * b + (a * b + (-(a * b))) = -(a * b) := by
    simpa [MyRing.add_zero] using h6
  simpa [MyRing.add_left_neg, MyRing.add_zero] using h7

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

-- ============================================
-- 环论基础定理（使用mathlib4标准定义）
-- ============================================

-- 理想判别法
theorem ideal_criterion {R : Type*} [CommRing R] (I : Set R) (h_ne : I.Nonempty) :
  (∀ a b : R, a ∈ I → b ∈ I → a - b ∈ I) ∧
  (∀ a : R, a ∈ I → ∀ r : R, r * a ∈ I) ↔
  ∃ (J : Ideal R), ↑J = I := by
  -- 使用mathlib4的Ideal定义
  constructor
  · -- 充分性：如果I满足理想判别条件，则I是理想
    intro ⟨h_add, h_mul⟩
    -- 构造理想
    use {
      carrier := I
      add_mem' := by
        -- 需要证明：如果a, b ∈ I，则a + b ∈ I
        intro a b ha hb
        -- 由h_add，a - (-b) = a + b ∈ I
        have : a - (-b) ∈ I := h_add a (-b) ha (by
          -- 需要证明-b ∈ I
          -- 由h_mul，(-1) * b ∈ I
          have : (-1 : R) * b ∈ I := h_mul b hb (-1)
          rwa [neg_one_mul] at this)
        rwa [sub_neg_eq_add] at this
      zero_mem' := by
        -- 需要证明：0 ∈ I
        -- 由h_ne，存在a ∈ I
        obtain ⟨a, ha⟩ := h_ne
        -- 由h_add，a - a = 0 ∈ I
        have : a - a ∈ I := h_add a a ha ha
        rwa [sub_self] at this
      smul_mem' := by
        -- 需要证明：如果r : R，a ∈ I，则r • a ∈ I
        -- 在交换环中，r • a = r * a
        intro r a ha
        exact h_mul a ha r
    }
    rfl
  · -- 必要性：如果I是理想，则满足判别条件
    intro ⟨J, h_eq⟩
    constructor
    · -- 加法封闭性
      intro a b ha hb
      rw [← h_eq] at ha hb
      -- 由理想性质，a - b ∈ J
      exact J.sub_mem ha hb
    · -- 乘法吸收性
      intro a ha r
      rw [← h_eq] at ha
      -- 由理想性质，r * a ∈ J
      exact J.smul_mem r ha

-- 环同态基本定理
theorem first_isomorphism_theorem_ring {R S : Type*} [CommRing R] [CommRing S]
  (φ : R →+* S) :
  R ⧸ (RingHom.ker φ) ≃+* RingHom.range φ := by
  -- 使用mathlib4的quotientKerEquivRange
  exact RingHom.quotientKerEquivRange φ

-- 极大理想 ↔ 商环是域
theorem maximal_iff_quotient_field {R : Type*} [CommRing R] (M : Ideal R) :
  M.IsMaximal ↔ IsField (R ⧸ M) := by
  -- 使用mathlib4的Ideal.Quotient.maximal_ideal_iff_is_field_quotient
  exact Ideal.Quotient.maximal_ideal_iff_is_field_quotient M

-- 素理想 ↔ 商环是整环
theorem prime_iff_quotient_domain {R : Type*} [CommRing R] (P : Ideal R) :
  P.IsPrime ↔ IsDomain (R ⧸ P) := by
  -- 使用mathlib4的Ideal.Quotient.isDomain_iff_prime
  exact Ideal.Quotient.isDomain_iff_prime P

-- 有限整环是域
theorem finite_integral_domain_is_field {R : Type*} [CommRing R] [IsDomain R] [Fintype R] :
  IsField R := by
  -- 使用Fintype.fieldOfDomain
  exact Fintype.fieldOfDomain R

end Exercises.Algebra
