/-!
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`，版本随 `lakefile.lean` 固定到 stable 或已验证提交
- 最小导入：`import Std`, `import Mathlib`
-/

import Std
import Mathlib
import Mathlib.Analysis.Calculus.LocalExtr.Rolle
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Topology.MetricSpace.Sequences
import Mathlib.Topology.Algebra.InfiniteSum.Basic
import Mathlib.Analysis.Normed.Group.InfiniteSum

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

-- ============================================
-- 实数完备性定理
-- ============================================

-- 上界和下界的定义
def IsUpperBound (S : Set ℝ) (M : ℝ) : Prop :=
  ∀ x ∈ S, x ≤ M

def IsLowerBound (S : Set ℝ) (m : ℝ) : Prop :=
  ∀ x ∈ S, x ≥ m

-- 上确界定义
def IsSupremum (S : Set ℝ) (α : ℝ) : Prop :=
  IsUpperBound S α ∧ (∀ β, IsUpperBound S β → α ≤ β)

-- 单调有界定理（已在文档中完整证明）
theorem monotone_bounded_converges
  (a : ℕ → ℝ)
  (h_mono : ∀ n, a n ≤ a (n + 1))
  (h_bd : ∃ M, ∀ n, a n ≤ M) :
  ∃ L, Filter.Tendsto a Filter.atTop (𝓝 L) ∧ L = sSup (Set.range a) := by
  -- 证明集合 S = {a n} 非空有上界
  have S_def : Set ℝ := Set.range a
  have h_ne : S_def.Nonempty := ⟨a 0, 0, rfl⟩
  have h_ub : ∃ M, IsUpperBound S_def M := by
    obtain ⟨M, hM⟩ := h_bd
    use M
    intro x ⟨n, rfl⟩
    exact hM n

  -- 由完备性公理，sup S存在
  obtain ⟨α, h_sup⟩ := Real.sSup_exists h_ne h_ub
  use α

  constructor
  · -- 证明 a n → α
    rw [Metric.tendsto_atTop]
    intro ε hε
    -- 由上确界性质，存在 N 使 a N > α - ε
    have : ∃ N, a N > α - ε := by
      by_contra h_neg
      push_neg at h_neg
      have : IsUpperBound S_def (α - ε) := by
        intro x ⟨n, rfl⟩
        exact le_of_not_gt (h_neg n)
      have : α ≤ α - ε := h_sup.2 (α - ε) this
      linarith
    obtain ⟨N, hN⟩ := this
    use N
    intro n hn
    -- 对 n ≥ N，a N ≤ a n ≤ α
    have h1 : a N ≤ a n := by
      induction n, hn using Nat.le_induction with
      | base => rfl.le
      | succ n' _ ih => exact le_trans ih (h_mono n')
    have h2 : a n ≤ α := h_sup.1 ⟨n, rfl⟩
    -- 因此 |a n - α| < ε
    rw [Real.dist_eq]
    have : 0 ≤ α - a n := by linarith
    rw [abs_of_nonneg this]
    linarith

  · -- 证明 α = sSup S
    rfl

-- 区间套结构
structure NestedIntervals where
  a : ℕ → ℝ
  b : ℕ → ℝ
  h_nested : ∀ n, a n ≤ a (n + 1) ∧ b (n + 1) ≤ b n
  h_length : Filter.Tendsto (fun n => b n - a n) Filter.atTop (𝓝 0)

-- 区间套定理（完整证明）
theorem nested_intervals_theorem (I : NestedIntervals) :
  ∃! ξ, ∀ n, I.a n ≤ ξ ∧ ξ ≤ I.b n := by
  -- 第一步：证明 {a n} 单调递增有上界
  have h_mono_a : ∀ n, I.a n ≤ I.a (n + 1) := fun n => (I.h_nested n).1
  have h_bd_a : ∃ M, ∀ n, I.a n ≤ M := by
    use I.b 0
    intro n
    have : I.a n ≤ I.b n := by
      induction n with
      | zero =>
        have h1 := (I.h_nested 0).1
        have h2 := (I.h_nested 0).2
        linarith
      | succ k ih =>
        have h1 := (I.h_nested k).1
        have h2 := (I.h_nested k).2
        have h3 := (I.h_nested (k + 1)).1
        have h4 := (I.h_nested (k + 1)).2
        linarith
    have : I.b n ≤ I.b 0 := by
      induction n with
      | zero => rfl.le
      | succ k ih =>
        exact le_trans ((I.h_nested k).2) ih
    linarith

  -- 由单调有界定理，{a n} 收敛
  obtain ⟨α, h_conv_a, h_sup_a⟩ := monotone_bounded_converges I.a h_mono_a h_bd_a

  -- 第二步：证明 {b n} 单调递减有下界
  have h_mono_b : ∀ n, I.b (n + 1) ≤ I.b n := fun n => (I.h_nested n).2
  have h_bd_b : ∃ m, ∀ n, m ≤ I.b n := by
    use I.a 0
    intro n
    have : I.a n ≤ I.b n := by
      induction n with
      | zero =>
        have h1 := (I.h_nested 0).1
        have h2 := (I.h_nested 0).2
        linarith
      | succ k ih =>
        have h1 := (I.h_nested k).1
        have h2 := (I.h_nested k).2
        have h3 := (I.h_nested (k + 1)).1
        have h4 := (I.h_nested (k + 1)).2
        linarith
    have : I.a 0 ≤ I.a n := by
      induction n with
      | zero => rfl.le
      | succ k ih =>
        exact le_trans ih ((I.h_nested k).1)
    linarith

  -- 对 {b n} 应用单调有界定理（递减版本）
  have h_conv_b : ∃ β, Filter.Tendsto I.b Filter.atTop (𝓝 β) := by
    -- 构造递增序列 {-b n}
    let neg_b : ℕ → ℝ := fun n => -I.b n
    have h_mono_neg : ∀ n, neg_b n ≤ neg_b (n + 1) := by
      intro n
      simp [neg_b]
      have := (I.h_nested n).2
      linarith
    have h_bd_neg : ∃ M, ∀ n, neg_b n ≤ M := by
      use -I.a 0
      intro n
      simp [neg_b]
      have : I.a 0 ≤ I.b n := by
        have : I.a n ≤ I.b n := by
          induction n with
          | zero =>
            have h1 := (I.h_nested 0).1
            have h2 := (I.h_nested 0).2
            linarith
          | succ k ih =>
            have h1 := (I.h_nested k).1
            have h2 := (I.h_nested k).2
            have h3 := (I.h_nested (k + 1)).1
            have h4 := (I.h_nested (k + 1)).2
            linarith
        have : I.a 0 ≤ I.a n := by
          induction n with
          | zero => rfl.le
          | succ k ih =>
            exact le_trans ih ((I.h_nested k).1)
        linarith
      linarith
    obtain ⟨γ, h_conv_neg, _⟩ := monotone_bounded_converges neg_b h_mono_neg h_bd_neg
    use -γ
    convert Filter.Tendsto.neg h_conv_neg using 1
    ext n
    simp [neg_b]

  obtain ⟨β, h_conv_b⟩ := h_conv_b

  -- 第三步：证明 α = β
  have h_eq : α = β := by
    apply tendsto_nhds_unique h_conv_a
    convert h_conv_b using 1
    ext n
    -- 由 h_length，b n - a n → 0
    have h_diff : Filter.Tendsto (fun n => I.b n - I.a n) Filter.atTop (𝓝 0) := I.h_length
    -- 因此 a n = b n - (b n - a n) → β - 0 = β
    have : I.a = fun n => I.b n - (I.b n - I.a n) := by
      ext n
      ring
    rw [this]
    have : Filter.Tendsto (fun n => I.b n - (I.b n - I.a n)) Filter.atTop (𝓝 (β - 0)) := by
      apply Filter.Tendsto.sub h_conv_b h_diff
    simp at this
    exact this

  -- 第四步：证明 ξ = α 满足条件
  use α
  constructor
  · -- 证明 ∀ n, a n ≤ α ≤ b n
    intro n
    constructor
    · -- a n ≤ α
      have : I.a n ∈ Set.range I.a := ⟨n, rfl⟩
      have h_sup_val : α = sSup (Set.range I.a) := h_sup_a.symm
      rw [h_sup_val]
      exact le_csSup (Set.range_nonempty _) (by
        use I.b 0
        intro x ⟨m, rfl⟩
        have : I.a m ≤ I.b m := by
          induction m with
          | zero =>
            have h1 := (I.h_nested 0).1
            have h2 := (I.h_nested 0).2
            linarith
          | succ k ih =>
            have h1 := (I.h_nested k).1
            have h2 := (I.h_nested k).2
            have h3 := (I.h_nested (k + 1)).1
            have h4 := (I.h_nested (k + 1)).2
            linarith
        have : I.b m ≤ I.b 0 := by
          induction m with
          | zero => rfl.le
          | succ k ih =>
            exact le_trans ((I.h_nested k).2) ih
        linarith)
    · -- α ≤ b n
      rw [h_eq]
      -- 需要证明 β ≤ b n
      -- 由于 {b n} 单调递减且收敛到 β，对任意 n，有 β ≤ b n
      -- 这可以从单调递减序列的极限性质得出
      have h_mono_b' : ∀ m n, m ≤ n → I.b n ≤ I.b m := by
        intro m n hmn
        induction hmn with
        | refl => rfl.le
        | step k _ ih =>
          exact le_trans ((I.h_nested k).2) ih
      -- 由于 b n ≤ b 0 对所有 n，且 b n → β，有 β ≤ b n
      -- 使用极限的保序性：如果对所有 m ≥ n 有 b m ≤ b n，则极限 β ≤ b n
      have : β ≤ I.b n := by
        -- 使用Filter.Tendsto的保序性
        -- 由于对所有 m ≥ n，b m ≤ b n，且 b m → β，因此 β ≤ b n
        have h_bound : ∀ m ≥ n, I.b m ≤ I.b n := fun m hmn => h_mono_b' n m hmn
        -- 这需要从极限的保序性得出，使用le_of_tendsto或类似定理
        -- 在mathlib4中，可以使用Filter.Tendsto.le_of_eventually_le
        have : Filter.Tendsto I.b Filter.atTop (𝓝 β) := h_conv_b
        -- 由于eventually (fun m => b m ≤ b n)，且b m → β，因此β ≤ b n
        have : ∀ᶠ m in Filter.atTop, I.b m ≤ I.b n := by
          apply Filter.eventually_atTop.mpr
          use n
          intro m hmn
          exact h_bound m hmn
        -- 使用le_of_tendsto_of_eventually_le（mathlib4 API）
        -- 在mathlib4中，可以使用Filter.Tendsto.le_of_eventually_le
        -- 或者使用tendsto_le_of_eventually_le
        exact tendsto_le_of_eventually_le h_conv_b this
      exact this

  · -- 证明唯一性
    intro ξ' h_ξ'
    -- 由条件，对所有 n 有 a n ≤ ξ' ≤ b n
    -- 取极限得 α ≤ ξ' ≤ β，而 α = β，故 ξ' = α
    have h1 : α ≤ ξ' := by
      -- 由 a n → α 和 a n ≤ ξ'，取极限得 α ≤ ξ'
      have : Filter.Tendsto (fun n => I.a n) Filter.atTop (𝓝 α) := h_conv_a
      have : ∀ n, I.a n ≤ ξ' := fun n => (h_ξ' n).1
      -- 使用极限的保序性：如果对所有 n，a n ≤ ξ'，且 a n → α，则 α ≤ ξ'
      have : ∀ᶠ n in Filter.atTop, I.a n ≤ ξ' := by
        apply Filter.eventually_atTop.mpr
        use 0
        intro n _
        exact this n
      -- 使用le_of_tendsto_of_eventually_le（mathlib4 API）
      -- 在mathlib4中，可以使用tendsto_le_of_eventually_le
      exact tendsto_le_of_eventually_le h_conv_a this
    have h2 : ξ' ≤ β := by
      rw [h_eq]
      -- 类似地，由 b n → β 和 ξ' ≤ b n，取极限得 ξ' ≤ β
      have : ∀ n, ξ' ≤ I.b n := fun n => (h_ξ' n).2
      have : ∀ᶠ n in Filter.atTop, ξ' ≤ I.b n := by
        apply Filter.eventually_atTop.mpr
        use 0
        intro n _
        exact this n
      -- 使用le_of_tendsto_of_eventually_le（mathlib4 API）
      -- 在mathlib4中，对于常数序列，可以使用tendsto_const_nhds和tendsto_le_of_eventually_le
      -- 但这里我们需要证明 ξ' ≤ β，其中 ξ' 是常数，b n → β
      -- 可以使用tendsto_le_of_eventually_le，但需要构造tendsto (fun n => ξ') atTop (𝓝 ξ')
      have h_const : Filter.Tendsto (fun n => ξ') Filter.atTop (𝓝 ξ') := Filter.tendsto_const_nhds
      exact tendsto_le_of_eventually_le h_const this
    have : α = β := h_eq
    linarith

-- Bolzano-Weierstrass定理（完整证明）
theorem bolzano_weierstrass
  (a : ℕ → ℝ)
  (h_bd : ∃ M, ∀ n, |a n| ≤ M) :
  ∃ φ : ℕ → ℕ, StrictMono φ ∧ ∃ L, Filter.Tendsto (a ∘ φ) Filter.atTop (𝓝 L) := by
  obtain ⟨M, hM⟩ := h_bd

  -- 证明Set.range a是有界的
  have h_bounded : IsBounded (Set.range a) := by
    use M
    intro x ⟨n, rfl⟩
    exact hM n

  -- 证明序列a的所有项都在Set.range a中
  have h_range : ∀ n, a n ∈ Set.range a := by
    intro n
    exact ⟨n, rfl⟩

  -- 使用mathlib4的tendsto_subseq_of_bounded
  -- 在ℝ中，有界集是proper的，因此可以使用tendsto_subseq_of_bounded
  obtain ⟨L, hL_mem, φ, hφ_mono, h_tendsto⟩ := tendsto_subseq_of_bounded h_bounded h_range

  -- 返回结果
  use φ, hφ_mono, L
  exact h_tendsto

-- ============================================
-- 极限与连续性定理
-- ============================================

-- 序列极限定义（使用mathlib4的标准定义）
-- 注：mathlib4中通常直接使用Filter.Tendsto，这里提供等价定义
def SequenceLimit (a : ℕ → ℝ) (L : ℝ) : Prop :=
  Filter.Tendsto a Filter.atTop (𝓝 L)

-- 函数极限定义
def FunctionLimit (f : ℝ → ℝ) (x₀ L : ℝ) : Prop :=
  Filter.Tendsto f (𝓝 x₀) (𝓝 L)

-- 连续性定义
def ContinuousAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  Filter.Tendsto f (𝓝 x₀) (𝓝 (f x₀))

-- 定理：极限唯一性
theorem limit_unique (a : ℕ → ℝ) (L₁ L₂ : ℝ)
  (h₁ : SequenceLimit a L₁) (h₂ : SequenceLimit a L₂) : L₁ = L₂ := by
  -- 使用mathlib4的tendsto_nhds_unique
  exact tendsto_nhds_unique h₁ h₂

-- 定理：介值定理
theorem intermediate_value
  {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : ContinuousOn f (Set.Icc a b))
  {y : ℝ} (hy : y ∈ Set.Icc (f a) (f b) ∨ y ∈ Set.Icc (f b) (f a)) :
  ∃ c ∈ Set.Icc a b, f c = y := by
  -- 使用mathlib4的intermediate_value_Icc
  -- 首先处理两种情况：f a ≤ f b 或 f b ≤ f a
  cases hy with
  | inl h1 =>
    -- y ∈ Set.Icc (f a) (f b)，即 f a ≤ y ≤ f b
    have : f a ≤ y ∧ y ≤ f b := h1
    -- 使用intermediate_value_Icc
    exact intermediate_value_Icc hab hf this.1 this.2
  | inr h2 =>
    -- y ∈ Set.Icc (f b) (f a)，即 f b ≤ y ≤ f a
    have h_bounds : f b ≤ y ∧ y ≤ f a := h2
    -- 由于a < b，且f b ≤ y ≤ f a
    -- 考虑函数g(x) = f(x) - y，它在[a, b]上连续
    -- g(b) = f(b) - y ≤ 0，g(a) = f(a) - y ≥ 0
    -- 由零点定理（介值定理的特殊情况），存在c ∈ [a, b]使得g(c) = 0，即f(c) = y

    -- 更直接的方法：使用intermediate_value_Icc，但需要调整顺序
    -- 由于f在[a, b]上连续，且f b ≤ y ≤ f a
    -- 我们可以通过考虑-f来转换，或者直接使用更通用的方法

    -- 简化处理：如果f b ≤ y ≤ f a，且a < b
    -- 由于f在[a, b]上连续，由介值定理的推广形式，存在c ∈ [a, b]使得f c = y
    -- 这可以通过考虑函数h(x) = f(x) - y，它在[a, b]上连续
    -- h(a) = f(a) - y ≥ 0，h(b) = f(b) - y ≤ 0
    -- 由零点定理，存在c ∈ [a, b]使得h(c) = 0，即f(c) = y

    -- 使用mathlib4的intermediate_value_Icc'（如果存在）或通过零点定理
    -- 这里我们使用intermediate_value_Icc，但需要调整
    -- 实际上，我们可以直接使用intermediate_value_Icc，因为：
    -- 如果f b ≤ y ≤ f a，且a < b，则y在f(a)和f(b)之间
    -- 但intermediate_value_Icc要求f a ≤ y ≤ f b
    -- 所以我们需要使用intermediate_value_Icc'或类似定理

    -- 简化：直接使用intermediate_value_Icc'（如果mathlib4有）
    -- 或者通过考虑-f来转换
    -- 方法：考虑函数g(x) = -f(x)，则g在[a, b]上连续
    -- 且g(a) = -f(a) ≤ -y，g(b) = -f(b) ≥ -y
    -- 由intermediate_value_Icc，存在c使得g(c) = -y，即f(c) = y
    have h_cont_neg : ContinuousOn (fun x => -f x) (Set.Icc a b) := by
      exact ContinuousOn.neg hf
    have h_bounds_neg : -f a ≤ -y ∧ -y ≤ -f b := by
      constructor
      · -- -f a ≤ -y，即y ≤ f a
        exact h_bounds.2
      · -- -y ≤ -f b，即f b ≤ y
        exact h_bounds.1
    -- 使用intermediate_value_Icc在[-f a, -f b]上
    -- 但需要-f a ≤ -f b，即f b ≤ f a
    by_cases h_order : -f a ≤ -f b
    · -- -f a ≤ -f b的情况，直接使用intermediate_value_Icc
      exact intermediate_value_Icc hab h_cont_neg h_bounds_neg.1 h_bounds_neg.2
    · -- -f a > -f b的情况，即f a < f b
      -- 此时y ∈ [f b, f a]不成立，但y ∈ [f a, f b]成立
      -- 这与h2矛盾，因为h2说y ∈ [f b, f a]
      push_neg at h_order
      have : f a < f b := h_order
      -- 但h_bounds说f b ≤ y ≤ f a，这与f a < f b矛盾
      have : f b ≤ f a := h_bounds.2
      linarith

-- 定理：最值定理
theorem extreme_value
  {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
  (hf : ContinuousOn f (Set.Icc a b)) :
  (∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f y ≤ f x) ∧
  (∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f x ≤ f y) := by
  -- 使用mathlib4的紧致性和连续函数的最值定理
  -- Set.Icc a b是紧致的（在ℝ中）
  have h_compact : IsCompact (Set.Icc a b) := by
    exact isCompact_Icc

  -- f在紧致集上连续，因此有最大值和最小值
  -- 使用IsCompact.exists_isMaxOn和IsCompact.exists_isMinOn
  constructor
  · -- 存在最大值
    obtain ⟨x, hx_mem, hx_max⟩ := h_compact.exists_isMaxOn (Set.nonempty_Icc.mpr hab) hf.continuousOn
    use x, hx_mem
    intro y hy_mem
    exact hx_max y hy_mem
  · -- 存在最小值
    obtain ⟨x, hx_mem, hx_min⟩ := h_compact.exists_isMinOn (Set.nonempty_Icc.mpr hab) hf.continuousOn
    use x, hx_mem
    intro y hy_mem
    exact hx_min y hy_mem

-- ============================================
-- 微分学基础定理
-- ============================================

-- 导数定义（使用mathlib4标准定义）
-- 注：mathlib4中通常使用HasDerivAt或DifferentiableAt
def HasDerivAt (f : ℝ → ℝ) (f' : ℝ) (x : ℝ) : Prop :=
  Filter.Tendsto (fun h => (f (x + h) - f x) / h) (𝓝[≠] 0) (𝓝 f')

-- 定理：可导必连续
theorem differentiable_implies_continuous
  {f : ℝ → ℝ} {x : ℝ} (hf : DifferentiableAt ℝ f x) :
  ContinuousAt f x := by
  exact DifferentiableAt.continuousAt hf

-- 定理：乘积法则
theorem mul_deriv
  {f g : ℝ → ℝ} {x f' g' : ℝ}
  (hf : HasDerivAt f f' x) (hg : HasDerivAt g g' x) :
  HasDerivAt (fun x => f x * g x) (f' * g x + f x * g') x := by
  -- 使用mathlib4的HasDerivAt.mul
  exact HasDerivAt.mul hf hg

-- 定理：链式法则
theorem chain_rule
  {f : ℝ → ℝ} {g : ℝ → ℝ} {x f' g' : ℝ}
  (hg : HasDerivAt g g' x) (hf : HasDerivAt f f' (g x)) :
  HasDerivAt (f ∘ g) (f' * g') x := by
  -- 使用mathlib4的HasDerivAt.comp
  exact HasDerivAt.comp x hf hg

-- Rolle定理
theorem rolle
  {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : ContinuousOn f (Set.Icc a b))
  (hf' : ∀ x ∈ Set.Ioo a b, DifferentiableAt ℝ f x)
  (hfab : f a = f b) :
  ∃ c ∈ Set.Ioo a b, deriv f c = 0 := by
  -- 使用mathlib4的exists_deriv_eq_zero
  -- 注意：mathlib4的exists_deriv_eq_zero只需要f在[a, b]上连续，在(a, b)内可导，且f(a) = f(b)
  -- 不需要边界点的可导性
  exact exists_deriv_eq_zero hab hf hfab

-- Lagrange中值定理
theorem mean_value_theorem
  {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : ContinuousOn f (Set.Icc a b))
  (hf' : ∀ x ∈ Set.Ioo a b, DifferentiableAt ℝ f x) :
  ∃ c ∈ Set.Ioo a b, deriv f c = (f b - f a) / (b - a) := by
  -- 使用mathlib4的exists_deriv_eq_slope
  -- 注意：mathlib4的exists_deriv_eq_slope只需要f在[a, b]上连续，在(a, b)内可导
  -- 不需要边界点的可导性
  have h_diff_on : DifferentiableOn ℝ f (Set.Ioo a b) := by
    intro x hx
    exact hf' x hx
  exact exists_deriv_eq_slope hab hf h_diff_on

-- Taylor定理（一阶情形）
theorem taylor_first_order
  {f : ℝ → ℝ} {a x : ℝ}
  (hf_cont : ContinuousOn f (Set.Icc (min a x) (max a x)))
  (hf_diff : ∀ y ∈ Set.Ioo (min a x) (max a x), DifferentiableAt ℝ f y) :
  ∃ θ ∈ Set.Ioo 0 1, f x = f a + deriv f (a + θ * (x - a)) * (x - a) := by
  -- 这是Lagrange中值定理的特殊形式
  -- 如果x = a，则θ可以是任意值
  by_cases h_eq : x = a
  · -- x = a的情况
    use 1/2
    constructor
    · constructor
      · linarith
      · linarith
    · -- f a = f a + deriv f (a + (1/2) * 0) * 0
      rw [h_eq]
      ring
  · -- x ≠ a的情况，使用Lagrange中值定理
    -- 不失一般性，假设a < x（如果x < a，可以交换）
    by_cases h_lt : a < x
    · -- a < x的情况
      have h_interval : Set.Icc a x = Set.Icc (min a x) (max a x) := by
        simp [min_eq_left (le_of_lt h_lt), max_eq_right (le_of_lt h_lt)]
      have h_cont : ContinuousOn f (Set.Icc a x) := by
        rwa [h_interval]
      have h_diff' : ∀ y ∈ Set.Ioo a x, DifferentiableAt ℝ f y := by
        intro y hy
        have : y ∈ Set.Ioo (min a x) (max a x) := by
          simp [min_eq_left (le_of_lt h_lt), max_eq_right (le_of_lt h_lt)]
          exact hy
        exact hf_diff y this
      -- 使用Lagrange中值定理
      obtain ⟨c, hc_mem, hc_deriv⟩ := mean_value_theorem h_lt h_cont h_diff'
      -- 计算θ使得a + θ * (x - a) = c
      -- θ = (c - a) / (x - a)
      have h_denom_ne_zero : x - a ≠ 0 := sub_ne_zero.mpr (Ne.symm h_eq)
      let θ := (c - a) / (x - a)
      -- 证明θ ∈ (0, 1)
      have h_θ_mem : θ ∈ Set.Ioo 0 1 := by
        constructor
        · -- 0 < θ
          have : 0 < c - a := by
            have : a < c := hc_mem.1
            linarith
          have : 0 < x - a := by linarith
          exact div_pos this this
        · -- θ < 1
          have : c - a < x - a := by
            have : c < x := hc_mem.2
            linarith
          exact (div_lt_div_right (by linarith)).mpr this
      -- 证明f x = f a + deriv f (a + θ * (x - a)) * (x - a)
      use θ, h_θ_mem
      -- 由Lagrange中值定理，deriv f c = (f x - f a) / (x - a)
      -- 因此f x = f a + deriv f c * (x - a)
      -- 且a + θ * (x - a) = c
      have h_c_eq : a + θ * (x - a) = c := by
        field_simp [θ]
        ring
      rw [h_c_eq]
      -- 由hc_deriv，deriv f c = (f x - f a) / (x - a)
      rw [hc_deriv]
      field_simp [h_denom_ne_zero]
      ring
    · -- x < a的情况
      push_neg at h_lt
      have h_lt' : x < a := Ne.lt_or_lt.mp h_eq |>.resolve_left h_lt
      have h_interval : Set.Icc x a = Set.Icc (min a x) (max a x) := by
        simp [min_eq_right (le_of_lt h_lt'), max_eq_left (le_of_lt h_lt')]
      have h_cont : ContinuousOn f (Set.Icc x a) := by
        rwa [h_interval]
      have h_diff' : ∀ y ∈ Set.Ioo x a, DifferentiableAt ℝ f y := by
        intro y hy
        have : y ∈ Set.Ioo (min a x) (max a x) := by
          simp [min_eq_right (le_of_lt h_lt'), max_eq_left (le_of_lt h_lt')]
          exact hy
        exact hf_diff y this
      -- 使用Lagrange中值定理
      obtain ⟨c, hc_mem, hc_deriv⟩ := mean_value_theorem h_lt' h_cont h_diff'
      -- 计算θ使得a + θ * (x - a) = c
      -- θ = (c - a) / (x - a)
      have h_denom_ne_zero : x - a ≠ 0 := sub_ne_zero.mpr h_eq
      let θ := (c - a) / (x - a)
      -- 证明θ ∈ (0, 1)
      have h_θ_mem : θ ∈ Set.Ioo 0 1 := by
        constructor
        · -- 0 < θ
          have : c - a < 0 := by
            have : c < a := hc_mem.2
            linarith
          have : x - a < 0 := by linarith
          exact div_pos_of_neg_of_neg this this
        · -- θ < 1
          have : x - a < c - a := by
            have : x < c := hc_mem.1
            linarith
          have : x - a < 0 := by linarith
          exact (div_lt_one_of_lt (by linarith)).mpr this
      -- 证明f x = f a + deriv f (a + θ * (x - a)) * (x - a)
      use θ, h_θ_mem
      -- 由Lagrange中值定理，deriv f c = (f a - f x) / (a - x)
      -- 因此f x = f a - deriv f c * (a - x) = f a + deriv f c * (x - a)
      -- 且a + θ * (x - a) = c
      have h_c_eq : a + θ * (x - a) = c := by
        field_simp [θ]
        ring
      rw [h_c_eq]
      -- 由hc_deriv，deriv f c = (f a - f x) / (a - x) = (f x - f a) / (x - a)
      rw [hc_deriv]
      field_simp [h_denom_ne_zero]
      ring

-- ============================================
-- Riemann积分定理
-- ============================================

-- 微积分基本定理 I (积分函数的导数)
theorem fundamental_theorem_calculus_I
  {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
  (hf : IntervalIntegrable f volume a b) :
  let F := fun x => ∫ t in a..x, f t
  ContinuousOn F (Set.Icc a b) ∧
  (∀ x ∈ Set.Ioo a b, HasDerivAt F (f x) x) := by
  -- 使用mathlib4的integral_hasStrictDerivAt
  intro F
  constructor
  · -- 连续性
    -- 使用integral_continuousOn
    exact intervalIntegral.continuousOn_integral hab hf
  · -- 可导性
    intro x hx
    -- 使用integral_hasStrictDerivAt
    have : HasStrictDerivAt F (f x) x := by
      exact intervalIntegral.integral_hasStrictDerivAt hf hx.1 hx.2
    exact this.hasDerivAt

-- 微积分基本定理 II (Newton-Leibniz公式)
theorem fundamental_theorem_calculus_II
  {f F : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
  (hF_cont : ContinuousOn F (Set.Icc a b))
  (hF' : ∀ x ∈ Set.Ioo a b, HasDerivAt F (f x) x)
  (hf : IntervalIntegrable f volume a b) :
  ∫ x in a..b, f x = F b - F a := by
  -- 使用mathlib4的integral_eq_sub_of_hasDerivAt
  exact intervalIntegral.integral_eq_sub_of_hasDerivAt hF_cont hF' hab hf

-- 积分中值定理
theorem integral_mean_value
  {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : ContinuousOn f (Set.Icc a b)) :
  ∃ ξ ∈ Set.Icc a b, ∫ x in a..b, f x = f ξ * (b - a) := by
  -- 使用连续性和介值定理
  -- 首先证明f在[a, b]上可积
  have h_integrable : IntervalIntegrable f volume a b := by
    exact continuousOn_intervalIntegrable hf hab.le

  -- 由微积分基本定理I，F(x) = ∫[a,x] f在[a, b]上连续
  let F := fun x => ∫ t in a..x, f t
  have hF_cont : ContinuousOn F (Set.Icc a b) := by
    exact intervalIntegral.continuousOn_integral hab.le h_integrable

  -- 由微积分基本定理II，∫[a,b] f = F(b) - F(a)
  have h_integral : ∫ x in a..b, f x = F b - F a := by
    have hF' : ∀ x ∈ Set.Ioo a b, HasDerivAt F (f x) x := by
      intro x hx
      exact (intervalIntegral.integral_hasStrictDerivAt h_integrable hx.1 hx.2).hasDerivAt
    exact intervalIntegral.integral_eq_sub_of_hasDerivAt hF_cont hF' hab.le h_integrable

  -- 由最值定理，f在[a, b]上有最大值M和最小值m
  obtain ⟨m, h_m_min⟩ := isCompact_Icc.exists_isMinOn (Set.nonempty_Icc.mpr hab.le) hf.continuousOn
  obtain ⟨M, h_M_max⟩ := isCompact_Icc.exists_isMaxOn (Set.nonempty_Icc.mpr hab.le) hf.continuousOn

  -- 由积分的基本性质，m(b-a) ≤ ∫[a,b] f ≤ M(b-a)
  have h_bounds : m * (b - a) ≤ ∫ x in a..b, f x ∧ ∫ x in a..b, f x ≤ M * (b - a) := by
    -- 这需要积分的不等式性质
    -- 使用积分的单调性：如果f ≤ g，则∫ f ≤ ∫ g
    constructor
    · -- m * (b - a) ≤ ∫[a,b] f
      -- 由于m ≤ f x对所有x ∈ [a, b]，由积分的单调性
      have h_m_le_f : ∀ x ∈ Set.Icc a b, m ≤ f x := by
        intro x hx
        exact h_m_min x hx
      have h_const_integrable : IntervalIntegrable (fun _ => m) volume a b := by
        exact continuousOn_const.intervalIntegrable hab.le
      have h_integral_const : ∫ x in a..b, m = m * (b - a) := by
        simp [intervalIntegral.integral_const]
      have h_le : ∫ x in a..b, m ≤ ∫ x in a..b, f x := by
        exact intervalIntegral.integral_mono_on hab.le h_const_integrable h_integrable
          (fun x hx => h_m_le_f x hx)
      rw [h_integral_const] at h_le
      exact h_le
    · -- ∫[a,b] f ≤ M * (b - a)
      -- 由于f x ≤ M对所有x ∈ [a, b]，由积分的单调性
      have h_f_le_M : ∀ x ∈ Set.Icc a b, f x ≤ M := by
        intro x hx
        exact h_M_max x hx
      have h_const_integrable : IntervalIntegrable (fun _ => M) volume a b := by
        exact continuousOn_const.intervalIntegrable hab.le
      have h_integral_const : ∫ x in a..b, M = M * (b - a) := by
        simp [intervalIntegral.integral_const]
      have h_le : ∫ x in a..b, f x ≤ ∫ x in a..b, M := by
        exact intervalIntegral.integral_mono_on hab.le h_integrable h_const_integrable
          (fun x hx => h_f_le_M x hx)
      rw [h_integral_const] at h_le
      exact h_le

  -- 由介值定理，存在ξ ∈ [a, b]使得f(ξ) = (∫[a,b] f) / (b - a)
  have h_div_ne_zero : b - a ≠ 0 := sub_ne_zero.mpr (ne_of_lt hab).symm
  let y := (∫ x in a..b, f x) / (b - a)
  have h_y_bounds : m ≤ y ∧ y ≤ M := by
    constructor
    · -- m ≤ y
      have : m * (b - a) ≤ ∫ x in a..b, f x := h_bounds.1
      have : m ≤ y := by
        field_simp [y]
        have h_pos : 0 < b - a := sub_pos.mpr hab
        exact (div_le_div_right h_pos).mpr this
      exact this
    · -- y ≤ M
      have : ∫ x in a..b, f x ≤ M * (b - a) := h_bounds.2
      have : y ≤ M := by
        field_simp [y]
        have h_pos : 0 < b - a := sub_pos.mpr hab
        exact (div_le_div_right h_pos).mpr this
      exact this

  -- 由介值定理，存在ξ ∈ [a, b]使得f(ξ) = y
  have h_ivt : ∃ ξ ∈ Set.Icc a b, f ξ = y := by
    -- 需要证明y在f的最小值和最大值之间
    -- 然后使用介值定理
    -- 由h_y_bounds，m ≤ y ≤ M
    -- 由最值定理，存在x_m, x_M ∈ [a, b]使得f(x_m) = m, f(x_M) = M
    obtain ⟨x_m, hx_m_mem, hx_m_val⟩ := h_m_min
    obtain ⟨x_M, hx_M_mem, hx_M_val⟩ := h_M_max
    -- 如果y = m或y = M，直接使用x_m或x_M
    by_cases h_y_eq_m : y = m
    · use x_m, hx_m_mem
      rw [h_y_eq_m, hx_m_val]
    by_cases h_y_eq_M : y = M
    · use x_M, hx_M_mem
      rw [h_y_eq_M, hx_M_val]
    -- 否则，m < y < M
    have h_strict : m < y ∧ y < M := by
      constructor
      · exact lt_of_le_of_ne h_y_bounds.1 h_y_eq_m
      · exact lt_of_le_of_ne h_y_bounds.2 h_y_eq_M.symm
    -- 由介值定理，存在ξ ∈ [a, b]使得f(ξ) = y
    -- 使用intermediate_value_Icc
    have h_ivt_strict : ∃ ξ ∈ Set.Icc a b, f ξ = y := by
      -- 由于f在[a, b]上连续，且m < y < M
      -- 存在ξ ∈ [a, b]使得f(ξ) = y
      -- 使用intermediate_value_Icc，但需要f a ≤ y ≤ f b或f b ≤ y ≤ f a
      -- 由于m ≤ y ≤ M，且m和M分别是f的最小值和最大值
      -- 存在x_m, x_M使得f(x_m) = m, f(x_M) = M
      -- 如果x_m < x_M，则在[x_m, x_M]上使用介值定理
      -- 如果x_M < x_m，则在[x_M, x_m]上使用介值定理
      -- 简化：直接使用intermediate_value_Icc'（如果存在）或通过零点定理
      -- 考虑函数g(x) = f(x) - y，它在[a, b]上连续
      -- g(x_m) = m - y < 0，g(x_M) = M - y > 0
      -- 由零点定理，存在ξ使得g(ξ) = 0，即f(ξ) = y
      -- 使用intermediate_value_Icc
      have h_cont_g : ContinuousOn (fun x => f x - y) (Set.Icc a b) := by
        exact ContinuousOn.sub hf.continuousOn continuousOn_const
      have h_g_bounds : (fun x => f x - y) x_m ≤ 0 ∧ 0 ≤ (fun x => f x - y) x_M := by
        constructor
        · -- f(x_m) - y ≤ 0，即m ≤ y
          simp [hx_m_val]
          exact h_y_bounds.1
        · -- 0 ≤ f(x_M) - y，即y ≤ M
          simp [hx_M_val]
          exact h_y_bounds.2
      -- 使用intermediate_value_Icc
      have h_ivt_g : ∃ ξ ∈ Set.Icc a b, (fun x => f x - y) ξ = 0 := by
        -- 需要确定x_m和x_M的顺序
        by_cases h_order : x_m ≤ x_M
        · -- x_m ≤ x_M的情况
          have h_ivt_local : ∃ ξ ∈ Set.Icc x_m x_M, (fun x => f x - y) ξ = 0 := by
            exact intermediate_value_Icc (le_trans hx_m_mem.1 hx_M_mem.2) h_cont_g
              h_g_bounds.1 h_g_bounds.2
          obtain ⟨ξ, h_ξ_mem, h_ξ_val⟩ := h_ivt_local
          use ξ
          constructor
          · -- ξ ∈ Set.Icc a b
            exact ⟨le_trans hx_m_mem.1 h_ξ_mem.1, le_trans h_ξ_mem.2 hx_M_mem.2⟩
          · exact h_ξ_val
        · -- x_M < x_m的情况
          have h_order' : x_M ≤ x_m := le_of_not_le h_order
          have h_ivt_local : ∃ ξ ∈ Set.Icc x_M x_m, (fun x => f x - y) ξ = 0 := by
            exact intermediate_value_Icc (le_trans hx_M_mem.1 hx_m_mem.2) h_cont_g
              h_g_bounds.2 h_g_bounds.1
          obtain ⟨ξ, h_ξ_mem, h_ξ_val⟩ := h_ivt_local
          use ξ
          constructor
          · -- ξ ∈ Set.Icc a b
            exact ⟨le_trans hx_M_mem.1 h_ξ_mem.1, le_trans h_ξ_mem.2 hx_m_mem.2⟩
          · exact h_ξ_val
      obtain ⟨ξ, h_ξ_mem, h_ξ_val⟩ := h_ivt_g
      use ξ, h_ξ_mem
      simp at h_ξ_val
      exact h_ξ_val
    exact h_ivt_strict

  obtain ⟨ξ, h_ξ_mem, h_ξ_val⟩ := h_ivt
  use ξ, h_ξ_mem
  -- f(ξ) * (b - a) = y * (b - a) = ∫[a,b] f
  rw [h_ξ_val]
  field_simp [y]

-- 换元积分法
theorem integration_by_substitution
  {f φ : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
  (hf : ContinuousOn f (Set.Icc (φ a) (φ b)))
  (hφ : ContinuousOn φ (Set.Icc a b))
  (hφ' : ∀ x ∈ Set.Ioo a b, DifferentiableAt ℝ φ x) :
  ∫ x in a..b, f (φ x) * (deriv φ x) = ∫ u in φ a..φ b, f u := by
  -- 使用mathlib4的integral_comp_smul_deriv
  -- 需要先证明f(φ(x)) * φ'(x)在[a, b]上可积
  have h_integrable : IntervalIntegrable (fun x => f (φ x) * deriv φ x) volume a b := by
    -- 需要从hf, hφ, hφ'推导出可积性
    -- 由于f在[φ a, φ b]上连续，φ在[a, b]上连续，deriv φ在[a, b]上连续
    -- 因此f(φ(x))在[a, b]上连续（复合函数连续性）
    -- deriv φ在[a, b]上连续
    -- 因此f(φ(x)) * deriv φ x在[a, b]上连续（乘积连续性）
    -- 连续函数在闭区间上可积
    have h_f_comp_cont : ContinuousOn (fun x => f (φ x)) (Set.Icc a b) := by
      -- f在[φ a, φ b]上连续，φ在[a, b]上连续
      -- 由复合函数连续性，f(φ(x))在[a, b]上连续
      exact ContinuousOn.comp hf hφ (Set.mapsTo_image φ (Set.Icc a b))
    have h_deriv_cont : ContinuousOn (deriv φ) (Set.Icc a b) := by
      -- 由hφ'，φ在(a, b)内可导，需要证明deriv φ在[a, b]上连续
      -- 简化处理：假设deriv φ在[a, b]上连续（这需要额外的条件）
      -- 在实际应用中，通常需要假设deriv φ在[a, b]上连续
      -- 这里我们假设这个条件已经满足
      sorry -- TODO: 从DifferentiableAt推导deriv的连续性（需要额外条件）
    have h_product_cont : ContinuousOn (fun x => f (φ x) * deriv φ x) (Set.Icc a b) := by
      -- 两个连续函数的乘积连续
      exact ContinuousOn.mul h_f_comp_cont h_deriv_cont
    -- 连续函数在闭区间上可积
    exact continuousOn_intervalIntegrable h_product_cont hab.le

  -- 使用integral_comp_smul_deriv
  exact intervalIntegral.integral_comp_smul_deriv hab hf hφ hφ' h_integrable

-- 分部积分法
theorem integration_by_parts
  {u v : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
  (hu_cont : ContinuousOn u (Set.Icc a b))
  (hu' : ∀ x ∈ Set.Ioo a b, DifferentiableAt ℝ u x)
  (hv_cont : ContinuousOn v (Set.Icc a b))
  (hv' : ∀ x ∈ Set.Ioo a b, DifferentiableAt ℝ v x) :
  ∫ x in a..b, u x * (deriv v x) =
    u b * v b - u a * v a - ∫ x in a..b, (deriv u x) * v x := by
  -- 使用mathlib4的integral_deriv_mul_eq_sub
  -- 需要先证明u * v'和u' * v在[a, b]上可积
  have h_integrable_uv' : IntervalIntegrable (fun x => u x * deriv v x) volume a b := by
    -- u在[a, b]上连续，deriv v在[a, b]上连续
    -- 因此u * deriv v在[a, b]上连续（乘积连续性）
    -- 连续函数在闭区间上可积
    have h_product_cont : ContinuousOn (fun x => u x * deriv v x) (Set.Icc a b) := by
      -- 需要证明deriv v在[a, b]上连续
      -- 简化处理：假设deriv v在[a, b]上连续（这需要额外的条件）
      have h_deriv_v_cont : ContinuousOn (deriv v) (Set.Icc a b) := by
        sorry -- TODO: 从DifferentiableAt推导deriv的连续性（需要额外条件）
      exact ContinuousOn.mul hu_cont h_deriv_v_cont
    exact continuousOn_intervalIntegrable h_product_cont hab.le

  have h_integrable_u'v : IntervalIntegrable (fun x => deriv u x * v x) volume a b := by
    -- deriv u在[a, b]上连续，v在[a, b]上连续
    -- 因此deriv u * v在[a, b]上连续（乘积连续性）
    -- 连续函数在闭区间上可积
    have h_product_cont : ContinuousOn (fun x => deriv u x * v x) (Set.Icc a b) := by
      -- 需要证明deriv u在[a, b]上连续
      -- 简化处理：假设deriv u在[a, b]上连续（这需要额外的条件）
      have h_deriv_u_cont : ContinuousOn (deriv u) (Set.Icc a b) := by
        sorry -- TODO: 从DifferentiableAt推导deriv的连续性（需要额外条件）
      exact ContinuousOn.mul h_deriv_u_cont hv_cont
    exact continuousOn_intervalIntegrable h_product_cont hab.le

  -- 使用integral_deriv_mul_eq_sub
  -- 需要构造(u * v)' = u' * v + u * v'
  have h_product_deriv : ∀ x ∈ Set.Ioo a b, HasDerivAt (fun x => u x * v x) (deriv u x * v x + u x * deriv v x) x := by
    intro x hx
    -- 使用乘积法则
    have hu_deriv : HasDerivAt u (deriv u x) x := by
      exact DifferentiableAt.hasDerivAt (hu' x hx)
    have hv_deriv : HasDerivAt v (deriv v x) x := by
      exact DifferentiableAt.hasDerivAt (hv' x hx)
    exact HasDerivAt.mul hu_deriv hv_deriv

  -- 使用微积分基本定理II
  have h_fundamental : ∫ x in a..b, (deriv u x * v x + u x * deriv v x) =
    (u b * v b) - (u a * v a) := by
    have h_cont : ContinuousOn (fun x => u x * v x) (Set.Icc a b) := by
      exact ContinuousOn.mul hu_cont hv_cont
    exact fundamental_theorem_calculus_II hab h_cont h_product_deriv
      (intervalIntegrable.add h_integrable_u'v h_integrable_uv')

  -- 由积分的线性性
  have h_linear : ∫ x in a..b, (deriv u x * v x + u x * deriv v x) =
    ∫ x in a..b, (deriv u x * v x) + ∫ x in a..b, (u x * deriv v x) := by
    exact intervalIntegral.integral_add h_integrable_u'v h_integrable_uv'

  -- 整理得到分部积分公式
  rw [h_linear] at h_fundamental
  linarith [h_fundamental]

-- ============================================
-- 级数理论定理
-- ============================================

-- 级数收敛定义（使用mathlib4标准定义）
def SeriesConverges (a : ℕ → ℝ) : Prop :=
  ∃ S, Filter.Tendsto (fun n => ∑ k in Finset.range n, a k) Filter.atTop (𝓝 S)

-- Cauchy准则
theorem series_converges_iff_cauchy (a : ℕ → ℝ) :
  SeriesConverges a ↔
  ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N →
    |∑ k in Finset.Ico n m, a k| < ε := by
  -- 使用Cauchy收敛准则
  -- 级数收敛当且仅当部分和序列是Cauchy序列
  constructor
  · -- 收敛蕴含Cauchy
    intro h_conv
    obtain ⟨S, h_tendsto⟩ := h_conv
    -- 使用tendsto_cauchy_seq
    intro ε hε
    have : ∀ᶠ n in Filter.atTop, |(∑ k in Finset.range n, a k) - S| < ε / 2 := by
      exact tendsto_def.mp h_tendsto (Metric.ball S (ε / 2)) (Metric.ball_mem_nhds S (half_pos hε))
    obtain ⟨N, hN⟩ := eventually_atTop.mp this
    use N
    intro m n hm hn
    -- 需要证明|∑[n,m] a k| < ε
    -- 使用三角不等式：|∑[n,m] a k| = |s(m) - s(n)| ≤ |s(m) - S| + |s(n) - S|
    have h_m : |(∑ k in Finset.range m, a k) - S| < ε / 2 := hN m hm
    have h_n : |(∑ k in Finset.range n, a k) - S| < ε / 2 := hN n hn
    -- 计算∑[n,m] a k = s(m) - s(n)（假设m ≥ n）
    by_cases h_le : n ≤ m
    · -- n ≤ m的情况
      have h_sum_eq : ∑ k in Finset.Ico n m, a k = (∑ k in Finset.range m, a k) - (∑ k in Finset.range n, a k) := by
        rw [Finset.sum_Ico_eq_sub _ h_le]
      rw [h_sum_eq]
      -- 使用三角不等式
      have : |(∑ k in Finset.range m, a k) - (∑ k in Finset.range n, a k)| ≤
        |(∑ k in Finset.range m, a k) - S| + |(∑ k in Finset.range n, a k) - S| := by
        exact abs_sub_le _ _ _
      linarith
    · -- m < n的情况，交换顺序
      push_neg at h_le
      have h_sum_eq : ∑ k in Finset.Ico n m, a k = -(∑ k in Finset.Ico m n, a k) := by
        rw [Finset.sum_Ico_eq_sub _ (le_of_lt h_le)]
        ring
      rw [h_sum_eq, abs_neg]
      -- 类似地处理
      have h_sum_eq' : ∑ k in Finset.Ico m n, a k = (∑ k in Finset.range n, a k) - (∑ k in Finset.range m, a k) := by
        rw [Finset.sum_Ico_eq_sub _ (le_of_lt h_le)]
      rw [h_sum_eq']
      have : |(∑ k in Finset.range n, a k) - (∑ k in Finset.range m, a k)| ≤
        |(∑ k in Finset.range n, a k) - S| + |(∑ k in Finset.range m, a k) - S| := by
        exact abs_sub_le _ _ _
      linarith
  · -- Cauchy蕴含收敛
    intro h_cauchy
    -- 构造部分和序列
    let s : ℕ → ℝ := fun n => ∑ k in Finset.range n, a k
    -- 证明s是Cauchy序列
    have h_s_cauchy : CauchySeq s := by
      -- 从h_cauchy得出s是Cauchy序列
      -- 使用Metric.cauchySeq_iff
      rw [Metric.cauchySeq_iff]
      intro ε hε
      obtain ⟨N, hN⟩ := h_cauchy ε hε
      use N
      intro m n hm hn
      -- 需要证明|s(m) - s(n)| < ε
      -- 这需要从h_cauchy得出
      by_cases h_le : n ≤ m
      · -- n ≤ m的情况
        have h_sum_eq : s m - s n = ∑ k in Finset.Ico n m, a k := by
          simp [s]
          rw [Finset.sum_Ico_eq_sub _ h_le]
        rw [h_sum_eq]
        exact hN m n hm hn
      · -- m < n的情况
        push_neg at h_le
        have h_sum_eq : s m - s n = -(∑ k in Finset.Ico m n, a k) := by
          simp [s]
          rw [Finset.sum_Ico_eq_sub _ (le_of_lt h_le)]
          ring
        rw [h_sum_eq, abs_neg]
        exact hN n m hn hm
    -- 由实数完备性，Cauchy序列收敛
    exact exists_tendsto_of_cauchySeq h_s_cauchy

-- 绝对收敛蕴含收敛
theorem abs_convergent_imp_convergent (a : ℕ → ℝ) :
  SeriesConverges (fun n => |a n|) → SeriesConverges a := by
  -- 使用Cauchy准则
  intro h_abs_conv
  -- 如果|a|的级数收敛，则a的级数也收敛
  -- 这需要从Cauchy准则和三角不等式得出
  have h_cauchy : ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N →
    |∑ k in Finset.Ico n m, |a k|| < ε := by
    -- 从h_abs_conv得出
    rw [series_converges_iff_cauchy] at h_abs_conv
    exact h_abs_conv
  -- 由三角不等式，|∑[n,m] a k| ≤ ∑[n,m] |a k|
  -- 因此a的级数也满足Cauchy准则
  have h_cauchy_a : ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N →
    |∑ k in Finset.Ico n m, a k| < ε := by
    intro ε hε
    obtain ⟨N, hN⟩ := h_cauchy ε hε
    use N
    intro m n hm hn
    -- 使用三角不等式
    have : |∑ k in Finset.Ico n m, a k| ≤ ∑ k in Finset.Ico n m, |a k| := by
      exact abs_sum_le_sum_abs _ _
    have h_sum : ∑ k in Finset.Ico n m, |a k| < ε := by
      -- 注意：h_cauchy给出的是|∑[n,m] |a k||，但∑[n,m] |a k| ≥ 0，所以绝对值可以去掉
      have : |∑ k in Finset.Ico n m, |a k|| = ∑ k in Finset.Ico n m, |a k| := by
        exact abs_of_nonneg (Finset.sum_nonneg fun _ _ => abs_nonneg _)
      rw [← this]
      exact hN m n hm hn
    linarith
  -- 由Cauchy准则，a的级数收敛
  exact (series_converges_iff_cauchy a).mpr h_cauchy_a

-- 比值判别法
theorem ratio_test (a : ℕ → ℝ) (ha : ∀ n, a n > 0) :
  let ρ := liminf (fun n => a (n + 1) / a n) Filter.atTop
  (ρ < 1 → SeriesConverges a) ∧ (ρ > 1 → ¬SeriesConverges a) := by
  intro ρ
  constructor
  · -- ρ < 1 蕴含收敛
    intro h_ρ_lt_one
    -- 如果liminf < 1，则存在N使得对所有n ≥ N，a(n+1)/a(n) < r < 1
    -- 因此a(n) < a(N) * r^(n-N)，由几何级数比较得出收敛
    sorry -- TODO: 使用几何级数比较
  · -- ρ > 1 蕴含发散
    intro h_ρ_gt_one
    -- 如果liminf > 1，则存在N使得对所有n ≥ N，a(n+1)/a(n) > 1
    -- 因此a(n)不趋于0，级数发散
    by_contra h_conv
    have : Filter.Tendsto a Filter.atTop (𝓝 0) := by
      -- 如果级数收敛，则通项趋于0
      sorry -- TODO: 使用级数收敛的性质
    -- 但ρ > 1意味着a(n)不趋于0，矛盾
    sorry -- TODO: 完成矛盾证明

-- 根式判别法
theorem root_test (a : ℕ → ℝ) (ha : ∀ n, a n ≥ 0) :
  let ρ := limsup (fun n => (a n) ^ (1 / n : ℝ)) Filter.atTop
  (ρ < 1 → SeriesConverges a) ∧ (ρ > 1 → ¬SeriesConverges a) := by
  intro ρ
  constructor
  · -- ρ < 1 蕴含收敛
    intro h_ρ_lt_one
    -- 如果limsup < 1，则存在N使得对所有n ≥ N，a(n)^(1/n) < r < 1
    -- 因此a(n) < r^n，由几何级数比较得出收敛
    sorry -- TODO: 使用几何级数比较
  · -- ρ > 1 蕴含发散
    intro h_ρ_gt_one
    -- 如果limsup > 1，则存在无穷多个n使得a(n)^(1/n) > 1
    -- 因此a(n) > 1，通项不趋于0，级数发散
    by_contra h_conv
    have : Filter.Tendsto a Filter.atTop (𝓝 0) := by
      -- 如果级数收敛，则通项趋于0
      sorry -- TODO: 使用级数收敛的性质
    -- 但ρ > 1意味着a(n)不趋于0，矛盾
    sorry -- TODO: 完成矛盾证明

-- Leibniz交错级数判别法
theorem leibniz_test (a : ℕ → ℝ)
  (ha_pos : ∀ n, a n > 0)
  (ha_decr : ∀ n, a (n + 1) ≤ a n)
  (ha_lim : Filter.Tendsto a Filter.atTop (𝓝 0)) :
  SeriesConverges (fun n => (-1) ^ n * a n) := by
  -- 使用部分和单调有界
  -- 交错级数的部分和序列有界且单调
  let s : ℕ → ℝ := fun n => ∑ k in Finset.range n, (-1) ^ k * a k
  -- 证明s(2n)单调递增有上界，s(2n+1)单调递减有下界
  -- 且它们的极限相等
  have h_even_mono : ∀ n, s (2 * n) ≤ s (2 * (n + 1)) := by
    intro n
    -- s(2n+2) - s(2n) = (-1)^(2n+1) * a(2n+1) + (-1)^(2n+2) * a(2n+2)
    -- = -a(2n+1) + a(2n+2) ≥ 0（因为a递减）
    sorry -- TODO: 完成证明
  have h_even_bounded : ∃ M, ∀ n, s (2 * n) ≤ M := by
    -- s(2n) ≤ s(1) = a(0)
    use a 0
    intro n
    sorry -- TODO: 完成证明
  have h_odd_mono : ∀ n, s (2 * n + 3) ≤ s (2 * n + 1) := by
    intro n
    -- 类似地，s(2n+1)单调递减
    sorry -- TODO: 完成证明
  have h_odd_bounded : ∃ m, ∀ n, m ≤ s (2 * n + 1) := by
    -- s(2n+1) ≥ s(0) = 0
    use 0
    intro n
    sorry -- TODO: 完成证明
  -- 由单调有界定理，s(2n)和s(2n+1)都收敛
  -- 且它们的极限相等（因为a(n) → 0）
  sorry -- TODO: 完成收敛性证明

-- 幂级数收敛半径
def PowerSeriesRadius (a : ℕ → ℝ) : ℝ :=
  -- 使用Cauchy-Hadamard公式
  -- R = 1 / limsup |a(n)|^(1/n)
  let L := limsup (fun n => (|a n|) ^ (1 / n : ℝ)) Filter.atTop
  if L = 0 then ⊤ else if L = ⊤ then 0 else 1 / L

-- 幂级数在收敛半径内连续
theorem power_series_continuous_in_radius
  (a : ℕ → ℝ) (R : ℝ) :
  let f := fun x => ∑' n, a n * x ^ n
  (PowerSeriesRadius a = R) →
  ∀ x ∈ Set.Ioo (-R) R, ContinuousAt f x := by
  -- 使用一致收敛性
  -- 在收敛半径内的任意紧致集上，幂级数一致收敛
  intro f h_radius x hx
  -- 需要证明f在x处连续
  -- 这需要证明幂级数在x的邻域内一致收敛
  sorry -- TODO: 使用一致收敛性和连续性

end Exercises.Analysis
