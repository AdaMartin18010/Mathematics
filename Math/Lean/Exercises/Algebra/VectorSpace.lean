-- 向量空间理论练习 | Vector Space Theory Exercises
-- 对齐国际标准：MIT 18.06、Stanford Math 113、Cambridge Part II
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
-- 向量空间基础定理（使用mathlib4标准定义）
-- ============================================

-- 子空间判别法
theorem subspace_criterion {K V : Type*} [Field K] [AddCommGroup V] [Module K V]
  (W : Set V) (h_ne : W.Nonempty) :
  (∀ a b : K, ∀ u v : V, u ∈ W → v ∈ W → a • u + b • v ∈ W) ↔
  ∃ (S : Submodule K V), ↑S = W := by
  -- 使用mathlib4的Submodule定义
  constructor
  · -- 充分性：如果W满足子空间判别条件，则W是子空间
    intro h_criterion
    -- 构造子空间
    use {
      carrier := W
      add_mem' := by
        -- 需要证明：如果u, v ∈ W，则u + v ∈ W
        intro u v hu hv
        -- 由判别条件，1 • u + 1 • v = u + v ∈ W
        have : (1 : K) • u + (1 : K) • v ∈ W := h_criterion 1 1 u v hu hv
        simpa [one_smul] using this
      zero_mem' := by
        -- 需要证明：0 ∈ W
        -- 由h_ne，存在u ∈ W
        obtain ⟨u, hu⟩ := h_ne
        -- 由判别条件，0 • u + 0 • v = 0 ∈ W
        have : (0 : K) • u + (0 : K) • u ∈ W := h_criterion 0 0 u u hu hu
        simpa [zero_smul, add_zero] using this
      smul_mem' := by
        -- 需要证明：如果a : K，u ∈ W，则a • u ∈ W
        intro a u hu
        -- 由判别条件，a • u + 0 • u = a • u ∈ W
        have : a • u + (0 : K) • u ∈ W := h_criterion a 0 u u hu hu
        simpa [zero_smul, add_zero] using this
    }
    rfl
  · -- 必要性：如果W是子空间，则满足判别条件
    intro ⟨S, h_eq⟩
    intro a b u v hu hv
    rw [← h_eq] at hu hv
    -- 由子空间性质，a • u + b • v ∈ S
    exact S.add_mem (S.smul_mem a hu) (S.smul_mem b hv)

-- 线性无关的定义（使用mathlib4标准定义）
-- 注：mathlib4中已有LinearIndependent定义，这里提供等价形式
def LinearIndependent' {K V : Type*} [Field K] [AddCommGroup V] [Module K V]
  {ι : Type*} (v : ι → V) : Prop :=
  LinearIndependent K v

-- 维数的良定义性
theorem dimension_well_defined {K V : Type*} [Field K] [AddCommGroup V] [Module K V]
  [FiniteDimensional K V] (B1 B2 : Basis ι K V) :
  Fintype.card ι = Fintype.card (Basis.indexType B2) := by
  -- 使用mathlib4的Basis.mk_eq_dim
  -- 如果B1和B2都是V的基，则它们的维数相等
  have h1 : FiniteDimensional.finrank K V = Fintype.card ι := by
    exact Basis.mk_eq_dim B1
  have h2 : FiniteDimensional.finrank K V = Fintype.card (Basis.indexType B2) := by
    exact Basis.mk_eq_dim B2
  linarith

-- 子空间维数定理
theorem subspace_dimension {K V : Type*} [Field K] [AddCommGroup V] [Module K V]
  [FiniteDimensional K V] (W : Submodule K V) :
  FiniteDimensional.finrank K W ≤ FiniteDimensional.finrank K V := by
  -- 使用mathlib4的Submodule.finrank_le
  exact Submodule.finrank_le W

-- 子空间维数公式
theorem subspace_dimension_formula {K V : Type*} [Field K] [AddCommGroup V] [Module K V]
  [FiniteDimensional K V] (W1 W2 : Submodule K V) :
  FiniteDimensional.finrank K (W1 ⊔ W2) + FiniteDimensional.finrank K (W1 ⊓ W2) =
    FiniteDimensional.finrank K W1 + FiniteDimensional.finrank K W2 := by
  -- 使用mathlib4的Submodule.rank_sup_add_rank_inf_eq
  -- 对于有限维情况，使用finrank版本
  exact Submodule.finrank_sup_add_finrank_inf_eq W1 W2

-- ============================================
-- 线性变换基础定理（使用mathlib4标准定义）
-- ============================================

-- 不同特征值对应的特征向量线性无关
theorem eigenvectors_linear_independent {K V : Type*} [Field K] [AddCommGroup V] [Module K V]
  {ι : Type*} [Fintype ι] (T : Module.End K V)
  (λ : ι → K) (v : ι → V)
  (h_eigen : ∀ i, T (v i) = λ i • v i)
  (h_ne : ∀ i j, i ≠ j → λ i ≠ λ j)
  (h_nonzero : ∀ i, v i ≠ 0) :
  LinearIndependent K v := by
  -- 使用mathlib4的Module.End.eigenvectors_linearIndependent'
  -- 需要证明λ是单射的
  have h_injective : Function.Injective λ := by
    intro i j h_eq
    -- 如果λ i = λ j，且i ≠ j，则与h_ne矛盾
    by_contra h_ne_ij
    have : λ i ≠ λ j := h_ne i j h_ne_ij
    contradiction
  -- 需要证明每个v i是特征向量
  have h_eigenvec : ∀ i, T.HasEigenvector (λ i) (v i) := by
    intro i
    constructor
    · -- v i ≠ 0
      exact h_nonzero i
    · -- T (v i) = λ i • v i
      exact h_eigen i
  -- 使用Module.End.eigenvectors_linearIndependent'
  exact Module.End.eigenvectors_linearIndependent' T λ h_injective v h_eigenvec

end Exercises.Algebra
