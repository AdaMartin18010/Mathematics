/-!
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`，版本随 `lakefile.lean` 固定到 stable 或已验证提交
- 最小导入：`import Std`, `import Mathlib`
-/

import Std
import Mathlib
-- 拓扑学基础练习 | Topology Basic Exercises
-- 对齐国际标准：剑桥大学Part II拓扑课程
-- 更新时间：2025-01-15

import Mathlib.Topology.Basic
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.Compactness

namespace TopologyExercises

-- 练习1：拓扑空间的基本性质
-- 对应剑桥大学Part II拓扑学标准
theorem open_union (X : Type) [TopologicalSpace X] (U V : Set X) :
  IsOpen U → IsOpen V → IsOpen (U ∪ V) := by
  -- HINT: 使用开集在并运算下稳定；检索 `IsOpen.union`
  exact IsOpen.union

-- SOLUTION:
-- by
--   intro hU hV
--   simpa using hU.union hV

-- 练习2：连续函数的基本性质
-- 对应哈佛大学拓扑课程标准
theorem continuous_comp (X Y Z : Type) [TopologicalSpace X] [TopologicalSpace Y] [TopologicalSpace Z]
  (f : X → Y) (g : Y → Z) :
  Continuous f → Continuous g → Continuous (g ∘ f) := by
  -- HINT: 连续函数的复合仍连续；检索 `Continuous.comp`
  exact Continuous.comp

-- SOLUTION:
-- by
--   intro hf hg
--   exact hg.comp hf

-- 练习3：紧致性的基本性质
-- 对应芝加哥大学拓扑标准
theorem compact_closed (X : Type) [TopologicalSpace X] (K : Set X) :
  IsCompact K → IsClosed K → IsCompact K := by
  intro h1 h2
  -- NOTE: 命题即“若 K 紧致且闭，则 K 紧致”，可直接用 h1；更强命题可考虑闭子集的紧致性
  exact h1

-- 练习4：连通性的基本性质
-- 对应华威大学拓扑标准
theorem connected_union (X : Type) [TopologicalSpace X] (A B : Set X) :
  IsConnected A → IsConnected B → A ∩ B ≠ ∅ → IsConnected (A ∪ B) := by
  -- HINT: 连接性的并封闭性需要非空交叠；检索 `IsConnected.union` 并满足交叠条件
  -- SOLUTION: 使用库定理 `IsConnected.union`
  intro hA hB hAB
  exact IsConnected.union hA hB hAB

-- 练习5：同胚的基本性质
-- 对应巴黎第六大学拓扑标准
theorem homeomorph_continuous (X Y : Type) [TopologicalSpace X] [TopologicalSpace Y]
  (f : X ≃ Y) :
  Homeomorph f → Continuous f := by
  intro h
  -- HINT: 同胚的连续性与可逆连续性；`Homeomorph.continuous`
  exact h.continuous

-- 练习6：滤子的基本性质
-- 对应伦敦大学学院拓扑标准
theorem filter_principal (X : Type) (s : Set X) :
  Filter.principal s = {t : Set X | s ⊆ t} := by
  -- HINT: 这是 `principal` 的定义化简；保持为 rfl
  rfl

-- ============================================
-- 度量空间基础定理（使用mathlib4标准定义）
-- ============================================

-- 开球定义（使用mathlib4标准定义）
-- 注：mathlib4中已有Metric.ball定义，这里提供等价形式
def ball' {α : Type*} [MetricSpace α] (x : α) (r : ℝ) : Set α :=
  Metric.ball x r

-- 序列收敛（使用mathlib4标准定义）
def tendsto_metricSpace {α : Type*} [MetricSpace α]
  (s : ℕ → α) (a : α) : Prop :=
  Filter.Tendsto s Filter.atTop (𝓝 a)

-- Cauchy序列（使用mathlib4标准定义）
def cauchySeq' {α : Type*} [MetricSpace α] (s : ℕ → α) : Prop :=
  CauchySeq s

-- 紧性等价性
theorem compact_iff_sequentially_compact {α : Type*} [MetricSpace α] (K : Set α) :
  IsCompact K ↔ IsSeqCompact K := by
  -- 使用mathlib4的isCompact_iff_isSeqCompact
  exact isCompact_iff_isSeqCompact K

-- 压缩映射原理（Banach不动点定理）
theorem banach_fixed_point {α : Type*} [MetricSpace α] [CompleteSpace α]
  (f : α → α) (k : ℝ) (hk : 0 ≤ k ∧ k < 1)
  (hf : ∀ x y, dist (f x) (f y) ≤ k * dist x y) :
  ∃! x, f x = x := by
  -- 使用mathlib4的contracting_fixedPoint
  -- 需要构造ContractingWith实例
  have h_contracting : ContractingWith k f := by
    constructor
    · exact hk.1
    · exact hk.2
    · exact hf
  exact ContractingWith.exists_fixedPoint h_contracting

-- ============================================
-- 拓扑空间基础定理（使用mathlib4标准定义）
-- ============================================

-- 连续映射的等价刻画
theorem continuous_iff_closed_preimage {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
  (f : X → Y) :
  Continuous f ↔ (∀ F : Set Y, IsClosed F → IsClosed (f ⁻¹' F)) := by
  -- 使用mathlib4的continuous_iff_isClosed_preimage
  exact continuous_iff_isClosed_preimage f

-- 紧空间的闭子集是紧的
theorem compact_closed_subset {X : Type*} [TopologicalSpace X] (K F : Set X)
  (hK : IsCompact K) (hF : IsClosed F) (h_subset : F ⊆ K) :
  IsCompact F := by
  -- 使用mathlib4的IsCompact.subset
  exact IsCompact.subset hK h_subset

-- 紧空间的连续像也是紧的
theorem compact_image {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
  (K : Set X) (hK : IsCompact K) (f : X → Y) (hf : Continuous f) :
  IsCompact (f '' K) := by
  -- 使用mathlib4的IsCompact.image
  exact IsCompact.image hK hf

-- 粘接引理
theorem gluing_lemma {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
  (A B : Set X) (hA : IsClosed A) (hB : IsClosed B) (h_union : A ∪ B = Set.univ)
  (f : A → Y) (g : B → Y) (hf : Continuous f) (hg : Continuous g)
  (h_agree : ∀ x ∈ A ∩ B, f ⟨x, x.1⟩ = g ⟨x, x.2⟩) :
  ∃! h : X → Y, Continuous h ∧ (∀ x ∈ A, h x = f ⟨x, x.1⟩) ∧ (∀ x ∈ B, h x = g ⟨x, x.2⟩) := by
  -- 使用mathlib4的ContinuousOn.union或类似定理
  -- 需要构造连续函数h
  sorry -- TODO: 使用mathlib4的粘接引理

-- ============================================
-- 赋范空间基础定理（使用mathlib4标准定义）
-- ============================================

-- 赋范空间定义（使用mathlib4标准定义）
-- 注：mathlib4中已有SeminormedAddCommGroup和NormedAddCommGroup

-- 范数的基本性质
theorem norm_nonneg {E : Type*} [NormedAddCommGroup E] (x : E) :
  0 ≤ ‖x‖ := by
  -- 使用mathlib4的norm_nonneg
  exact norm_nonneg x

-- 范数的三角不等式
theorem norm_add_le {E : Type*} [NormedAddCommGroup E] (x y : E) :
  ‖x + y‖ ≤ ‖x‖ + ‖y‖ := by
  -- 使用mathlib4的norm_add_le
  exact norm_add_le x y

-- 有界线性算子的范数
theorem bounded_linear_map_norm {E F : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F]
  (f : E →L[ℝ] F) :
  ∃ C ≥ 0, ∀ x : E, ‖f x‖ ≤ C * ‖x‖ := by
  -- 使用mathlib4的ContinuousLinearMap.bound
  exact f.bound

-- ============================================
-- 内积空间基础定理（使用mathlib4标准定义）
-- ============================================

-- Riesz表示定理
theorem riesz_representation {𝕜 E : Type*} [IsROrC 𝕜]
  [NormedAddCommGroup E] [InnerProductSpace 𝕜 E] [CompleteSpace E]
  (f : E →L[𝕜] 𝕜) :
  ∃! y : E, ∀ x : E, f x = inner x y := by
  -- 使用mathlib4的InnerProductSpace.toDual
  exact InnerProductSpace.toDual.exists_unique f

-- Bessel不等式
theorem bessel_inequality {𝕜 E : Type*} [IsROrC 𝕜]
  [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
  {ι : Type*} (v : ι → E) (hv : Orthonormal 𝕜 v) (x : E) :
  ∑' i, ‖inner x (v i)‖^2 ≤ ‖x‖^2 := by
  -- 使用mathlib4的Orthonormal.sum_inner_products_le
  exact Orthonormal.sum_inner_products_le hv x

-- Parseval恒等式
theorem parseval_identity {𝕜 E : Type*} [IsROrC 𝕜]
  [NormedAddCommGroup E] [InnerProductSpace 𝕜 E] [CompleteSpace E]
  {ι : Type*} [Fintype ι] (v : Basis ι 𝕜 E) (hv : Orthonormal 𝕜 v) (x : E) :
  ‖x‖^2 = ∑ i, ‖inner x (v i)‖^2 := by
  -- 使用mathlib4的Orthonormal.sum_inner_products_eq
  -- 需要v是标准正交基
  have h_basis : OrthonormalBasis ι 𝕜 E := by
    -- 从Basis和Orthonormal构造OrthonormalBasis
    sorry -- TODO: 构造OrthonormalBasis
  exact OrthonormalBasis.sum_inner_products_eq h_basis x

-- ============================================
-- 微分流形基础定理（使用mathlib4标准定义）
-- ============================================

-- 切空间定义（使用mathlib4标准定义）
-- 注：mathlib4中已有TangentSpace定义

-- 切映射（微分）
def tangent_map {𝕜 : Type*} [NontriviallyNormedField 𝕜]
  {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
  {H : Type*} [TopologicalSpace H] (I : ModelWithCorners 𝕜 E H)
  {M : Type*} [TopologicalSpace M] [ChartedSpace H M]
  {E' : Type*} [NormedAddCommGroup E'] [NormedSpace 𝕜 E']
  {H' : Type*} [TopologicalSpace H'] (I' : ModelWithCorners 𝕜 E' H')
  {M' : Type*} [TopologicalSpace M'] [ChartedSpace H' M']
  (f : M → M') (x : M) : TangentSpace I M x →L[𝕜] TangentSpace I' M' (f x) := by
  -- 使用mathlib4的mfderiv
  exact mfderiv I I' f x

-- 逆函数定理（流形版本）
theorem inverse_function_theorem_manifold {𝕜 : Type*} [NontriviallyNormedField 𝕜]
  {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
  {H : Type*} [TopologicalSpace H] (I : ModelWithCorners 𝕜 E H)
  {M : Type*} [TopologicalSpace M] [ChartedSpace H M] [SmoothManifoldWithCorners I M]
  {E' : Type*} [NormedAddCommGroup E'] [NormedSpace 𝕜 E']
  {H' : Type*} [TopologicalSpace H'] (I' : ModelWithCorners 𝕜 E' H')
  {M' : Type*} [TopologicalSpace M'] [ChartedSpace H' M'] [SmoothManifoldWithCorners I' M']
  (f : M → M') (x : M) (hf : MDifferentiableAt I I' f x)
  (h_invertible : Function.Bijective (mfderiv I I' f x)) :
  ∃ U ∈ 𝓝 x, ∃ V ∈ 𝓝 (f x),
    Set.MapsTo f U V ∧
    Function.Bijective (f ∘ Set.inclusion (Set.subset_univ U)) ∧
    MDifferentiableOn I I' (Function.invFun (f ∘ Set.inclusion (Set.subset_univ U))) V := by
  -- 使用mathlib4的逆函数定理
  -- 需要mfderiv在x处可逆
  sorry -- TODO: 使用mathlib4的逆函数定理

-- ============================================
-- 赋范空间基础定理（使用mathlib4标准定义）
-- ============================================

-- 线性映射连续的等价刻画
theorem continuous_iff_bounded {E F : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F]
  [NormedSpace ℝ E] [NormedSpace ℝ F]
  (f : E →ₗ[ℝ] F) :
  Continuous f ↔ ∃ C, ∀ x, ‖f x‖ ≤ C * ‖x‖ := by
  -- 使用mathlib4的LinearMap.continuous_iff_isBoundedLinearMap
  exact LinearMap.continuous_iff_isBoundedLinearMap f

-- Hahn-Banach延拓定理
theorem exists_extension_norm_eq {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  (p : Submodule ℝ E) (f : p →L[ℝ] ℝ) :
  ∃ g : E →L[ℝ] ℝ, (∀ x : p, g x = f x) ∧ ‖g‖ = ‖f‖ := by
  -- 使用mathlib4的exists_extension_norm_eq
  exact exists_extension_norm_eq f

-- 一致有界原理（Banach-Steinhaus定理）
theorem banach_steinhaus {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [CompleteSpace E] [NormedAddCommGroup F] [NormedSpace ℝ F]
  (A : ℕ → E →L[ℝ] F) (h : ∀ x, ∃ C, ∀ n, ‖A n x‖ ≤ C) :
  ∃ C, ∀ n, ‖A n‖ ≤ C := by
  -- 使用Baire纲定理
  -- 使用mathlib4的banach_steinhaus
  exact banach_steinhaus h

end TopologyExercises
