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

-- 练习2：连续函数的基本性质
-- 对应哈佛大学拓扑课程标准
theorem continuous_comp (X Y Z : Type) [TopologicalSpace X] [TopologicalSpace Y] [TopologicalSpace Z]
  (f : X → Y) (g : Y → Z) :
  Continuous f → Continuous g → Continuous (g ∘ f) := by
  -- HINT: 连续函数的复合仍连续；检索 `Continuous.comp`
  exact Continuous.comp

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
  sorry

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

end TopologyExercises
