# Phase 3剩余问题修复方案

**创建日期**: 2025-12-21
**版本**: v1.0
**状态**: 进行中
**参考**: [Phase 3问题修复计划](./内容质量检查-Phase3问题修复计划.md)

---

## 📋 剩余问题概览

### P0问题剩余：7个sorry占位符

1. **级数判别法**（4处）- 结构已重构，需API
2. **幂级数连续性**（1处）- 需API
3. **拓扑学证明**（6处）- 需API

---

## 🔧 修复方案

### 1. 级数判别法（4处sorry）

#### 问题1-2: ratio_test的eventually条件推导

**位置**: `Real.lean` 第1110、1128行

**修复方法**:

```lean
-- 从liminf < 1推导eventually条件
have h_eventually : ∃ r < 1, ∃ N, ∀ n ≥ N, a (n + 1) / a n < r := by
  -- 使用liminf的性质：liminf f < c → ∃ᶠ n, f n < c'
  -- 在mathlib4中，可能需要使用：
  -- Filter.liminf_lt_iff_eventually_lt 或类似API
  -- 如果API不存在，可以使用liminf的定义展开
  sorry -- 需要查找mathlib4的liminf API

-- 从liminf > 1推导frequently条件
have h_frequently : ∃ᶠ n in Filter.atTop, a (n + 1) / a n > 1 := by
  -- 使用liminf的性质：liminf f > c → ∃ᶠ n, f n > c'
  -- 在mathlib4中，可能需要使用：
  -- Filter.liminf_gt_iff_frequently_gt 或类似API
  sorry -- 需要查找mathlib4的liminf API
```

**API查找方向**:
- `Filter.liminf_lt_iff_eventually_lt`
- `Filter.liminf_gt_iff_frequently_gt`
- `Filter.liminf_le_iff_eventually_le`
- `Filter.liminf_ge_iff_frequently_ge`

#### 问题3-4: root_test的eventually条件推导

**位置**: `Real.lean` 第1166、1204行

**修复方法**: 类似ratio_test，使用limsup API

**API查找方向**:
- `Filter.limsup_lt_iff_eventually_lt`
- `Filter.limsup_gt_iff_frequently_gt`

#### 问题5-6: 几何级数收敛和比较判别法

**位置**: `Real.lean` 第1114、1119、1179、1184行

**修复方法**:

```lean
-- 几何级数收敛
have h_geom_conv : SeriesConverges (fun n => r^n) := by
  -- 使用几何级数收敛定理
  -- 在mathlib4中，可能需要使用：
  -- HasSum.geometric_series 或类似API
  -- 如果r < 1，则∑r^n = 1/(1-r)
  sorry -- 需要查找mathlib4的几何级数API

-- 比较判别法
have h_conv : SeriesConverges a := by
  -- 使用比较判别法：如果0 ≤ a(n) ≤ b(n)且∑b(n)收敛，则∑a(n)收敛
  -- 在mathlib4中，可能需要使用：
  -- Summable.of_nonneg_of_le 或类似API
  sorry -- 需要查找mathlib4的比较判别法API
```

**API查找方向**:
- `HasSum.geometric_series`
- `Summable.geometric_series`
- `Summable.of_nonneg_of_le`
- `Summable.of_nonneg_of_eventually_le`

---

### 2. 幂级数连续性（1处sorry）

#### 问题: power_series_continuous_in_radius

**位置**: `Real.lean` 第1501行

**修复方法**:

```lean
-- 使用一致收敛性和连续性
have h_cont : ContinuousAt f x := by
  -- 方法1: 使用Weierstrass M-判别法证明一致收敛
  -- 方法2: 使用幂级数的一致收敛性定理
  -- 方法3: 直接使用幂级数的连续性定理

  -- 在mathlib4中，可能需要使用：
  -- PowerSeries.continuousOn_ball
  -- UniformConvergesOn.continuous
  -- 或类似API
  sorry -- 需要查找mathlib4的幂级数连续性API
```

**API查找方向**:
- `PowerSeries.continuousOn_ball`
- `UniformConvergesOn.continuous`
- `UniformConvergence.continuous`
- `WeierstrassMTest`

---

### 3. 拓扑学证明（6处sorry）

#### 问题1-5: 粘接引理的连续性证明

**位置**: `Topology/Basic.lean` 第234、249、273、280、295行

**修复方法**:

```lean
-- 问题1: h在A上连续
have h_cont_on_A : ContinuousOn h A := by
  -- 由于h在A上等于f，且f连续（作为A → Y）
  -- 需要使用ContinuousOn的定义
  -- 在mathlib4中，可能需要使用：
  -- ContinuousOn.restrict
  -- ContinuousOn.codRestrict
  -- 或直接使用连续性定义
  sorry -- 需要查找mathlib4的连续性API

-- 问题2: h在B上连续（类似）

-- 问题3-5: h在A ∪ B上连续
have h_cont_on_union : ContinuousOn h (A ∪ B) := by
  -- 使用ContinuousOn.union或类似API
  -- 需要A和B都是闭集，且h在A ∩ B上一致
  sorry -- 需要查找mathlib4的连续性API
```

**API查找方向**:
- `ContinuousOn.restrict`
- `ContinuousOn.codRestrict`
- `ContinuousOn.union`
- `ContinuousOn.union'`

#### 问题6: 标准正交基构造

**位置**: `Topology/Basic.lean` 第390行

**修复方法**:

```lean
have h_orthonormal_basis : OrthonormalBasis ι ℝ V := by
  -- 从Basis和Orthonormal构造OrthonormalBasis
  -- 在mathlib4中，可能需要使用：
  -- OrthonormalBasis.mk
  -- OrthonormalBasis.ofRepr
  -- 或类似API
  sorry -- 需要查找mathlib4的标准正交基API
```

**API查找方向**:
- `OrthonormalBasis.mk`
- `OrthonormalBasis.ofRepr`
- `Basis.toOrthonormalBasis`

#### 问题7: 逆函数定理

**位置**: `Topology/Basic.lean` 第456行

**修复方法**:

```lean
have h_inv : ... := by
  -- 使用逆函数定理
  -- 在mathlib4中，可能需要使用：
  -- inverseFunctionTheorem
  -- InverseFunctionTheorem.exists_nhds_slice
  -- 或流形版本的逆函数定理
  sorry -- 需要查找mathlib4的逆函数定理API
```

**API查找方向**:
- `InverseFunctionTheorem.exists_nhds_slice`
- `InverseFunctionTheorem.toLocalHomeomorph`
- 流形版本的逆函数定理

---

## 📚 API查找资源

### Mathlib4文档

1. **Filter API**
   - [Filter文档](https://leanprover-community.github.io/mathlib4_docs/find/?pattern=Filter.liminf)
   - [Filter.liminf文档](https://leanprover-community.github.io/mathlib4_docs/find/?pattern=liminf)

2. **级数API**
   - [Summable文档](https://leanprover-community.github.io/mathlib4_docs/find/?pattern=Summable)
   - [HasSum文档](https://leanprover-community.github.io/mathlib4_docs/find/?pattern=HasSum)

3. **连续性API**
   - [ContinuousOn文档](https://leanprover-community.github.io/mathlib4_docs/find/?pattern=ContinuousOn)
   - [UniformConvergence文档](https://leanprover-community.github.io/mathlib4_docs/find/?pattern=UniformConvergence)

4. **拓扑学API**
   - [OrthonormalBasis文档](https://leanprover-community.github.io/mathlib4_docs/find/?pattern=OrthonormalBasis)
   - [InverseFunctionTheorem文档](https://leanprover-community.github.io/mathlib4_docs/find/?pattern=InverseFunctionTheorem)

### 搜索方法

1. **在mathlib4中搜索**
   ```bash
   # 搜索liminf相关API
   grep -r "liminf" Mathlib/

   # 搜索几何级数相关API
   grep -r "geometric" Mathlib/

   # 搜索比较判别法相关API
   grep -r "Summable.*le" Mathlib/
   ```

2. **使用Lean 4的#check命令**
   ```lean
   #check Filter.liminf
   #check Summable.geometric_series
   #check ContinuousOn.union
   ```

3. **查看mathlib4源码**
   - 查看`Mathlib/Analysis/Filter.lean`
   - 查看`Mathlib/Analysis/Series.lean`
   - 查看`Mathlib/Topology/Continuous.lean`

---

## 🎯 修复优先级

### 高优先级（立即修复）

1. ✅ 导数连续性（3处）- **已完成**
2. ⏳ 级数判别法（4处）- **结构已重构，需API**
3. ⏳ 幂级数连续性（1处）- **需API**

### 中优先级（短期修复）

4. ⏳ 拓扑学证明（6处）- **需API**

### 低优先级（长期修复）

5. ⏳ P1问题（18个）- **需要长期工作**
6. ⏳ P2问题（8个）- **需要长期工作**

---

## 📋 修复检查清单

### 级数判别法

- [ ] 查找liminf/limsup API
- [ ] 完成eventually条件推导（2处）
- [ ] 完成frequently条件推导（2处）
- [ ] 查找几何级数收敛API
- [ ] 完成几何级数收敛证明（2处）
- [ ] 查找比较判别法API
- [ ] 完成比较判别法证明（2处）

### 幂级数连续性

- [ ] 查找一致收敛性API
- [ ] 查找幂级数连续性API
- [ ] 完成连续性证明（1处）

### 拓扑学证明

- [ ] 查找连续性API（粘接引理）
- [ ] 完成粘接引理证明（5处）
- [ ] 查找标准正交基API
- [ ] 完成标准正交基构造（1处）
- [ ] 查找逆函数定理API
- [ ] 完成逆函数定理应用（1处）

---

## 🔗 相关文档

- [Phase 3问题修复计划](./内容质量检查-Phase3问题修复计划.md)
- [Phase 3-P0问题修复指南](./内容质量检查-Phase3-P0问题修复指南.md)
- [Phase 3修复进度报告](./内容质量检查-Phase3修复进度报告.md)

---

**最后更新**: 2025-12-21
**状态**: 进行中
**下一步**: 查找mathlib4 API，完成剩余修复
