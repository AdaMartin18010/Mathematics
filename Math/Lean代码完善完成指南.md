# Lean代码完善完成指南

**创建日期**: 2025年10月1日
**目标**: 完成剩余17个技术性sorry的证明

---

## 📋 剩余工作清单

### 1. 唯一分解定理（3个sorry）

#### 1.1 `Multiset.prod_map_mul`

**位置**: `Lean/Exercises/Algebra/Polynomial.lean` (约第153行)

**目标**: 证明 `(factors.map fun q => q * Polynomial.C q.leadingCoeff⁻¹).prod = factors.prod * (factors.map fun q => Polynomial.C q.leadingCoeff⁻¹).prod`

**完成步骤**:

1. 查找API: `Multiset.prod_map_mul`, `Multiset.prod_map_mul'`, `Multiset.prod_map_prod`
2. 如果API不存在，使用归纳法：

   ```lean
   -- 对Multiset的大小进行归纳
   -- 基础情况：空Multiset
   -- 归纳步骤：添加一个元素
   ```

3. 或者直接展开定义，使用`Multiset.prod_map`和乘法的性质

**参考**: 已添加详细的证明步骤和替代方案

---

#### 1.2 `Polynomial.C`的乘法同态性质

**位置**: `Lean/Exercises/Algebra/Polynomial.lean` (约第179行)

**目标**: 证明 `(factors.map fun q => Polynomial.C q.leadingCoeff⁻¹).prod = Polynomial.C (factors.map fun q => q.leadingCoeff⁻¹).prod`

**完成步骤**:

1. 查找API: `Multiset.prod_hom`, `Polynomial.map_prod`, `Polynomial.C_mul`, `Polynomial.C_prod`
2. 如果API不存在，使用归纳法：

   ```lean
   -- 对Multiset的大小进行归纳
   -- 使用Polynomial.C_mul: C(a) * C(b) = C(a * b)
   ```

3. 或者直接展开定义，使用C的乘法性质

**参考**: 已添加详细的证明步骤和替代方案

---

#### 1.3 `isUnit_iff_C`

**位置**: `Lean/Exercises/Algebra/Polynomial.lean` (约第275行)

**目标**: 证明单位多项式u = C(u.leadingCoeff)

**完成步骤**:

1. 查找API: `Polynomial.isUnit_iff`, `Polynomial.isUnit_iff_C`, `Polynomial.eq_C_of_degree_eq_zero`, `Polynomial.degree_eq_zero_of_isUnit`
2. 如果API不存在，使用degree和coeff的性质：

   ```lean
   -- 步骤1：证明u.degree = 0（使用isUnit的性质和degree_mul）
   -- 步骤2：使用eq_C_of_degree_eq_zero得到u = C(u.coeff 0)
   -- 步骤3：证明u.coeff 0 = u.leadingCoeff（当degree = 0时）
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

### 2. 粘接引理（4个sorry）

#### 2.1 `h_cont_on_A`

**位置**: `Lean/Exercises/Topology/Basic.lean` (约第210行)

**目标**: 证明h在A上连续（作为X → Y在A上的限制）

**完成步骤**:

1. 查找API: `ContinuousOn.restrict`, `ContinuousOn.codRestrict`, `Continuous.restrict`
2. 如果API不存在，使用连续性定义：

   ```lean
   -- 对于x ∈ A，由于h在A上等于f，且f连续（作为A → Y）
   -- 因此h在x处连续（相对于A的子空间拓扑）
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

#### 2.2 `h_cont_on_B`

**位置**: `Lean/Exercises/Topology/Basic.lean` (约第241行)

**目标**: 证明h在B上连续（作为X → Y在B上的限制）

**完成步骤**: 类似2.1，使用g的连续性

**参考**: 已添加详细的证明步骤和替代方案

---

#### 2.3 `h_cont_on_union`

**位置**: `Lean/Exercises/Topology/Basic.lean` (约第257行)

**目标**: 证明h在A ∪ B上连续

**完成步骤**:

1. 查找API: `ContinuousOn.union`, `ContinuousOn.union'`, `ContinuousOn.union_closed`
2. 如果API不存在，使用分情况讨论：

   ```lean
   -- 对于x ∈ A ∪ B，要么x ∈ A，要么x ∈ B
   -- 如果x ∈ A，使用h_cont_on_A
   -- 如果x ∈ B，使用h_cont_on_B
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

#### 2.4 `ContinuousOn.univ_iff`

**位置**: `Lean/Exercises/Topology/Basic.lean` (约第285行)

**目标**: 从`ContinuousOn h Set.univ`推导`Continuous h`

**完成步骤**:

1. 查找API: `ContinuousOn.univ_iff`, `ContinuousOn.continuous`, `ContinuousOn.continuous_on_univ`
2. 如果API不存在，使用连续性定义：

   ```lean
   -- 由于h在Set.univ上连续（作为限制），且Set.univ = X
   -- 因此对于任意x ∈ X，h在x处连续
   -- 因此h连续
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

### 3. 级数理论判别法（5个sorry）

#### 3.1 比值判别法收敛部分

**位置**: `Lean/Exercises/Analysis/Real.lean` (约第1130行)

**目标**: 使用liminf性质和几何级数比较判别法

**完成步骤**:

1. 查找API: `liminf_lt_iff_eventually_lt`, `Filter.eventually_atTop`, `HasSum.geometric_series`
2. 如果API不存在，使用归纳法：

   ```lean
   -- 步骤1：使用liminf_lt_iff_eventually_lt找到N和r
   -- 步骤2：通过归纳证明a(n) < a(N) * r^(n-N)对所有n ≥ N成立
   -- 步骤3：使用几何级数的收敛性和比较判别法
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

#### 3.2 比值判别法发散部分

**位置**: `Lean/Exercises/Analysis/Real.lean` (约第1190行)

**目标**: 使用liminf性质证明存在无穷多个n使得a(n+1)/a(n) > 1

**完成步骤**:

1. 查找API: `liminf_gt_iff_frequently_gt`, `Filter.frequently_atTop`, `Filter.frequently_iff`
2. 如果API不存在，使用子列性质：

   ```lean
   -- 步骤1：使用liminf_gt_iff_frequently_gt证明存在无穷多个n
   -- 步骤2：通过归纳证明a(n) > a(N)（对于某个N）
   -- 步骤3：使用级数收敛的必要条件
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

#### 3.3 根式判别法收敛部分

**位置**: `Lean/Exercises/Analysis/Real.lean` (约第1234行)

**目标**: 使用limsup性质和几何级数比较判别法

**完成步骤**:

1. 查找API: `limsup_lt_iff_eventually_lt`, `Filter.eventually_atTop`, `HasSum.geometric_series`
2. 如果API不存在，直接比较：

   ```lean
   -- 步骤1：使用limsup_lt_iff_eventually_lt找到N和r
   -- 步骤2：对所有n ≥ N，a(n) < r^n
   -- 步骤3：使用几何级数的收敛性和比较判别法
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

#### 3.4 根式判别法发散部分

**位置**: `Lean/Exercises/Analysis/Real.lean` (约第1290行)

**目标**: 使用limsup性质证明存在无穷多个n使得a(n) > 1

**完成步骤**:

1. 查找API: `limsup_gt_iff_frequently_gt`, `Filter.frequently_atTop`, `Filter.frequently_iff`
2. 如果API不存在，使用子列性质：

   ```lean
   -- 步骤1：使用limsup_gt_iff_frequently_gt证明存在无穷多个n使得a(n)^(1/n) > 1
   -- 步骤2：因此存在无穷多个n使得a(n) > 1
   -- 步骤3：使用级数收敛的必要条件
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

#### 3.5 幂级数连续性

**位置**: `Lean/Exercises/Analysis/Real.lean` (约第1546行)

**目标**: 使用一致收敛性和连续性

**完成步骤**:

1. 查找API: `UniformContinuous.continuous`, `UniformConvergence.continuous`, `PowerSeries.continuousOn_ball`
2. 如果API不存在，使用Weierstrass M-判别法：

   ```lean
   -- 步骤1：在收敛半径内的任意紧致集上，幂级数一致收敛
   -- 步骤2：一致收敛的连续函数序列的极限函数连续
   -- 步骤3：因此f在x处连续
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

### 4. deriv连续性（3个sorry）

#### 4.1 换元积分法：deriv φ的连续性

**位置**: `Lean/Exercises/Analysis/Real.lean` (约第874行)

**目标**: 从`DifferentiableAt`推导`deriv φ`的连续性

**完成步骤**:

1. **修改定理前提**（推荐）:

   ```lean
   -- 在定理前提中添加：
   (h_deriv_cont : ContinuousOn (deriv φ) (Set.Icc a b))
   ```

2. 或者查找API: `ContDiff.continuous_deriv`
3. 如果API存在，添加前提：

   ```lean
   (h_cont_diff : ContDiffOn ℝ 1 φ (Set.Icc a b))
   ```

**参考**: 已添加详细的证明步骤和替代方案

---

#### 4.2 分部积分法：deriv v的连续性

**位置**: `Lean/Exercises/Analysis/Real.lean` (约第915行)

**完成步骤**: 类似4.1，针对v

**参考**: 已添加详细的证明步骤和替代方案

---

#### 4.3 分部积分法：deriv u的连续性

**位置**: `Lean/Exercises/Analysis/Real.lean` (约第938行)

**完成步骤**: 类似4.1，针对u

**参考**: 已添加详细的证明步骤和替代方案

---

### 5. 其他（2个sorry）

#### 5.1 Parseval恒等式

**位置**: `Lean/Exercises/Topology/Basic.lean` (约第363行)

**目标**: 从`Basis`和`Orthonormal`构造`OrthonormalBasis`

**完成步骤**:

1. 查找API: `OrthonormalBasis.mk`, `OrthonormalBasis.ofBasis`, `OrthonormalBasis.mkOfOrthonormal`
2. 如果API存在，使用：

   ```lean
   -- 方法1：OrthonormalBasis.mk (v.repr) hv
   -- 方法2：OrthonormalBasis.ofBasis v hv
   -- 方法3：OrthonormalBasis.mkOfOrthonormal hv v.span_eq_top
   ```

3. 如果API不存在，查找: `Orthonormal.sum_inner_products_eq`（可能不需要构造OrthonormalBasis）
4. 或者手动证明：使用Basis的性质和Orthonormal的性质

**参考**: 已添加详细的证明步骤和替代方案

---

#### 5.2 逆函数定理（流形版本）

**位置**: `Lean/Exercises/Topology/Basic.lean` (约第414行)

**目标**: 使用mathlib4的逆函数定理（流形版本）

**完成步骤**:

1. 查找API: `MDifferentiableAt.localInverse`, `mfderiv_bijective_iff_localInverse`
2. 如果API存在，直接使用
3. 如果API不存在，查找: `HasStrictFDerivAt.localInverse`（Banach空间版本）
4. 或者使用局部坐标，将问题转化为Banach空间上的逆函数定理

**参考**: 已添加详细的证明步骤和替代方案

---

## 🔍 API查找指南

### 在mathlib4中查找API的方法

1. **使用Lean的`#check`命令**:

   ```lean
   #check Multiset.prod_map_mul
   #check OrthonormalBasis.mk
   ```

2. **使用Lean的`#find`命令**:

   ```lean
   #find Multiset.prod_map
   #find OrthonormalBasis
   ```

3. **在mathlib4文档中搜索**:
   - 访问: <https://leanprover-community.github.io/mathlib4_docs/>
   - 搜索相关的API名称

4. **在mathlib4源码中搜索**:
   - 访问: <https://github.com/leanprover-community/mathlib4>
   - 使用GitHub的搜索功能

---

## 📝 完成检查清单

完成每个sorry后，请检查：

- [ ] 代码编译通过（`lake build`）
- [ ] 没有类型错误
- [ ] 没有警告
- [ ] 证明逻辑清晰
- [ ] 注释完整
- [ ] 遵循mathlib4命名规范

---

## 🎯 完成优先级

**高优先级**（影响核心功能）:

1. 唯一分解定理（3个sorry）
2. 级数理论判别法（5个sorry）

**中优先级**（影响完整性）:
3. deriv连续性（3个sorry）
4. 粘接引理（4个sorry）

**低优先级**（高级功能）:
5. Parseval恒等式和逆函数定理（2个sorry）

---

## 📚 参考资源

1. **mathlib4文档**: <https://leanprover-community.github.io/mathlib4_docs/>
2. **Lean 4手册**: <https://leanprover.github.io/lean4/doc/>
3. **mathlib4源码**: <https://github.com/leanprover-community/mathlib4>
4. **Lean Zulip**: <https://leanprover.zulipchat.com/>

---

**最后更新**: 2025年10月1日
**状态**: 所有框架已完善，等待API查找和替代方案实施
