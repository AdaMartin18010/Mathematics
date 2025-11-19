# Lean代码完善快速参考

**创建日期**: 2025年10月1日
**用途**: 快速查找剩余sorry的位置和关键信息

---

## 📍 剩余sorry位置速查表

| # | 模块 | 文件 | 行号 | 关键API | 优先级 |
|---|------|------|------|---------|--------|
| 1 | 唯一分解 | `Algebra/Polynomial.lean` | ~153 | `Multiset.prod_map_mul` | 高 |
| 2 | 唯一分解 | `Algebra/Polynomial.lean` | ~179 | `Multiset.prod_hom` | 高 |
| 3 | 唯一分解 | `Algebra/Polynomial.lean` | ~275 | `Polynomial.isUnit_iff_C` | 高 |
| 4 | 粘接引理 | `Topology/Basic.lean` | ~210 | `ContinuousOn.restrict` | 中 |
| 5 | 粘接引理 | `Topology/Basic.lean` | ~241 | `ContinuousOn.restrict` | 中 |
| 6 | 粘接引理 | `Topology/Basic.lean` | ~257 | `ContinuousOn.union` | 中 |
| 7 | 粘接引理 | `Topology/Basic.lean` | ~285 | `ContinuousOn.univ_iff` | 中 |
| 8 | 级数理论 | `Analysis/Real.lean` | ~1130 | `liminf_lt_iff_eventually_lt` | 高 |
| 9 | 级数理论 | `Analysis/Real.lean` | ~1190 | `liminf_gt_iff_frequently_gt` | 高 |
| 10 | 级数理论 | `Analysis/Real.lean` | ~1234 | `limsup_lt_iff_eventually_lt` | 高 |
| 11 | 级数理论 | `Analysis/Real.lean` | ~1290 | `limsup_gt_iff_frequently_gt` | 高 |
| 12 | 级数理论 | `Analysis/Real.lean` | ~1546 | `PowerSeries.continuousOn_ball` | 高 |
| 13 | deriv连续性 | `Analysis/Real.lean` | ~874 | `ContDiff.continuous_deriv` | 中 |
| 14 | deriv连续性 | `Analysis/Real.lean` | ~915 | `ContDiff.continuous_deriv` | 中 |
| 15 | deriv连续性 | `Analysis/Real.lean` | ~938 | `ContDiff.continuous_deriv` | 中 |
| 16 | Parseval | `Topology/Basic.lean` | ~363 | `OrthonormalBasis.mk` | 低 |
| 17 | 逆函数定理 | `Topology/Basic.lean` | ~414 | `MDifferentiableAt.localInverse` | 低 |

---

## 🔑 关键API速查

### Multiset相关

- `Multiset.prod_map_mul` - 乘积映射的分配律
- `Multiset.prod_map_mul'` - 乘积映射的分配律（变体）
- `Multiset.prod_map_prod` - 乘积映射的乘积
- `Multiset.prod_hom` - 同态映射的乘积

### Polynomial相关

- `Polynomial.isUnit_iff` - 单位多项式的特征
- `Polynomial.isUnit_iff_C` - 单位多项式是常数
- `Polynomial.eq_C_of_degree_eq_zero` - 零次多项式等于常数
- `Polynomial.degree_eq_zero_of_isUnit` - 单位多项式的次数为零
- `Polynomial.C_mul` - 常数多项式的乘法
- `Polynomial.C_prod` - 常数多项式的乘积
- `Polynomial.map_prod` - 映射的乘积

### 连续性相关

- `ContinuousOn.restrict` - 限制的连续性
- `ContinuousOn.codRestrict` - 值域限制的连续性
- `Continuous.restrict` - 连续函数的限制
- `ContinuousOn.union` - 并集的连续性
- `ContinuousOn.union'` - 并集的连续性（变体）
- `ContinuousOn.union_closed` - 闭集并集的连续性
- `ContinuousOn.univ_iff` - 全空间的连续性等价
- `ContinuousOn.continuous` - 从ContinuousOn推导Continuous

### 级数理论相关

- `liminf_lt_iff_eventually_lt` - liminf小于的等价条件
- `liminf_gt_iff_frequently_gt` - liminf大于的等价条件
- `limsup_lt_iff_eventually_lt` - limsup小于的等价条件
- `limsup_gt_iff_frequently_gt` - limsup大于的等价条件
- `Filter.eventually_atTop` - 最终成立
- `Filter.frequently_atTop` - 频繁成立
- `HasSum.geometric_series` - 几何级数的和
- `PowerSeries.continuousOn_ball` - 幂级数在球上的连续性
- `UniformContinuous.continuous` - 一致连续蕴含连续
- `UniformConvergence.continuous` - 一致收敛的连续性

### 微分相关

- `ContDiff.continuous_deriv` - 连续可微的导数连续
- `ContDiffOn` - 在集合上连续可微

### 内积空间相关

- `OrthonormalBasis.mk` - 构造标准正交基
- `OrthonormalBasis.ofBasis` - 从基构造标准正交基
- `OrthonormalBasis.mkOfOrthonormal` - 从标准正交集构造
- `Orthonormal.sum_inner_products_eq` - Parseval恒等式

### 流形相关

- `MDifferentiableAt.localInverse` - 流形上的局部逆
- `mfderiv_bijective_iff_localInverse` - 流形导数的双射与局部逆
- `HasStrictFDerivAt.localInverse` - Banach空间上的局部逆

---

## 🛠️ 常用命令

### 检查API是否存在

```lean
#check Multiset.prod_map_mul
#check OrthonormalBasis.mk
```

### 查找相关API

```lean
#find Multiset.prod_map
#find OrthonormalBasis
```

### 编译检查

```bash
lake build
```

### 类型检查

```bash
lake env lean --check Lean/Exercises/Analysis/Real.lean
```

---

## 📝 完成流程

1. **查找API** → 使用`#check`或`#find`命令
2. **如果API存在** → 直接使用
3. **如果API不存在** → 查看完成指南中的替代方案
4. **实施替代方案** → 按照详细步骤进行
5. **验证** → 运行`lake build`检查
6. **更新** → 在完成指南中标记完成

---

## 🎯 优先级说明

- **高优先级**: 影响核心功能，建议优先完成
- **中优先级**: 影响完整性，建议其次完成
- **低优先级**: 高级功能，可以最后完成

---

## 📚 快速链接

- **详细指南**: `Lean代码完善完成指南.md`
- **进度报告**: `Lean代码完善进度报告.md`
- **mathlib4文档**: <https://leanprover-community.github.io/mathlib4_docs/>
- **mathlib4源码**: <https://github.com/leanprover-community/mathlib4>

---

**最后更新**: 2025年10月1日
**状态**: 快速参考卡片已创建
