# Phase 3 - P0问题修复详细指南

**创建日期**: 2025-12-21
**版本**: v1.0
**状态**: 进行中
**优先级**: P0 - 紧急

---

## 📋 概述

本指南提供P0问题（14个sorry占位符）的详细修复方法。每个问题都包含：

- 问题描述
- 代码位置
- 修复方法
- 参考资源
- 验证方法

---

## 🔧 问题1-3: 导数连续性证明（3处）

### 问题描述

从`DifferentiableAt`推导`deriv`的连续性需要额外条件。`DifferentiableAt`只保证导数存在，不保证导数连续。

### 代码位置

- `Exercises/Analysis/Real.lean` 第890行
- `Exercises/Analysis/Real.lean` 第939行
- `Exercises/Analysis/Real.lean` 第971行

### 修复方法

#### 方法1: 添加前提条件（推荐）

修改定理前提，添加：

```lean
(h_deriv_cont : ContinuousOn (deriv φ) (Set.Icc a b))
```

然后直接使用这个前提条件。

#### 方法2: 使用ContDiff

如果函数是`ContDiff`（连续可微），则导数自动连续：

```lean
(h_cont_diff : ContDiffOn ℝ 1 φ (Set.Icc a b))
```

然后使用：

```lean
have h_deriv_cont : ContinuousOn (deriv φ) (Set.Icc a b) :=
  ContDiffOn.continuousOn_deriv h_cont_diff
```

### 参考资源

- Mathlib4文档: `ContDiffOn.continuousOn_deriv`
- Mathlib4文档: `DifferentiableAt.continuousAt`（只保证函数连续，不保证导数连续）

### 验证方法

编译验证：

```bash
cd 01-核心内容/Lean/Exercises
lake build Analysis/Real.lean
```

---

## 🔧 问题4-7: 级数判别法证明（4处）

### 问题描述

需要liminf/limsup的性质和几何级数比较判别法来证明级数收敛性。

### 代码位置

- `Exercises/Analysis/Real.lean` 第1171行（比值判别法，ρ < 1情况）
- `Exercises/Analysis/Real.lean` 第1235行（比值判别法，ρ > 1情况）
- `Exercises/Analysis/Real.lean` 第1285行（根式判别法，ρ < 1情况）
- `Exercises/Analysis/Real.lean` 第1335行（根式判别法，ρ > 1情况）

### 修复方法

#### 比值判别法（第1171行）

```lean
-- 如果liminf < 1，则存在r < 1和N使得对所有n ≥ N，a(n+1)/a(n) < r
-- 使用liminf的性质
have h_eventually : ∃ᶠ n in Filter.atTop, a (n + 1) / a n < r := by
  -- 使用liminf_lt_iff_eventually_lt或类似API
  sorry -- 需要查找正确的API

-- 通过归纳证明a(n) < a(N) * r^(n-N)
-- 使用几何级数比较判别法
```

#### 根式判别法（第1285行）

```lean
-- 如果limsup < 1，则存在r < 1和N使得对所有n ≥ N，a(n)^(1/n) < r
-- 使用limsup的性质
have h_eventually : ∃ᶠ n in Filter.atTop, (a n) ^ (1 / n : ℝ) < r := by
  -- 使用limsup_lt_iff_eventually_lt或类似API
  sorry -- 需要查找正确的API

-- 因此a(n) < r^n，使用几何级数比较判别法
```

### 参考资源

- Mathlib4文档: `Filter.liminf`, `Filter.limsup`
- Mathlib4文档: `Filter.eventually_atTop`
- Mathlib4文档: 几何级数收敛定理
- Mathlib4文档: 比较判别法API

### 验证方法

编译验证并运行测试用例。

---

## 🔧 问题8: 幂级数连续性证明（1处）

### 问题描述

需要证明幂级数在收敛半径内连续，这需要一致收敛性和连续性的定理。

### 代码位置

- `Exercises/Analysis/Real.lean` 第1629行

### 修复方法

#### 方法1: 使用Weierstrass M-判别法

```lean
-- 1. 在收敛半径内的任意紧致集上，幂级数一致收敛
-- 2. 一致收敛的连续函数序列的极限函数连续
-- 3. 因此幂级数在收敛半径内连续

-- 使用Weierstrass M-判别法证明一致收敛
have h_uniform_conv : UniformConvergesOn ... := by
  -- 需要构造M_n使得|a_n * x^n| ≤ M_n且∑M_n收敛
  sorry -- 需要查找Weierstrass M-判别法API

-- 使用一致收敛的连续函数序列的极限函数连续
have h_cont : ContinuousAt f x := by
  -- 使用UniformConvergesOn.continuous或类似API
  sorry -- 需要查找连续性API
```

#### 方法2: 使用幂级数的连续性定理

如果mathlib4有幂级数的连续性定理，可以直接使用：

```lean
exact PowerSeries.continuousOn_ball h_radius x hx
```

### 参考资源

- Mathlib4文档: `UniformConvergesOn`
- Mathlib4文档: `UniformConvergesOn.continuous`
- Mathlib4文档: `PowerSeries.continuousOn_ball`（如果存在）
- Mathlib4文档: Weierstrass M-判别法

### 验证方法

编译验证并测试幂级数的连续性。

---

## 🔧 问题9-14: 拓扑学证明（6处）

### 问题描述

需要连续性定义和粘接引理的完整证明，以及标准正交基的构造和逆函数定理的应用。

### 代码位置

- `Exercises/Topology/Basic.lean` 第234-295行（粘接引理）
- `Exercises/Topology/Basic.lean` 第390行（标准正交基构造）
- `Exercises/Topology/Basic.lean` 第456行（逆函数定理）

### 修复方法

#### 粘接引理（第234-295行）

```lean
-- 如果f在A上连续，g在B上连续，且f|A∩B = g|A∩B
-- 则h = f on A, h = g on B在A∪B上连续

-- 使用mathlib4的连续性API
have h_cont_on_A : ContinuousOn h A := by
  -- 使用ContinuousOn.congr或类似API
  exact ContinuousOn.congr hf (fun x hx => h_h_on_A x hx)

have h_cont_on_B : ContinuousOn h B := by
  -- 类似地处理B
  exact ContinuousOn.congr hg (fun x hx => h_h_on_B x hx)

-- 使用粘接引理
have h_cont : ContinuousOn h (A ∪ B) := by
  -- 使用ContinuousOn.union或类似API
  exact ContinuousOn.union h_cont_on_A h_cont_on_B
```

#### 标准正交基构造（第390行）

```lean
-- 从Basis和Orthonormal构造OrthonormalBasis
-- 使用OrthonormalBasis.mk或类似方法

have h_orthonormal_basis : OrthonormalBasis ι ℝ V := by
  -- 使用OrthonormalBasis.mk
  exact OrthonormalBasis.mk h_basis h_orthonormal
```

#### 逆函数定理（第456行）

```lean
-- 使用mathlib4的逆函数定理（流形版本）
-- 需要查找正确的API名称

have h_inv : ... := by
  -- 使用逆函数定理
  exact inverseFunctionTheorem h_f h_df h_invertible
```

### 参考资源

- Mathlib4文档: `ContinuousOn.union`
- Mathlib4文档: `OrthonormalBasis.mk`
- Mathlib4文档: 逆函数定理API
- Mathlib4文档: 流形上的逆函数定理

### 验证方法

编译验证并运行测试用例。

---

## 📚 通用修复流程

### 步骤1: 理解问题

1. 阅读代码上下文
2. 理解数学定理
3. 查找mathlib4相关API

### 步骤2: 查找API

1. 搜索mathlib4文档
2. 查找相关定理
3. 理解API使用方式

### 步骤3: 实现修复

1. 编写修复代码
2. 添加必要的前提条件
3. 使用正确的API

### 步骤4: 验证

1. 编译验证
2. 运行测试用例
3. 检查证明完整性

---

## 🔗 参考资源

### Mathlib4文档

- [Analysis文档](https://leanprover-community.github.io/mathlib4_docs/)
- [Topology文档](https://leanprover-community.github.io/mathlib4_docs/)
- [Filter文档](https://leanprover-community.github.io/mathlib4_docs/)

### 学习资源

- [Mathlib4教程](https://leanprover-community.github.io/learn.html)
- [Lean 4手册](https://leanprover.github.io/lean4/doc/)
- [定理证明社区](https://leanprover-community.github.io/)

---

**最后更新**: 2025-12-21
**状态**: 进行中
**下一步**: 开始修复第一个sorry占位符
