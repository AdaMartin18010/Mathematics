# Lean快速上手指南 | Lean Quick Start Guide

## 📋 目录 | Table of Contents

- [Lean快速上手指南 | Lean Quick Start Guide](#lean快速上手指南--lean-quick-start-guide)
  - [📋 目录 | Table of Contents](#-目录--table-of-contents)
  - [🎯 学习目标 | Learning Objectives](#-学习目标--learning-objectives)
  - [🛠️ 环境搭建 | Environment Setup](#️-环境搭建--environment-setup)
    - [必需工具](#必需工具)
    - [安装步骤](#安装步骤)
    - [验证安装](#验证安装)
  - [📚 核心概念 | Core Concepts](#-核心概念--core-concepts)
    - [1. 类型系统 | Type System](#1-类型系统--type-system)
    - [2. 定义与声明 | Definitions and Declarations](#2-定义与声明--definitions-and-declarations)
    - [3. 命名空间 | Namespaces](#3-命名空间--namespaces)
  - [🔧 基本语法 | Basic Syntax](#-基本语法--basic-syntax)
    - [函数定义语法](#函数定义语法)
    - [定理证明语法](#定理证明语法)
    - [导入和模块](#导入和模块)
  - [🎯 核心练习 | Core Exercises](#-核心练习--core-exercises)
    - [练习1：命题逻辑证明](#练习1命题逻辑证明)
    - [练习2：递归定义与证明](#练习2递归定义与证明)
    - [练习3：归纳证明](#练习3归纳证明)
  - [🚀 进阶技巧 | Advanced Techniques](#-进阶技巧--advanced-techniques)
    - [1. 证明策略组合](#1-证明策略组合)
    - [2. 使用计算器](#2-使用计算器)
    - [3. 使用重写](#3-使用重写)
  - [📖 常用策略 | Common Tactics](#-常用策略--common-tactics)
  - [🔍 调试技巧 | Debugging Tips](#-调试技巧--debugging-tips)
    - [1. 使用 `#check` 检查类型](#1-使用-check-检查类型)
    - [2. 使用 `#eval` 计算值](#2-使用-eval-计算值)
    - [3. 使用 `#print` 查看定义](#3-使用-print-查看定义)
    - [4. 使用 `#reduce` 归约表达式](#4-使用-reduce-归约表达式)
  - [📚 学习资源 | Learning Resources](#-学习资源--learning-resources)
    - [官方文档](#官方文档)
    - [社区资源](#社区资源)
    - [练习项目](#练习项目)
  - [🎉 下一步 | Next Steps](#-下一步--next-steps)

## 🎯 学习目标 | Learning Objectives

- 了解Lean基本语法与交互方式
- 掌握核心概念：类型、函数、定理、证明
- 完成三个核心练习：命题证明、递归定义、归纳证明
- 建立Lean开发环境

---

## 🛠️ 环境搭建 | Environment Setup

### 必需工具

- **Lean 4** 与 `elan`: 工具链与版本管理
- **Lake**: Lean 4 项目与依赖管理
- **VS Code** 与 Lean 4 扩展

### 安装步骤

1. 安装 `elan`（将自动安装 Lean 4 工具链）：`Windows` 建议按官方指南执行：`https://leanprover-community.github.io/get_started.html`
2. 安装 VS Code: `https://code.visualstudio.com/`
3. 安装 Lean 4 扩展（Marketplace 搜索 "Lean 4"）
4. 初始化 `lake`（随 Lean 4 提供），确保可用：`lake --version`

### 验证安装

```bash
# 检查Lean版本
lean --version

# 创建新项目
lake new my_project
cd my_project
lake build

# 查看工具链
type lean-toolchain | cat
```

---

## 📚 核心概念 | Core Concepts

### 1. 类型系统 | Type System

```lean
-- 基本类型
#check Nat        -- 自然数类型
#check Bool       -- 布尔类型
#check String     -- 字符串类型
#check List Nat   -- 自然数列表类型

-- 函数类型
#check Nat → Nat  -- 自然数到自然数的函数类型
#check Nat → Bool → Nat  -- 多参数函数类型
```

### 2. 定义与声明 | Definitions and Declarations

```lean
-- 函数定义
def add (a b : Nat) : Nat := a + b

-- 定理声明
theorem add_comm (a b : Nat) : a + b = b + a := by
  -- 证明内容
  admit

-- 引理声明
lemma add_zero (n : Nat) : n + 0 = n := by
  -- 证明内容
  admit
```

### 3. 命名空间 | Namespaces

```lean
namespace MyMath

def double (n : Nat) : Nat := 2 * n

theorem double_zero : double 0 = 0 := by
  simp[double]

end MyMath
```

---

## 🔧 基本语法 | Basic Syntax

### 函数定义语法

```lean
-- 基本函数
def square (x : Nat) : Nat := x * x

-- 带类型注解的函数
def max (a b : Nat) : Nat := if a > b then a else b

-- 递归函数
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n
```

### 定理证明语法

```lean
-- 简单定理
theorem example1 (P Q : Prop) : P → Q → P := by
  intro hP  -- 引入假设P
  intro hQ  -- 引入假设Q
  exact hP  -- 使用假设P完成证明

-- 使用策略的定理
theorem example2 (n : Nat) : n + 0 = n := by
  simp  -- 使用简化策略
```

### 导入和模块

```lean
-- 导入标准库
import Std
import Mathlib

-- 打开命名空间
open Nat
open List
open scoped BigOperators
```

---

## 🎯 核心练习 | Core Exercises

### 练习1：命题逻辑证明

**目标**: 证明 `(True ∧ P) → P`

```lean
theorem id_left (P : Prop) : (True ∧ P) → P := by
  intro h
  cases h with
  | intro hTrue hP => exact hP
```

**提示**: 使用 `intro` 和 `cases` 策略

### 练习2：递归定义与证明

**目标**: 定义 `pow2 : Nat → Nat` 使得 `pow2 n = 2^n`，并证明 `pow2 0 = 1`

```lean
def pow2 : Nat → Nat
  | 0 => 1
  | n + 1 => 2 * pow2 n

theorem pow2_zero : pow2 0 = 1 := by
  simp[pow2]
```

**提示**: 使用递归定义和 `simp` 策略

### 练习3：归纳证明

**目标**: 证明自然数加法结合律 `(a + b) + c = a + (b + c)`

```lean
theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by
  induction a with
  | zero => simp
  | succ a ih => simp[ih]
```

**提示**: 使用 `induction` 和 `simp` 策略

---

## 🚀 进阶技巧 | Advanced Techniques

### 1. 证明策略组合

```lean
theorem complex_example (n : Nat) : n + n = 2 * n := by
  induction n with
  | zero => simp
  | succ n ih => 
    simp[ih]
    ring
```

### 2. 使用计算器

```lean
theorem calc_example (a b c : Nat) : (a + b) + c = a + (b + c) := by
  calc
    (a + b) + c = a + b + c := by simp
    _ = a + (b + c) := by simp
```

### 3. 使用重写

```lean
theorem rewrite_example (a b : Nat) (h : a = b) : a + 1 = b + 1 := by
  rw [h]
```

---

## 📖 常用策略 | Common Tactics

| 策略 | 用途 | 示例 |
| ---- |------| ---- |
| `intro` | 引入假设 | `intro h` |
| `exact` | 使用假设 | `exact h` |
| `simp` | 简化表达式 | `simp[def_name]` |
| `rw` | 重写表达式 | `rw [h]` |
| `induction` | 归纳证明 | `induction n with` |
| `cases` | 情况分析 | `cases h with` |
| `ring` | 环运算 | `ring` |
| `linarith` | 线性算术 | `linarith` |

---

## 🔍 调试技巧 | Debugging Tips

### 1. 使用 `#check` 检查类型

```lean
#check Nat
#check List Nat
#check Nat → Nat
```

### 2. 使用 `#eval` 计算值

```lean
#eval 2 + 3
#eval factorial 5
```

### 3. 使用 `#print` 查看定义

```lean
#print Nat.add
#print factorial
```

### 4. 使用 `#reduce` 归约表达式

```lean
#reduce factorial 3
#reduce 2 + 3
```

---

## 📚 学习资源 | Learning Resources

### 官方文档

- [Lean 教程](https://leanprover-community.github.io/learn/)
- [Mathlib4 文档](https://leanprover-community.github.io/mathlib4_docs/)
- [Lean 4 参考](https://leanprover-community.github.io/lean4/doc/)

### 社区资源

- [Lean社区论坛](https://leanprover.zulipchat.com/)
- [GitHub讨论](https://github.com/leanprover-community/lean/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/lean)

### 练习项目

- [Lean 4练习](https://leanprover-community.github.io/lean4/doc/exercises.html)
- [数学库贡献](https://leanprover-community.github.io/contribute/)
- [形式化数学项目](https://leanprover-community.github.io/mathlib_docs/project.html)

---

## 🎉 下一步 | Next Steps

完成本指南后，建议：

1. **深入学习**: 阅读官方教程和文档
2. **实践项目**: 尝试形式化简单的数学定理
3. **参与社区**: 加入Lean社区讨论
4. **贡献代码**: 为数学库贡献代码
5. **探索应用**: 尝试在其他领域的应用

---

*恭喜！您已经掌握了Lean的基础知识，可以开始您的形式化数学之旅了！*
