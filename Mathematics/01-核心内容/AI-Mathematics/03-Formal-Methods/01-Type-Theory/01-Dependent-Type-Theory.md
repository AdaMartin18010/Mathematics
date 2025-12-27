# 依值类型论 (Dependent Type Theory)

> **Dependent Type Theory (DTT)**  
> 形式化数学与程序验证的统一基础

---

## 📋 目录

- [依值类型论 (Dependent Type Theory)](#依值类型论-dependent-type-theory)
  - [📋 目录](#-目录)
  - [🎯 什么是依值类型论?](#-什么是依值类型论)
  - [📐 基础概念](#-基础概念)
    - [1. 简单类型论 vs 依值类型论](#1-简单类型论-vs-依值类型论)
    - [2. 类型的宇宙层次](#2-类型的宇宙层次)
    - [3. 函数类型与依值函数类型](#3-函数类型与依值函数类型)
  - [🔍 核心类型构造](#-核心类型构造)
    - [1. Π类型 (依值函数类型)](#1-π类型-依值函数类型)
    - [2. Σ类型 (依值对类型)](#2-σ类型-依值对类型)
    - [3. 归纳类型 (Inductive Types)](#3-归纳类型-inductive-types)
  - [🤖 Curry-Howard对应](#-curry-howard对应)
  - [💻 Lean 4实践](#-lean-4实践)
    - [基础类型定义](#基础类型定义)
    - [依值函数示例](#依值函数示例)
    - [归纳类型与递归](#归纳类型与递归)
  - [🔬 高级主题](#-高级主题)
    - [1. 同伦类型论 (HoTT)](#1-同伦类型论-hott)
    - [2. 归纳-递归类型](#2-归纳-递归类型)
    - [3. 高阶归纳类型 (HITs)](#3-高阶归纳类型-hits)
  - [🤖 在AI中的应用](#-在ai中的应用)
    - [1. 神经网络形状验证](#1-神经网络形状验证)
    - [2. 程序综合](#2-程序综合)
    - [3. 可解释AI](#3-可解释ai)
  - [📚 相关资源](#-相关资源)
    - [经典教材](#经典教材)
    - [重要论文](#重要论文)
    - [证明助手](#证明助手)
  - [🎓 对标课程](#-对标课程)
  - [💡 练习题](#-练习题)
    - [基础题](#基础题)
    - [进阶题](#进阶题)
    - [挑战题](#挑战题)

---

## 🎯 什么是依值类型论?

**依值类型论**是一种类型系统,其中**类型可以依赖于值**。这种强大的表达能力使得:

✅ **数学定理** = 类型  
✅ **证明** = 程序 (该类型的项)  
✅ **类型检查** = 证明验证  

**核心思想**:

$$
\text{类型} \quad A : \text{Type} \\
\text{项} \quad a : A \\
\text{依值类型} \quad B : A \to \text{Type}
$$

**示例**:

- `Vec α n`: 长度为`n`的向量类型 (类型依赖于自然数`n`)
- `Matrix α m n`: `m×n`矩阵类型 (类型依赖于两个自然数)
- `Sorted xs`: 证明列表`xs`已排序的类型

---

## 📐 基础概念

### 1. 简单类型论 vs 依值类型论

**简单类型论** (Simply Typed Lambda Calculus):

```text
类型:  A, B ::= Base | A → B
项:    t ::= x | λx:A. t | t₁ t₂
```

**限制**: 类型不能依赖于值

---

**依值类型论**:

```text
类型:  A, B ::= Type | (x : A) → B(x) | (x : A) × B(x) | ...
项:    t ::= x | λx:A. t | t₁ t₂ | (a, b) | ...
```

**关键**: `B(x)` 可以依赖于 `x : A`

---

### 2. 类型的宇宙层次

为避免Russell悖论,引入**类型宇宙层次**:

$$
\text{Type}_0 : \text{Type}_1 : \text{Type}_2 : \cdots
$$

- `Type₀`: 包含"小"类型 (如 `Nat`, `Bool`)
- `Type₁`: 包含 `Type₀` 和其上的函数
- 一般简写: `Type` = `Type₀`

**规则** (Universe Polymorphism):

```lean
-- Lean 4中的宇宙多态
def id {α : Type u} (x : α) : α := x

-- 可应用于任意宇宙层次
#check id (x : Nat)        -- Type 0
#check id (x : Type)       -- Type 1
```

---

### 3. 函数类型与依值函数类型

**普通函数类型**: $A \to B$

```lean
def f : Nat → Bool := fun n => n > 0
```

**依值函数类型** (Π-type): $(x : A) \to B(x)$

```lean
-- 返回类型依赖于输入值
def vector_head {α : Type} : (n : Nat) → Vec α (n + 1) → α
  | n, v => v.head
```

**关键区别**:

- 普通函数: 返回类型固定
- 依值函数: 返回类型可依赖输入的**值**

---

## 🔍 核心类型构造

### 1. Π类型 (依值函数类型)

**定义**:

$$
\frac{\Gamma, x : A \vdash B(x) : \text{Type}}{\Gamma \vdash \Pi_{(x:A)} B(x) : \text{Type}}
$$

**直观理解**:

- $\Pi_{(x:A)} B(x)$ 是"对所有 $x : A$, 产生 $B(x)$"的函数类型
- 等价于逻辑中的**全称量化**: $\forall x : A. B(x)$

**示例**:

```lean
-- 数学定理: ∀ n : Nat, n + 0 = n
def add_zero : (n : Nat) → n + 0 = n
  | 0 => rfl
  | n+1 => by simp [Nat.add_succ, add_zero n]
```

**类型**: `(n : Nat) → n + 0 = n` 就是一个Π类型

---

### 2. Σ类型 (依值对类型)

**定义**:

$$
\frac{\Gamma, x : A \vdash B(x) : \text{Type}}{\Gamma \vdash \Sigma_{(x:A)} B(x) : \text{Type}}
$$

**直观理解**:

- $\Sigma_{(x:A)} B(x)$ 是"存在 $x : A$ 使得 $B(x)$ 成立"的类型
- 等价于逻辑中的**存在量化**: $\exists x : A. B(x)$

**构造**:

```lean
-- 对的第二个分量类型依赖于第一个分量的值
structure Sigma {α : Type u} (β : α → Type v) where
  fst : α
  snd : β fst

-- 示例: 存在一个自然数使其平方为4
example : Σ (n : Nat), n^2 = 4 :=
  ⟨2, rfl⟩  -- 提供见证 n=2 和证明 2^2=4
```

---

### 3. 归纳类型 (Inductive Types)

**自然数**:

```lean
inductive Nat where
  | zero : Nat
  | succ : Nat → Nat
```

**依值向量**:

```lean
inductive Vec (α : Type u) : Nat → Type u where
  | nil : Vec α 0
  | cons : α → {n : Nat} → Vec α n → Vec α (n+1)
```

**关键**: `Vec α n` 的类型依赖于长度 `n`

**安全的头函数**:

```lean
def Vec.head {α : Type} {n : Nat} : Vec α (n+1) → α
  | cons a _ => a

-- 类型系统保证: 空向量无法调用head
-- Vec.head Vec.nil  -- 类型错误!
```

---

## 🤖 Curry-Howard对应

**核心思想**: 证明 = 程序

| 逻辑 | 类型论 | Lean语法 |
|------|--------|---------|
| 命题 $P$ | 类型 `P : Prop` | `theorem` |
| 证明 $p$ | 项 `p : P` | `proof` |
| $P \land Q$ | 积类型 `P × Q` | `And P Q` |
| $P \lor Q$ | 和类型 `P ⊕ Q` | `Or P Q` |
| $P \to Q$ | 函数类型 `P → Q` | `→` |
| $\forall x. P(x)$ | Π类型 `(x : A) → P x` | `∀` |
| $\exists x. P(x)$ | Σ类型 `(x : A) × P x` | `∃` |
| $\bot$ (假) | 空类型 `Empty` | `False` |
| $\top$ (真) | 单元类型 `Unit` | `True` |

**示例**:

```lean
-- 逻辑: ∀ P Q, P ∧ Q → Q ∧ P
theorem and_comm {P Q : Prop} : P ∧ Q → Q ∧ P :=
  fun ⟨hp, hq⟩ => ⟨hq, hp⟩

-- 类型论: 对任意类型A B, A × B → B × A
def prod_comm {A B : Type} : A × B → B × A :=
  fun (a, b) => (b, a)
```

---

## 💻 Lean 4实践

### 基础类型定义

```lean
-- 依值对 (Σ类型)
structure Sigma' {α : Type u} (β : α → Type v) where
  fst : α
  snd : β fst

-- 示例: 带长度的列表
def LengthList (α : Type) : Type :=
  Σ (n : Nat), Vec α n

-- 构造
def example_list : LengthList Nat :=
  ⟨3, Vec.cons 1 (Vec.cons 2 (Vec.cons 3 Vec.nil))⟩
```

---

### 依值函数示例

```lean
-- 安全的向量索引
def Vec.get {α : Type} {n : Nat} (v : Vec α n) (i : Fin n) : α :=
  match v, i with
  | cons a _, ⟨0, _⟩ => a
  | cons _ v', ⟨i'+1, h⟩ => Vec.get v' ⟨i', Nat.lt_of_succ_lt_succ h⟩

-- Fin n 是"小于n的自然数"类型
-- 类型系统保证索引不会越界!

example : Vec.get (Vec.cons 10 (Vec.cons 20 Vec.nil)) ⟨0, by norm_num⟩ = 10 := rfl
```

---

### 归纳类型与递归

```lean
-- 依值的二叉树
inductive Tree (α : Type) : Nat → Type where
  | leaf : α → Tree α 0
  | node : {n m : Nat} → Tree α n → Tree α m → Tree α (max n m + 1)

-- 计算深度(编码在类型中)
def Tree.depth {α : Type} : {n : Nat} → Tree α n → Nat
  | _, leaf _ => 0
  | _, node l r => max (depth l) (depth r) + 1

-- 定理: 实际深度等于类型中编码的深度
theorem Tree.depth_eq_type {α : Type} {n : Nat} (t : Tree α n) : 
  Tree.depth t = n := by
  induction t with
  | leaf _ => rfl
  | node l r ihl ihr => 
    simp [depth, ihl, ihr]
```

---

## 🔬 高级主题

### 1. 同伦类型论 (HoTT)

**核心思想**: 类型是空间,项是点,相等是路径

**同一性类型** (Identity Type):

$$
a =_A b \quad \text{(路径类型)}
$$

```lean
-- Lean 4中的相等类型
#check @Eq : {α : Type u} → α → α → Prop

-- 路径归纳 (J规则)
theorem path_induction {α : Type} {a : α}
  (C : (b : α) → a = b → Prop)
  (h : C a rfl) :
  ∀ {b : α} (p : a = b), C b p := by
  intro b p
  cases p
  exact h
```

**高阶路径**:

- `a = b` (路径)
- `p = q` (路径之间的路径, 即同伦)
- `α = β` (类型之间的同一性, 即univalence)

---

### 2. 归纳-递归类型

**同时定义类型和函数**:

```lean
-- 良类型的lambda项
inductive Term : Type
  | var : Nat → Term
  | app : Term → Term → Term
  | lam : Nat → Term → Term

-- 同时定义类型检查函数
def typecheck : Term → Option Type
  | Term.var n => some Nat  -- 简化示例
  | Term.app t1 t2 => 
      match typecheck t1, typecheck t2 with
      | some (Arrow A B), some A' => if A = A' then some B else none
      | _, _ => none
  | Term.lam x body => some (Arrow Nat (typecheck body).get!)
```

---

### 3. 高阶归纳类型 (HITs)

**带路径构造子的归纳类型**:

```lean
-- 圆 S¹
inductive Circle : Type where
  | base : Circle
  | loop : base = base  -- 路径构造子!

-- 整数是圆的基本群
theorem fundamental_group_circle : 
  (base = base) ≃ Int := sorry
```

---

## 🤖 在AI中的应用

### 1. 神经网络形状验证

```lean
-- 保证形状正确的矩阵乘法
def matmul {m n p : Nat} 
  (A : Matrix Float m n) 
  (B : Matrix Float n p) : 
  Matrix Float m p :=
  -- 类型系统保证 n 匹配!
  sorry

-- 神经网络层
structure Layer (input output : Nat) where
  weights : Matrix Float output input
  bias : Vec Float output
  
def Layer.forward {n m : Nat} (layer : Layer n m) (x : Vec Float n) : 
  Vec Float m :=
  -- 类型保证形状兼容
  sorry

-- 组合网络
def compose_layers {a b c : Nat} 
  (l1 : Layer a b) 
  (l2 : Layer b c) : 
  Layer a c :=
  -- 中间维度 b 自动对齐
  sorry
```

**好处**:

- **编译时检查**: 形状不匹配无法编译
- **零运行时开销**: 类型擦除后无额外代价
- **文档作用**: 类型即规格

---

### 2. 程序综合

**从类型自动生成程序**:

```lean
-- 给定类型,自动寻找实现
def synthesize (T : Type) : Option T := 
  -- 搜索该类型的项
  sorry

-- 示例: 自动证明简单定理
example : ∀ (P Q : Prop), P → (P → Q) → Q :=
  fun P Q hp hpq => hpq hp
  -- 或使用 tactic: by intro P Q hp hpq; exact hpq hp
```

---

### 3. 可解释AI

**形式化决策规则**:

```lean
-- 决策树类型
inductive DecisionTree (Feature Label : Type) : Type where
  | leaf : Label → DecisionTree Feature Label
  | node : (Feature → Bool) → 
           DecisionTree Feature Label →  -- true分支
           DecisionTree Feature Label →  -- false分支
           DecisionTree Feature Label

-- 可证明的预测
def predict_with_proof 
  {Feature Label : Type}
  (tree : DecisionTree Feature Label)
  (input : Feature) :
  Σ (output : Label), TreePath tree input output :=
  -- 返回预测结果和决策路径的证明
  sorry
```

---

## 📚 相关资源

### 经典教材

1. **Type Theory and Formal Proof**  
   Rob Nederpelt, Herman Geuvers (2014)  
   → 系统的类型论教材

2. **Homotopy Type Theory: Univalent Foundations**  
   The Univalent Foundations Program (2013)  
   → HoTT圣经

3. **Programming in Martin-Löf's Type Theory**  
   Bengt Nordström et al. (1990)  
   → MLTT经典

---

### 重要论文

1. **Martin-Löf, P. (1975)**  
   "An Intuitionistic Theory of Types"  
   → 依值类型论的奠基

2. **Coquand, T. & Huet, G. (1988)**  
   "The Calculus of Constructions"  
   → CoC, Coq的理论基础

3. **Voevodsky, V. (2013)**  
   "Univalent Foundations"  
   → Univalence公理

---

### 证明助手

- **Lean 4**: 现代化, 高性能, 活跃社区
- **Coq**: 成熟稳定, 大型项目经验
- **Agda**: 研究导向, 类型系统最丰富
- **Idris 2**: 面向程序验证

---

## 🎓 对标课程

| 大学 | 课程 | 内容 |
|------|------|------|
| CMU | 15-815 Type Systems | DTT, System F, CoC |
| Stanford | CS 359 Automated Deduction | 类型论与证明助手 |
| Cambridge | Part III Logic & Proof | Martin-Löf类型论 |
| ETH Zurich | Program Verification | Dependent Types in Lean |

---

## 💡 练习题

### 基础题

**1. 实现安全的列表索引**:

```lean
-- 实现这个函数,确保索引不会越界
def List.get_safe {α : Type} (xs : List α) (i : Fin xs.length) : α :=
  sorry

-- 测试
example : List.get_safe [1, 2, 3] ⟨1, by norm_num⟩ = 2 := sorry
```

---

**2. 证明向量拼接的长度**:

```lean
def Vec.append {α : Type} {m n : Nat} : 
  Vec α m → Vec α n → Vec α (m + n) :=
  sorry

-- 证明长度正确
theorem append_length {α : Type} {m n : Nat} 
  (v1 : Vec α m) (v2 : Vec α n) :
  (Vec.append v1 v2).length = m + n := by
  sorry
```

---

### 进阶题

**3. 实现类型安全的矩阵运算**:

```lean
structure Matrix (α : Type) (m n : Nat) where
  data : Vec (Vec α n) m

def Matrix.transpose {α : Type} {m n : Nat} : 
  Matrix α m n → Matrix α n m :=
  sorry

-- 证明转置两次是恒等
theorem transpose_transpose {α : Type} {m n : Nat} (A : Matrix α m n) :
  A.transpose.transpose = A := by
  sorry
```

---

### 挑战题

**4. 实现红黑树的形式化验证**:

```lean
-- 红黑树的不变式:
-- 1. 根节点是黑色
-- 2. 红色节点的子节点必须是黑色
-- 3. 从根到叶子的所有路径包含相同数量的黑色节点

inductive Color where
  | Red | Black

inductive RBTree (α : Type) : Nat → Type where  -- Nat编码黑高度
  | leaf : RBTree α 0
  | redNode : {bh : Nat} → 
      RBTree α bh → α → RBTree α bh → 
      RBTree α bh
  | blackNode : {bh : Nat} →
      RBTree α bh → α → RBTree α bh →
      RBTree α (bh + 1)

-- 实现插入操作,保持不变式
def RBTree.insert {α : Type} [Ord α] {bh : Nat} :
  RBTree α bh → α → Σ (bh' : Nat), RBTree α bh' :=
  sorry
```

---

**📌 下一主题**: [Lean 4证明助手](./02-Lean4-Proof-Assistant.md)

**🔙 返回**: [类型论](../README.md) | [形式化方法](../../README.md)
