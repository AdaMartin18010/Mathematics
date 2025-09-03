# Lean语法与类型系统 | Lean Syntax and Type System

## 📋 目录 | Table of Contents

- [Lean语法与类型系统 | Lean Syntax and Type System](#lean语法与类型系统--lean-syntax-and-type-system)
  - [📋 目录 | Table of Contents](#-目录--table-of-contents)
  - [🔤 基本语法 | Basic Syntax](#-基本语法--basic-syntax)
    - [注释语法](#注释语法)
    - [标识符规则](#标识符规则)
    - [命名约定](#命名约定)
  - [🏷️ 类型系统 | Type System](#️-类型系统--type-system)
    - [基本类型](#基本类型)
    - [函数类型](#函数类型)
    - [依赖类型](#依赖类型)
  - [🔧 函数定义 | Function Definitions](#-函数定义--function-definitions)
    - [基本函数定义](#基本函数定义)
    - [递归函数](#递归函数)
    - [高阶函数](#高阶函数)
  - [📊 数据结构 | Data Structures](#-数据结构--data-structures)
    - [列表 (List)](#列表-list)
    - [对 (Pair)](#对-pair)
    - [选项 (Option)](#选项-option)
  - [🎭 模式匹配 | Pattern Matching](#-模式匹配--pattern-matching)
    - [基本模式匹配](#基本模式匹配)
    - [守卫模式 (Guards)](#守卫模式-guards)
    - [模式匹配技巧](#模式匹配技巧)
  - [🏗️ 类型类 | Type Classes](#️-类型类--type-classes)
    - [基本类型类](#基本类型类)
    - [类型类实例](#类型类实例)
    - [类型类约束](#类型类约束)
  - [🔍 类型推导 | Type Inference](#-类型推导--type-inference)
    - [自动类型推导](#自动类型推导)
    - [类型注解](#类型注解)
  - [📝 总结 | Summary](#-总结--summary)
  - [🔗 相关资源 | Related Resources](#-相关资源--related-resources)

---

## 🔤 基本语法 | Basic Syntax

### 注释语法

```lean
-- 单行注释
/- 多行注释 -/
/- 嵌套注释 /- 内部注释 -/ -/
```

### 标识符规则

```lean
-- 有效标识符
def myFunction := 42
def MyType := Type
def _private := "private"
def `special-name` := "special"

-- 无效标识符（不能以数字开头）
-- def 1function := 42  -- 错误！
```

### 命名约定

```lean
-- 函数和变量：小写字母，下划线分隔
def add_numbers (a b : Nat) := a + b

-- 类型：首字母大写
def MyDataType := Type

-- 常量：全大写
def MAX_SIZE := 1000
```

---

## 🏷️ 类型系统 | Type System

### 基本类型

```lean
-- 自然数
#check Nat
#check 0 : Nat
#check 42 : Nat

-- 整数
#check Int
#check (-5) : Int

-- 有理数
#check Rat
#check (1/2) : Rat

-- 实数
#check Real
#check π : Real

-- 布尔值
#check Bool
#check true : Bool
#check false : Bool

-- 字符串
#check String
#check "Hello" : String

-- 单位类型
#check Unit
#check () : Unit

-- 空类型
#check Empty
-- 没有值可以构造
```

### 函数类型

```lean
-- 基本函数类型
#check Nat → Nat           -- 自然数到自然数
#check Nat → Bool          -- 自然数到布尔值
#check Nat → Nat → Nat     -- 两个自然数到自然数

-- 柯里化函数
def add : Nat → Nat → Nat := fun a b => a + b
def add_curried : Nat → (Nat → Nat) := fun a => fun b => a + b

-- 等价性
#check add = add_curried  -- 类型相同
```

### 依赖类型

```lean
-- 依赖函数类型
def Vector (α : Type) (n : Nat) := List α

-- 依赖对类型
def Sigma {α : Type} (β : α → Type) := {a : α} × β a

-- 依赖函数
def Vector.map {α β : Type} {n : Nat} (f : α → β) (v : Vector α n) : Vector β n :=
  match v with
  | [] => []
  | h :: t => f h :: Vector.map f t
```

---

## 🔧 函数定义 | Function Definitions

### 基本函数定义

```lean
-- 显式类型注解
def square (x : Nat) : Nat := x * x

-- 类型推导
def double x := x + x

-- 多参数函数
def add_three (a b c : Nat) : Nat := a + b + c

-- 带默认值的参数
def greet (name : String) (greeting : String := "Hello") : String :=
  greeting ++ ", " ++ name ++ "!"

#eval greet "Alice"           -- "Hello, Alice!"
#eval greet "Bob" "Hi"        -- "Hi, Bob!"
```

### 递归函数

```lean
-- 简单递归
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- 相互递归
mutual
  def even : Nat → Bool
    | 0 => true
    | n + 1 => odd n
  
  def odd : Nat → Bool
    | 0 => false
    | n + 1 => even n
end

-- 结构递归
def length {α : Type} : List α → Nat
  | [] => 0
  | _ :: xs => 1 + length xs
```

### 高阶函数

```lean
-- 函数作为参数
def apply_twice {α : Type} (f : α → α) (x : α) : α := f (f x)

-- 函数作为返回值
def const {α β : Type} (x : α) : β → α := fun _ => x

-- 函数组合
def compose {α β γ : Type} (f : β → γ) (g : α → β) : α → γ :=
  fun x => f (g x)

-- 部分应用
def add_five := add 5
#eval add_five 3  -- 8
```

---

## 📊 数据结构 | Data Structures

### 列表 (List)

```lean
-- 列表类型
#check List Nat
#check [1, 2, 3] : List Nat

-- 列表操作
def list_example : List Nat :=
  let empty : List Nat := []
  let single := [42]
  let multiple := 1 :: 2 :: 3 :: []
  let concatenated := [1, 2] ++ [3, 4]
  concatenated

-- 列表函数
def sum_list : List Nat → Nat
  | [] => 0
  | h :: t => h + sum_list t

def map_list {α β : Type} (f : α → β) : List α → List β
  | [] => []
  | h :: t => f h :: map_list f t
```

### 对 (Pair)

```lean
-- 对类型
#check Nat × String
#check (42, "answer") : Nat × String

-- 对操作
def pair_example : Nat × String := (42, "answer")
def first (p : Nat × String) : Nat := p.1
def second (p : Nat × String) : String := p.2

-- 模式匹配
def swap {α β : Type} (p : α × β) : β × α :=
  match p with
  | (a, b) => (b, a)
```

### 选项 (Option)

```lean
-- 选项类型
#check Option Nat
#check some 42 : Option Nat
#check none : Option Nat

-- 选项操作
def safe_divide (a b : Nat) : Option Nat :=
  if b = 0 then none else some (a / b)

def option_map {α β : Type} (f : α → β) : Option α → Option β
  | none => none
  | some x => some (f x)
```

---

## 🎭 模式匹配 | Pattern Matching

### 基本模式匹配

```lean
-- 自然数模式匹配
def is_zero : Nat → Bool
  | 0 => true
  | _ => false

-- 列表模式匹配
def head {α : Type} : List α → Option α
  | [] => none
  | h :: _ => some h

-- 嵌套模式匹配
def complex_match : List (Nat × String) → String
  | [] => "empty"
  | [(n, s)] => s
  | (n, s) :: _ :: _ => s
```

### 守卫模式 (Guards)

```lean
-- 使用if-then-else
def abs : Int → Nat
  | n => if n ≥ 0 then n.toNat else (-n).toNat

-- 使用match with guards
def compare (a b : Nat) : Ordering
  | a, b => 
    if a < b then Ordering.lt
    else if a = b then Ordering.eq
    else Ordering.gt
```

### 模式匹配技巧

```lean
-- 使用as模式
def list_example : List Nat → String
  | [] => "empty"
  | xs@(h :: _) => s!"non-empty with head {h}"

-- 使用通配符
def ignore_second : Nat × Nat × Nat → Nat
  | (a, _, c) => a + c

-- 使用类型注解
def type_annotated : List Nat → Nat
  | ([] : List Nat) => 0
  | (h :: t : List Nat) => h + length t
```

---

## 🏗️ 类型类 | Type Classes

### 基本类型类

```lean
-- 可比较类型
#check Ord Nat
#check Nat < 5

-- 数值类型
#check Add Nat
#check 2 + 3

-- 可显示类型
#check Repr Nat
#eval toString 42
```

### 类型类实例

```lean
-- 为自定义类型实现类型类
inductive Color where
  | red
  | green
  | blue

-- 实现可比较
instance : Ord Color where
  compare := fun a b =>
    match a, b with
    | Color.red, Color.red => Ordering.eq
    | Color.red, _ => Ordering.lt
    | Color.green, Color.red => Ordering.gt
    | Color.green, Color.green => Ordering.eq
    | Color.green, Color.blue => Ordering.lt
    | Color.blue, Color.blue => Ordering.eq
    | Color.blue, _ => Ordering.gt

-- 实现可显示
instance : Repr Color where
  reprPrec := fun c _ =>
    match c with
    | Color.red => "red"
    | Color.green => "green"
    | Color.blue => "blue"
```

### 类型类约束

```lean
-- 带类型类约束的函数
def max {α : Type} [Ord α] (a b : α) : α :=
  if a < b then b else a

-- 多个类型类约束
def show_and_add {α : Type} [Add α] [Repr α] (a b : α) : String :=
  s!"{a} + {b} = {a + b}"

-- 类型类继承
class Semigroup (α : Type) where
  mul : α → α → α
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)

class Monoid (α : Type) extends Semigroup α where
  one : α
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a
```

---

## 🔍 类型推导 | Type Inference

### 自动类型推导

```lean
-- Lean可以推导出类型
def x := 42        -- 推导为 Nat
def y := "hello"   -- 推导为 String
def z := [1, 2, 3] -- 推导为 List Nat

-- 函数类型推导
def add a b := a + b  -- 推导为 Nat → Nat → Nat
def concat xs ys := xs ++ ys  -- 推导为 List α → List α → List α
```

### 类型注解

```lean
-- 显式类型注解
def explicit : Nat := 42
def function_type : Nat → Nat := fun x => x + 1

-- 部分类型注解
def partial : Nat → _ := fun x => x + 1  -- 返回类型推导为 Nat

-- 类型约束
def constrained {α : Type} [Add α] (x y : α) : α := x + y
```

---

## 📝 总结 | Summary

Lean的类型系统提供了：

1. **强类型安全**：编译时类型检查
2. **类型推导**：减少显式类型注解
3. **依赖类型**：支持类型依赖值
4. **类型类**：支持多态和抽象
5. **模式匹配**：优雅的数据处理
6. **函数式编程**：纯函数和不可变数据

这些特性使Lean成为形式化数学和程序验证的强大工具。

---

## 🔗 相关资源 | Related Resources

- [Lean 4参考手册](https://leanprover-community.github.io/lean4/doc/)
- [类型论基础](https://leanprover-community.github.io/lean4/doc/lean4/tutorials/type-theory.html)
- [函数式编程](https://leanprover-community.github.io/lean4/doc/lean4/tutorials/functional-programming.html)
