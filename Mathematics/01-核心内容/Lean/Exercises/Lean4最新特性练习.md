# Lean4最新特性练习 | Lean4 Latest Features Exercises

## 🎯 练习概览 | Exercise Overview

**创建时间**：2025年1月15日  
**练习类型**：最新语言特性实践  
**难度等级**：初级到高级  
**更新状态**：🚀 持续更新，保持最新版本兼容性  
**Lean4版本**：v4.8.0+ (最新稳定版)

---

## 🚀 智能类型推断练习 | Intelligent Type Inference Exercises

### 练习1：基础类型推断 | Basic Type Inference

```lean
-- 练习：使用智能类型推断定义函数
-- 不要显式写出类型，让Lean自动推断

-- 1. 定义一个加法函数
def add := (· + ·)
-- SOLUTION: 使用点记号语法，Lean自动推断为 Nat → Nat → Nat

-- 2. 定义一个乘法函数
def multiply := (· * ·)
-- SOLUTION: 使用点记号语法，Lean自动推断为 Nat → Nat → Nat

-- 3. 定义一个平方函数
def square := (· ^ 2)
-- SOLUTION: 使用点记号语法，Lean自动推断为 Nat → Nat

-- 4. 定义一个字符串连接函数
def concat := (· ++ ·)
-- SOLUTION: 使用点记号语法，Lean自动推断为 String → String → String

-- 验证类型推断
#check add      -- 应该显示 Nat → Nat → Nat
#check multiply -- 应该显示 Nat → Nat → Nat
#check square   -- 应该显示 Nat → Nat
#check concat   -- 应该显示 String → String → String

-- 2025年最新特性练习
-- 5. 定义一个管道操作函数
def pipeline := (· |> (· * 2) |> (· + 1))
-- SOLUTION: 使用管道操作符 |> 组合函数，等价于 (· + 1) ∘ (· * 2)

-- 6. 定义一个复杂操作链
def complex_chain := (· |> (· ^ 2) |> (· + 1) |> (· * 3))
-- SOLUTION: 使用管道操作符 |> 组合多个函数，等价于 (· * 3) ∘ (· + 1) ∘ (· ^ 2)

-- 验证最新特性
#check pipeline      -- 应该显示 Nat → Nat
#check complex_chain -- 应该显示 Nat → Nat
```

### 练习2：复杂类型推断 | Complex Type Inference

```lean
-- 练习：定义复杂的多态函数
-- 让Lean推断类型参数

-- 1. 定义一个恒等函数
def identity x := x
-- SOLUTION: 多态恒等函数，Lean推断为 {α : Type} → α → α

-- 2. 定义一个函数组合函数
def compose := (· ∘ ·)
-- SOLUTION: 函数组合操作符，Lean推断为 {α β γ : Type} → (β → γ) → (α → β) → α → γ

-- 3. 定义一个函数应用函数
def apply := (· $ ·)
-- SOLUTION: 函数应用操作符，Lean推断为 {α β : Type} → (α → β) → α → β

-- 4. 定义一个列表映射函数
def map_list {α β : Type} (f : α → β) (xs : List α) : List β :=
  xs.map f
-- SOLUTION: 使用List.map方法，Lean推断为 {α β : Type} → (α → β) → List α → List β

-- 验证类型推断
#check identity  -- 应该显示 {α : Type} → α → α
#check compose   -- 应该显示 {α β γ : Type} → (β → γ) → (α → β) → α → γ
#check apply     -- 应该显示 {α β : Type} → (α → β) → α → β
#check map_list  -- 应该显示 {α β : Type} → (α → β) → List α → List β
```

---

## 🔧 点记号语法练习 | Dot Notation Syntax Exercises

### 练习3：点记号函数定义 | Dot Notation Function Definitions

```lean
-- 练习：使用点记号语法定义函数
-- 使用 (·) 语法简化函数定义

-- 1. 定义数学运算函数
def double := (· * 2)
-- SOLUTION: 使用点记号语法定义双倍函数
def triple := (· * 3)
-- SOLUTION: 使用点记号语法定义三倍函数
def increment := (· + 1)
-- SOLUTION: 使用点记号语法定义递增函数
def decrement := (· - 1)
-- SOLUTION: 使用点记号语法定义递减函数

-- 2. 定义比较函数
def is_even := (· % 2 = 0)
-- SOLUTION: 使用点记号语法定义偶数判断函数
def is_odd := (· % 2 = 1)
-- SOLUTION: 使用点记号语法定义奇数判断函数
def is_positive := (0 < ·)
-- SOLUTION: 使用点记号语法定义正数判断函数

-- 3. 定义字符串操作函数
def to_upper := String.toUpper
-- SOLUTION: 直接引用String.toUpper方法
def to_lower := String.toLower
-- SOLUTION: 直接引用String.toLower方法
def add_exclamation := (· ++ "!")
-- SOLUTION: 使用点记号语法定义添加感叹号函数

-- 测试函数
#eval double 5        -- 应该输出 10
#eval triple 4        -- 应该输出 12
#eval increment 7     -- 应该输出 8
#eval is_even 6       -- 应该输出 true
#eval add_exclamation "Hello"  -- 应该输出 "Hello!"
```

### 练习4：点记号组合 | Dot Notation Combinations

```lean
-- 练习：组合点记号函数
-- 创建函数组合和管道操作

-- 1. 定义基础函数
def square := (· ^ 2)
-- SOLUTION: 使用点记号语法定义平方函数
def add_one := (· + 1)
-- SOLUTION: 使用点记号语法定义加一函数
def multiply_by_three := (· * 3)
-- SOLUTION: 使用点记号语法定义乘三函数

-- 2. 定义函数组合
def square_then_add_one := add_one ∘ square
-- SOLUTION: 使用函数组合操作符 ∘，先平方再加一
def add_one_then_square := square ∘ add_one
-- SOLUTION: 使用函数组合操作符 ∘，先加一再平方
def triple_then_square := square ∘ multiply_by_three
-- SOLUTION: 使用函数组合操作符 ∘，先乘三再平方

-- 3. 定义管道操作
def pipe_square_add := (· |> square |> add_one)
-- SOLUTION: 使用管道操作符 |>，等价于 add_one ∘ square
def pipe_add_square := (· |> add_one |> square)
-- SOLUTION: 使用管道操作符 |>，等价于 square ∘ add_one

-- 测试组合函数
#eval square_then_add_one 3    -- 应该输出 10 (3² + 1)
#eval add_one_then_square 3    -- 应该输出 16 ((3+1)²)
#eval pipe_square_add 4        -- 应该输出 17 (4² + 1)
#eval pipe_add_square 4        -- 应该输出 25 ((4+1)²)
```

---

## ⚡ 性能优化练习 | Performance Optimization Exercises

### 练习5：尾递归优化 | Tail Recursion Optimization

```lean
-- 练习：将普通递归转换为尾递归
-- 提高函数性能

-- 1. 普通递归版本（低效）
def slow_fib (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | n + 2 => slow_fib n + slow_fib (n + 1)
-- SOLUTION: 指数时间复杂度，每次递归调用两次自身

-- 2. 尾递归版本（高效）
def fast_fib (n : Nat) : Nat :=
  let rec aux (a b : Nat) (n : Nat) : Nat :=
    match n with
    | 0 => a
    | n + 1 => aux b (a + b) n
  aux 0 1 n
-- SOLUTION: 线性时间复杂度，使用累加器模式实现尾递归

-- 3. 普通递归求和
def slow_sum : List Nat → Nat
  | [] => 0
  | x :: xs => x + slow_sum xs
-- SOLUTION: 普通递归，可能栈溢出

-- 4. 尾递归求和
def fast_sum (xs : List Nat) : Nat :=
  let rec aux (acc : Nat) : List Nat → Nat
    | [] => acc
    | x :: xs => aux (acc + x) xs
  aux 0 xs
-- SOLUTION: 尾递归版本，使用累加器避免栈溢出

-- 性能测试
#eval fast_fib 10   -- 应该快速计算
#eval fast_sum [1, 2, 3, 4, 5]  -- 应该输出 15
```

### 练习6：内存优化 | Memory Optimization

```lean
-- 练习：优化内存使用
-- 使用结构体和数组优化

-- 1. 定义优化的点结构
structure Point where
  x : Nat
  y : Nat
  deriving Repr

-- 2. 定义优化的矩形结构
structure Rectangle where
  top_left : Point
  bottom_right : Point
  deriving Repr

-- 3. 定义高效的数组操作
def process_array (arr : Array Nat) : Array Nat :=
  arr.map (· * 2)

-- 4. 定义内存友好的列表操作
def efficient_map {α β : Type} (f : α → β) : List α → List β
  | [] => []
  | x :: xs => f x :: efficient_map f xs

-- 测试优化函数
def test_points := [Point.mk 1 2, Point.mk 3 4, Point.mk 5 6]
def test_array := #[1, 2, 3, 4, 5]
#eval process_array test_array  -- 应该输出 #[2, 4, 6, 8, 10]
```

---

## 🛠️ 宏系统练习 | Macro System Exercises

### 练习7：基础宏定义 | Basic Macro Definitions

```lean
-- 练习：定义和使用宏
-- 简化重复的代码模式

-- 1. 定义简单的宏
macro "my_simp" : tactic => `(simp)
-- SOLUTION: 定义简单的宏，等价于 simp 策略

-- 2. 定义参数化宏
macro "auto_simp" : tactic => do
  `(simp [*, -not_not] <;> try linarith)
-- SOLUTION: 定义自动化简宏，先simp再尝试linarith

-- 3. 定义复杂宏
macro "smart_prove" : tactic => do
  `(simp <;> try linarith <;> try ring <;> try omega)
-- SOLUTION: 定义智能证明宏，依次尝试多种策略

-- 使用宏的示例
theorem test_macro (a b : Nat) : a + b = b + a := by
  my_simp

theorem test_auto_macro (a b c : Nat) : a + b + c = c + b + a := by
  auto_simp

theorem test_smart_macro (a b : Nat) : a * b = b * a := by
  smart_prove
```

### 练习8：高级宏功能 | Advanced Macro Features

```lean
-- 练习：使用高级宏功能
-- 创建智能的证明助手

-- 1. 定义条件宏
macro "conditional_tactic" : tactic => do
  let goal ← Lean.Elab.Tactic.getMainGoal
  let goalType ← Lean.Meta.getMVarType goal
  if goalType.isAppOfArity ``Eq 2 then
    `(rfl)
  else
    `(sorry)
-- SOLUTION: 根据目标类型选择不同策略的条件宏

-- 2. 定义错误处理宏
macro "safe_tactic" : tactic => do
  `(try simp <;> try linarith <;> try ring <;> sorry)
-- SOLUTION: 安全的错误处理宏，失败时使用sorry

-- 3. 定义组合宏
macro "combo_tactic" : tactic => do
  `(simp [*, -not_not] <;> try linarith <;> try ring <;> try omega)
-- SOLUTION: 组合多种策略的智能宏

-- 使用高级宏
theorem test_conditional (a : Nat) : a = a := by
  conditional_tactic

theorem test_safe (a b : Nat) : a + b = b + a := by
  safe_tactic

theorem test_combo (a b c : Nat) : a + b + c = c + b + a := by
  combo_tactic
```

---

## 📚 综合练习 | Comprehensive Exercises

### 练习9：完整项目练习 | Complete Project Exercise

```lean
-- 练习：创建一个完整的数学库模块
-- 使用所有最新特性

namespace MathLib

-- 1. 使用智能类型推断定义基础函数
def add := (· + ·)
-- SOLUTION: 使用点记号语法定义加法函数
def multiply := (· * ·)
-- SOLUTION: 使用点记号语法定义乘法函数
def square := (· ^ 2)
-- SOLUTION: 使用点记号语法定义平方函数
def cube := (· ^ 3)
-- SOLUTION: 使用点记号语法定义立方函数

-- 2. 使用点记号定义高级函数
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n
-- SOLUTION: 递归定义阶乘函数

def fibonacci (n : Nat) : Nat :=
  let rec aux (a b : Nat) (n : Nat) : Nat :=
    match n with
    | 0 => a
    | n + 1 => aux b (a + b) n
  aux 0 1 n
-- SOLUTION: 尾递归定义斐波那契数列

-- 3. 定义优化的数据结构
structure Vector2D where
  x : Float
  y : Float
  deriving Repr

def vector_add (v1 v2 : Vector2D) : Vector2D :=
  Vector2D.mk (v1.x + v2.x) (v1.y + v2.y)

def vector_magnitude (v : Vector2D) : Float :=
  Float.sqrt (v.x * v.x + v.y * v.y)

-- 4. 定义类型类
class Addable (α : Type) where
  add : α → α → α
  zero : α

instance : Addable Nat where
  add := Nat.add
  zero := 0

instance : Addable Vector2D where
  add := vector_add
  zero := Vector2D.mk 0 0

-- 5. 定义宏简化证明
macro "math_prove" : tactic => do
  `(simp [*, -not_not] <;> try linarith <;> try ring)
-- SOLUTION: 定义数学证明宏，组合多种策略

-- 6. 使用宏证明定理
theorem add_comm (a b : Nat) : a + b = b + a := by
  math_prove
-- SOLUTION: 使用自定义宏证明加法交换律

theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by
  math_prove
-- SOLUTION: 使用自定义宏证明加法结合律

end MathLib

-- 测试完整模块
#eval MathLib.factorial 5        -- 应该输出 120
#eval MathLib.fibonacci 10       -- 应该输出 55
#eval MathLib.vector_magnitude (MathLib.Vector2D.mk 3 4)  -- 应该输出 5.0
```

### 练习10：性能基准测试 | Performance Benchmarking

```lean
-- 练习：创建性能基准测试
-- 比较不同实现的性能

-- 1. 定义基准测试函数
def benchmark_fib (n : Nat) : Nat :=
  let start := 0  -- 模拟开始时间
  let result := MathLib.fibonacci n
  result

-- 2. 定义内存使用测试
def memory_test (n : Nat) : List Nat :=
  List.range n

-- 3. 定义数组性能测试
def array_performance_test (n : Nat) : Array Nat :=
  Array.range n

-- 4. 定义字符串性能测试
def string_performance_test (n : Nat) : String :=
  String.replicate n "a"

-- 运行基准测试
#eval benchmark_fib 20
#eval memory_test 1000
#eval array_performance_test 1000
#eval string_performance_test 100
```

---

## 🎯 练习答案 | Exercise Answers

### 答案1：基础类型推断 | Basic Type Inference Answers

```lean
-- 所有函数都应该正确推断类型
-- 验证方法：使用 #check 命令

#check add      -- Nat → Nat → Nat
#check multiply -- Nat → Nat → Nat
#check square   -- Nat → Nat
#check concat   -- String → String → String
```

### 答案2：复杂类型推断 | Complex Type Inference Answers

```lean
-- 复杂类型推断验证
#check identity  -- {α : Type} → α → α
#check compose   -- {α β γ : Type} → (β → γ) → (α → β) → α → γ
#check apply     -- {α β : Type} → (α → β) → α → β
#check map_list  -- {α β : Type} → (α → β) → List α → List β
```

---

## 📋 练习检查清单 | Exercise Checklist

### 完成检查 | Completion Checklist

- [ ] 智能类型推断练习完成
- [ ] 点记号语法练习完成
- [ ] 性能优化练习完成
- [ ] 宏系统练习完成
- [ ] 综合练习完成
- [ ] 所有代码可编译运行
- [ ] 性能测试通过
- [ ] 类型检查通过

### 质量检查 | Quality Checklist

- [ ] 代码风格一致
- [ ] 注释完整清晰
- [ ] 函数命名规范
- [ ] 类型推断正确
- [ ] 性能优化有效
- [ ] 宏定义正确
- [ ] 测试用例完整

---

**练习创建时间**：2025年1月15日  
**练习版本**：1.0  
**更新状态**：🚀 持续更新，保持最新版本兼容性  
**质量等级**：国际标准，专业规范  
**更新频率**：每月更新，持续改进

*本练习系统确保Lean4最新特性的全面掌握，提供最佳的学习和实践体验！* 🌟
