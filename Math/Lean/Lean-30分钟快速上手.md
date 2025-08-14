# Lean | 30分钟快速上手（含3个练习）

---

## 1. 目标 | Goals

- 了解 Lean 基本语法与交互
- 完成三个核心练习：命题证明、递归定义、简单归纳证明

---

## 2. 环境与基础 | Setup & Basics

- 工具：Lean 4 / VS Code + Lean 扩展
- 基本结构：theorem/def/lemma、namespace、import、#check/#eval

示例：

```lean
-- 自然数加法交换律（骨架）
open Nat

theorem add_comm_basic (a b : Nat) : a + b = b + a := by
  -- 归纳 a 或 b，使用 simp/arith 命令
  admit
```

---

## 3. 核心语法 | Core Syntax

- 定义：`def f (x : α) := ...`
- 定理：`theorem T : P := by ...`
- 模块：`namespace ... end`
- 推理：`intro`, `apply`, `have`, `calc`, `simp`, `rw`

---

## 4. 典例 | Worked Examples

1) 命题逻辑：`P → Q → P`

    ```lean
    theorem imp_example (P Q : Prop) : P → Q → P := by
      intro hP
      intro hQ
      exact hP
    ```

2) 递归定义与证明：

    ```lean
    open Nat

    def double (n : Nat) : Nat := 2 * n

    theorem double_zero : double 0 = 0 := by
      simp[double]
    ```

3) 归纳证明：

    ```lean
    open Nat

    theorem add_zero (n : Nat) : n + 0 = n := by
      induction n with
      | zero => simp
      | succ n ih => simp[ih]
    ```

---

## 5. 练习（3题）| Exercises (3)

1) 证明：`theorem id_left (P : Prop) : (True ∧ P) → P`
2) 定义 `pow2 : Nat → Nat` 使得 `pow2 n = 2^n`，并证明 `pow2 0 = 1`
3) 证明自然数加法结合律骨架：`(a + b) + c = a + (b + c)`（可用归纳与 simp）

提示：

- 使用 `intro`, `cases`, `simp`, `rw`, `induction`
- 打开命名空间：`open Nat`

---

## 6. 参考 | References

- `https://leanprover-community.github.io/`
- `https://lean-lang.org/`
