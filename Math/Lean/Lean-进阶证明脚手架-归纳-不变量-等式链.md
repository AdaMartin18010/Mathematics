# Lean | 进阶证明脚手架（归纳·不变量·等式链）

---

## 1. 目标 | Goals

- 掌握归纳证明、不变量技巧、等式链推理
- 完成3个进阶练习：数学归纳、循环不变量、代数等式链

---

## 2. 核心技巧 | Core Techniques

### 2.1 数学归纳法

```lean
theorem sum_formula (n : Nat) : sum (range n) = n * (n - 1) / 2 := by
  induction n with
  | zero => simp
  | succ n ih => 
    simp [sum_range_succ]
    rw [ih]
    ring
```

### 2.2 不变量证明

```lean
def is_sorted (l : List Nat) : Prop :=
  match l with
  | [] => True
  | [x] => True
  | x :: y :: xs => x ≤ y ∧ is_sorted (y :: xs)

theorem insert_preserves_sorted (x : Nat) (l : List Nat) :
  is_sorted l → is_sorted (insert x l) := by
  intro h
  induction l with
  | nil => simp [insert]
  | cons y ys ih =>
    cases h with
    | cons h1 h2 =>
      by_cases h : x ≤ y
      · simp [insert, h, h1]
      · simp [insert, h]
        constructor
        · exact h1
        · exact ih h2
```

### 2.3 等式链推理

```lean
theorem distributivity (a b c : Nat) : a * (b + c) = a * b + a * c := by
  calc
    a * (b + c) = a * b + a * c := by rw [Nat.mul_add]
```

---

## 3. 练习（3题）| Exercises (3)

1) 证明：`theorem factorial_pos (n : Nat) : 0 < factorial n`
2) 证明：`theorem list_length_append (l1 l2 : List α) : (l1 ++ l2).length = l1.length + l2.length`
3) 证明：`theorem power_add (a : Nat) (m n : Nat) : a^(m + n) = a^m * a^n`

提示：

- 使用 `induction`、`constructor`、`cases`、`calc`
- 注意递归定义的模式匹配

---

## 4. 参考 | References

- `https://leanprover-community.github.io/mathematics_in_lean/`
- `https://leanprover-community.github.io/theorem_proving_in_lean4/`
