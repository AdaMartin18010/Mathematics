# MWE｜形式化验证（Hoare 三元组 / SMT-LIB）

## 1) Hoare 三元组最小例

- 目标：证明赋值语句 `x := x + 1` 在前置条件 `x ≥ 0` 下满足后置条件 `x ≥ 1`。
- 形式：`{x ≥ 0} x := x + 1 {x ≥ 1}`
- 验证（赋值规则-最弱前置）：
  - 将后置条件中的 x 回代为 `x+1` 得到前置：`x+1 ≥ 1`，即 `x ≥ 0`
  - 前置成立 ⇒ 三元组成立

## 2) SMT-LIB 最小例（Z3）

```smt
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (>= x 0))
;; 反证法：若 x>=0 但 x+1 < 1，则不可满足
(assert (not (>= (+ x 1) 1)))
(check-sat) ; 期望 unsat
(get-model)
```

- 解释：若求解器返回 `unsat`，则性质 `(x≥0) ⇒ (x+1≥1)` 被验证。
- 可扩展：将程序抽象为过渡关系，引入循环不变式后以 `forall` 量词表达安全性。

## 3) 循环不变式极简例（计数循环）

- 程序：

```text
{x = 0 ∧ n ≥ 0}
while (x < n) do
  x := x + 1
od
{x = n}
```

- 不变式：I ≡ (0 ≤ x ≤ n)
  - 初始化：x=0 ⇒ I 成立
  - 保持：x<n 且 I 成立，执行 x:=x+1 后仍有 0≤x≤n
  - 终止：¬(x<n) ⇒ x≥n，与 I 合并得 x=n
- 终止度量：n - x 每次循环递减且有下界 0

## 4) 循环不变式例（数组边界与安全性）

- 程序骨架：

```text
{i = 0 ∧ 0 ≤ N ≤ len(A)}
while (i < N) do
  // 安全读取 A[i]
  acc := acc + A[i]
  i := i + 1
od
{i = N ∧ 0 ≤ N ≤ len(A)}
```

- 不变式：J ≡ (0 ≤ i ≤ N ∧ 0 ≤ N ≤ len(A))
  - 初始化：i=0 ⇒ 0≤i≤N；给定 0≤N≤len(A) ⇒ J 成立
  - 保持：若 i<N 且 J 成立，则 0≤i<N≤len(A) ⇒ 读 A[i] 安全；更新 i:=i+1 后仍有 0≤i≤N
  - 结束：¬(i<N) ⇒ i≥N，与 J 合并得 i=N 且 0≤N≤len(A)
- 结论：在循环全过程中访问均满足 0≤i<len(A)，数组读取安全

## 5) 术语与参考（补）

- invariant / variant / partial vs total correctness / array bounds safety
- 参考：Hoare Logic；Winskel《The Formal Semantics of Programming Languages》；Floyd–Hoare 方法

## 6) 数组累加的断言化SMT示例（骨架）

- 目标：程序在 0≤N≤len(A) 前提下，循环累加得到 acc = Σ_{i=0}^{N-1} A[i]。
- 模型化要点：
  - 以 i, N, len, A: Array Int 等作为变量，引入循环不变式 J ≡ (0 ≤ i ≤ N ≤ len ∧ acc = Σ_{k=0}^{i-1} A[k])。
  - 用蕴含约束刻画：初始化满足 J；保持性满足；结束条件合并 J 推出目标后置。
- SMT 伪代码骨架：

```smt
(set-logic ALL)
; 声明变量、数组与函数略
; 断言初始化：i=0 ∧ 0≤N≤len ⇒ acc = 0
; 断言保持：J ∧ i<N ⇒ J[i:=i+1, acc:=acc+A[i]]
; 断言终止：¬(i<N) ∧ J ⇒ acc = Sum(A, 0, N)
(check-sat)
```

- 备注：实际可运行版本需引入递归定义的 Sum 与数组选择/存储理论（Array theory），并可能使用归纳或 Horn 子句求解器。
