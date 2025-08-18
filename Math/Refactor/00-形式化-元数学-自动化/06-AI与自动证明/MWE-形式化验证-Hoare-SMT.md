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

## 数组求和与越界安全（SMT-LIB 可检查）

```smt
(set-logic QF_AUFLIA)
; 整数数组 a: Int -> Int，长度 n，循环计数 i，累加 s
(declare-const n Int)
(declare-const i Int)
(declare-const s Int)
(declare-const s' Int)
(declare-const a (Array Int Int))

; 先验约束与不变量
(assert (>= n 0))
(assert (and (>= i 0) (<= i n)))
; s' 为执行一轮 i<n 时：s' = s + a[i]
(assert (= s' (+ s (select a i))))

; 目标性质：若 i<n，则访问安全且下一步保持 0<=i+1<=n
(assert (=> (< i n)
  (and (<= 0 i) (< i n) (<= 0 (+ i 1)) (<= (+ i 1) n))))

(check-sat)
(get-model)
```

说明：此片段验证“当 i<n 时一次迭代的数组访问安全与计数推进边界”在算术与数组理论下可满足；可据此叠加循环不变量框架验证整段循环。

## 全循环验证骨架：数组求和（Hoare + SMT）

- 目标：将 a[0..n-1] 求和至 s，验证后置条件 s=∑_{k< n} a[k]
- 循环：

```text
{ n≥0 ∧ i=0 ∧ s=0 }
I(i,s) ≡ 0≤i≤n ∧ s = ∑_{k< i} a[k]
while (i < n) do
  s := s + a[i];
  i := i + 1
od
{ s = ∑_{k< n} a[k] }
```

- 证明要点：
  1) 初始化：i=0, s=0 ⇒ I 成立（空求和）。
  2) 保持：I∧i<n ⇒ 一步后 I 仍成立（代数展开）。
  3) 结束：I∧¬(i<n) ⇒ i=n ⇒ 结论。
- 变体函数：V=n−i（自然数下降）保证终止性。
- SMT 助力：以断言模板编码 I 与一步保持性，枚举/实例化索引演算；或用数组公理化与触发模式近似检查一步保持与边界安全。

## 循环不变式 SMT 模板（含量词占位）

```smt
(set-logic AUFLIA)
(declare-const n Int)
(declare-const i Int)
(declare-const s Int)
(declare-const a (Array Int Int))

; 公理化前缀和 Sum(0,i)
(declare-fun Sum (Int Int (Array Int Int)) Int)
; 约束：Sum(0,0,a)=0, Sum(0,i+1,a)=Sum(0,i,a)+select a i
(assert (= (Sum 0 0 a) 0))
(assert (forall ((k Int)) (=> (and (>= k 0)) (= (Sum 0 (+ k 1) a) (+ (Sum 0 k a) (select a k))))))

; 不变式 I(i,s): 0<=i<=n ∧ s = Sum(0,i,a)
(assert (>= n 0))
(assert (and (>= i 0) (<= i n)))
(assert (= s (Sum 0 i a)))

; 一步更新保持性（模板）：若 i<n，则 s':=s+a[i], i':=i+1 后仍满足 I
(declare-const ip Int)
(declare-const sp Int)
(assert (=> (< i n)
  (and (= sp (+ s (select a i))) (= ip (+ i 1))
       (and (>= ip 0) (<= ip n))
       (= sp (Sum 0 ip a)))))

(check-sat)
```

说明：以抽象函数 Sum 编码前缀和并加入递归公理，规避显式展开；此模板可被 SMT 求解器用于检查一步保持与边界安全的可满足性。

## 循环不变式模板（数组、链表）

### 数组求和：循环不变式模板（SMT-LIB）

```text
(set-logic AUFLIA)
(declare-fun N () Int)
(declare-fun i () Int)
(declare-fun a (Int) Int)
(declare-fun s () Int)

;; 先验约束：
(assert (>= N 0))
(assert (and (>= i 0) (<= i N)))

;; 不变式 Inv(i,s): s = Σ_{k< i} a(k)
(define-fun Inv () Bool (= s (sum 0 i a)))

;; sum 的递归定义（展开骨架；实际可用递归/数组消去或以公理刻画）
(define-fun-rec sum ((l Int) (r Int) (f (Int) Int)) Int
  (ite (>= l r) 0 (+ (f l) (sum (+ l 1) r f))))

;; 初始化保持：i=0, s=0 ⇒ Inv
(assert (=> (and (= i 0) (= s 0)) Inv))

;; 迭代保持：Inv ∧ i<N 且 s' = s + a(i), i' = i+1 ⇒ Inv'
(declare-fun ip () Int)
(declare-fun sp () Int)
(assert (=> (and Inv (< i N) (= ip (+ i 1)) (= sp (+ s (a i))))
            (= sp (sum 0 ip a))))

;; 终止性质：Inv ∧ i=N ⇒ s = Σ_{k<N} a(k)
(assert (=> (and Inv (= i N)) (= s (sum 0 N a))))
(check-sat)
```

### 单链表长度：循环不变式模板（SMT-LIB）

- 以数组模拟 next 指针；`-1` 表示 null。

```text
(set-logic AUFLIA)
(declare-fun next (Int) Int)
(declare-fun head () Int)
(declare-fun cur () Int)
(declare-fun len () Int)

;; 断言：无环（这里用抽象断言，实际可接入可达性公理或界限展开）
(declare-fun Reach (Int Int) Bool)
(assert (forall ((x Int)) (not (Reach x x))))

;; 不变式：len 等于从 head 到 cur 的步数，且 cur 在 head 的可达集内
(declare-fun dist (Int Int) Int)
(define-fun Inv () Bool (and (>= len 0) (Reach head cur) (= len (dist head cur))))

;; 初始化：cur=head, len=0 ⇒ Inv
(assert (=> (and (= cur head) (= len 0)) Inv))

;; 迭代：Inv ∧ cur ≠ -1 ⇒ len' = len+1, cur' = next(cur) ⇒ Inv'
(declare-fun curp () Int)
(declare-fun lenp () Int)
(assert (=> (and Inv (not (= cur (- 1))) (= lenp (+ len 1)) (= curp (next cur)))
            (and (Reach head curp) (= lenp (dist head curp)))))

(check-sat)
```

- 说明：`Reach/dist` 可由可达性递归定义或以边界展开近似，在验证器中以公理或背景理论供给。

### 边界安全/内存分配（补充断言）

- 数组：读写前置条件

```text
(assert (=> (< i N) (and (>= i 0) (< i N))))
(assert (forall ((k Int)) (=> (and (>= k 0) (< k N)) true)))
```

- 单链表：已分配与非空前置

```text
(declare-fun isAlloc (Int) Bool)
(assert (=> (and (not (= cur (- 1))) (isAlloc cur)) (isAlloc (next cur))))
(assert (or (= cur (- 1)) (isAlloc cur)))
```

### 指针/别名安全（最小约束，SMT-LIB）

```text
(set-logic AUFLIA)
(declare-fun alloc () Int)
(declare-fun p () Int)
(declare-fun q () Int)
(declare-fun isAlloc (Int) Bool)

;; 新分配返回新地址（与已分配集合不别名）：
(assert (=> (isAlloc p) (distinct alloc p)))
(assert (=> (isAlloc q) (distinct alloc q)))

;; 写-读帧条件骨架：不同地址互不影响（抽象示意）
(declare-fun val (Int) Int)
(declare-fun store (Int Int Int) Int) ;; store(a,addr,v)
(assert (forall ((a Int) (addr Int) (v Int) (x Int))
  (=> (distinct x addr)
      (= (val x) (val x)))))
(check-sat)
```

### 数组切片求和（带切片不变式，SMT-LIB）

```text
(set-logic AUFLIA)
(declare-fun A (Int) Int)
(declare-fun L () Int)
(declare-fun R () Int)
(declare-fun i () Int)
(declare-fun s () Int)

(assert (and (<= 0 L) (<= L R)))

(define-fun-rec sum ((l Int) (r Int)) Int
  (ite (>= l r) 0 (+ (A l) (sum (+ l 1) r))))

;; 不变式：s = Σ_{k∈[L,i)} A(k)，且 L ≤ i ≤ R
(define-fun Inv () Bool (and (<= L i) (<= i R) (= s (sum L i))))

;; 初始化 i=L, s=0 保持 Inv
(assert (=> (and (= i L) (= s 0)) Inv))

;; 迭代 i' = i+1, s' = s + A(i)
(declare-fun ip () Int)
(declare-fun sp () Int)
(assert (=> (and Inv (< i R) (= ip (+ i 1)) (= sp (+ s (A i))))
            (= sp (sum L ip))))

;; 终止 i=R ⇒ s = Σ_{k∈[L,R)} A(k)
(assert (=> (and Inv (= i R)) (= s (sum L R))))
(check-sat)
```

## 二分查找全规格（SMT-LIB 骨架）

```text
(set-logic AUFLIA)
(declare-const n Int)
(declare-const low Int)
(declare-const high Int)
(declare-const mid Int)
(declare-const key Int)
(declare-const A (Array Int Int))

(assert (and (>= n 0) (>= low 0) (<= high n)))

; 递增有序（抽象表述，实际可用 ∀ 约束）：
(declare-fun sorted (Array Int Int) Bool)
(assert (sorted A))

; 不变式：0 ≤ low ≤ high+1 ≤ n，目标元素若存在则位于 [low, high]
(define-fun Inv () Bool (and (<= 0 low) (<= low (+ high 1)) (<= (+ high 1) n)))

; 迭代一步保持（骨架示意）：
(declare-const low' Int)
(declare-const high' Int)
(assert (=> (and Inv (< low high) (= mid (div (+ low high) 2)))
            (or (= low' (+ mid 1)) (= high' (- mid 1)))))

; 终止：low > high ⇒ 未找到；或 A[mid]=key ⇒ 找到
(declare-const found Bool)
(assert (=> (and Inv (not (< low high))) (or (not found) found)))
(check-sat)
```

- 说明：将保持性细化为 `A[mid] < key ⇒ low' = mid+1` 与 `A[mid] > key ⇒ high' = mid-1` 两分支，并加入 `found ⇒ select(A,mid)=key` 断言即可形成可运行验证脚手架。

## 二分查找：分支保持性（可运行 SMT-LIB）

```text
(set-logic QF_AUFLIA)
(declare-const n Int)
(declare-const low Int)
(declare-const high Int)
(declare-const mid Int)
(declare-const key Int)
(declare-const A (Array Int Int))
(declare-const low' Int)
(declare-const high' Int)

(assert (and (>= n 0) (>= low 0) (<= high (- n 1)) (<= low high)))
(assert (= mid (div (+ low high) 2)))

; 分支1：A[mid] < key ⇒ 区间右移
(assert (=> (< (select A mid) key)
            (and (= low' (+ mid 1)) (= high' high) (<= low' high'))))

; 分支2：A[mid] > key ⇒ 区间左移
(assert (=> (> (select A mid) key)
            (and (= low' low) (= high' (- mid 1)) (<= low' high'))))

; 边界一致性（若相等则可返回）：
(assert (=> (= (select A mid) key) (and (= low' low) (= high' high))))
(check-sat)
```

- 说明：将本片段与“全规格骨架”中的不变式联用，即可完成一次迭代的保持性验证。

## 二分查找：返回正确性与后置（可运行 SMT-LIB）

```text
(set-logic QF_AUFLIA)
(declare-const n Int)
(declare-const low Int)
(declare-const high Int)
(declare-const key Int)
(declare-const A (Array Int Int))
(declare-const found Bool)
(declare-const idx Int)

(assert (and (>= n 0) (>= low 0) (<= high (- n 1))))

; 返回分支1：found ⇒ A[idx]=key 且 idx∈[0,n)
(assert (=> found (and (<= 0 idx) (< idx n) (= (select A idx) key))))

; 返回分支2：¬found ⇒ ∀i∈[0,n), A[i]≠key（用有界全称可在量词支持下表达; 此处骨架）
; 可在可满足性检查中以 Skolem 化或展开界限替代

(check-sat)
```

- 说明：与前述不变式和分支保持性联用，可组成“部分正确性+返回后置”的整体验证脚手架。

### 未找到分支（量词化后置，AUFLIA）

```text
(set-logic AUFLIA)
(declare-const n Int)
(declare-const key Int)
(declare-const A (Array Int Int))

; 未找到 ⇒ ∀i∈[0,n): A[i] ≠ key
(assert (forall ((i Int)) (=> (and (<= 0 i) (< i n)) (not (= (select A i) key)))))
(check-sat)
```

### 未找到分支（无量词有界展开）

```text
(set-logic QF_AUFLIA)
(declare-const n Int)
(declare-const key Int)
(declare-const A (Array Int Int))
; 选择展开界 K，要求 0 ≤ K ≤ n
(declare-const K Int)
(assert (and (>= K 0) (<= K n)))
; 例：K=3 的展开（模板可程序化生成）
(assert (not (= (select A 0) key)))
(assert (=> (<= 2 (- K 1)) (not (= (select A 1) key))))
(assert (=> (<= 3 (- K 1)) (not (= (select A 2) key))))
(check-sat)
```

- 说明：对固定 K 展开有界多项约束，便于无量词求解器检查；实际可由脚本生成至 K=n。
