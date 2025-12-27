# MWE｜(2,x) 在 Z[x] 中非主理想（最小例）

## 1. 断言

- 在整环 Z[x] 中，理想 I=(2,x) 不是主理想。

## 2. 证明提纲

- 反设 I=(f)。则 2∈(f), x∈(f)。
- 由 2∈(f) 知 f | 2，于是 f=±1, ±2。
- 若 f=±1，则 (f)=Z[x]，与 x∈I ⊊ Z[x] 矛盾。
- 若 f=±2，则 (f)={2g(x)}，不含任何非常数奇系数多项式；而 x∈I 非 2 的倍多项式，矛盾。
- 故 I 非主。

## 3. 启示

- Z[x] 非 PID；与 PID⇒UFD 的链条对比说明 UFD≠PID。

## 4. 参考

- Dummit & Foote, Abstract Algebra.
- Atiyah–Macdonald, Introduction to Commutative Algebra.

## 形式化细节补充：Z[x] 中 (2,x) 非主

- 设 (2,x)=(f)。令 cont(f)=c（内容），f=c·f_0，f_0 原始。若 c=2，则 (f)⊆(2)，与 x∉(2) 矛盾；故 c=1。
- 模 2 化到 F2[x]：有 (x)=(\bar f)。故 \bar f 与 x 伴随，deg f=1，写作 f=±x+2g(x)。
- 由 2∈(f)，存在 h(x) 使 2=f·h=(±x+2g)h。取常数项得 2=±0+2g(0)h(0)，故 h(0)=±1 且 g(0) 为整数。
- 再看 x-系数：左侧为 0，右侧首项为 ±h(0)·x + …，与 h(0)=±1 矛盾。故 (2,x) 不可能为主理想。
- 参考：Dummit–Foote, Abstract Algebra, “Ideals in Z[x]”。
