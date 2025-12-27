# MWE｜UFD ≠ PID：k[x,y] 为 UFD 但非 PID（最小例）

## 1. 断言

- 设 k 为域，则 k[x,y] 是唯一分解整环（UFD），但不是主理想整环（PID）。

## 2. 理由提纲

- UFD：k[x] 为 PID ⇒ UFD；多元情形 k[x,y] 仍为 UFD（高斯引理与多项式环 UFD 闭性）。
- 非 PID：理想 I=(x,y) 在 k[x,y] 中非主。若 I=(f)，则 x,y∈(f) ⇒ f|x 且 f|y ⇒ f 为单位或关联到一个变量的不可约因子；均矛盾于同时整除 x 与 y。

## 3. 启示

- UFD 与 PID 仅在某些情形一致（如欧几里得环、主理想域），一般不等价。

## 4. 参考

- Dummit & Foote, Abstract Algebra.
- Atiyah–Macdonald, Introduction to Commutative Algebra.

## 形式化细节补充：k[x,y] 为 UFD 非 PID

- 设 k 为域，R=k[x,y]。R 为 UFD（Gauss 引理与多元多项式的归纳），但非 PID。
- 证 (x,y) 非主：若 (x,y)=(f)，则 x,y ∈ (f)，于是 f|x 且 f|y。于 UFD 中，gcd(x,y)=1，故 f|1，f 为单位，矛盾（则 (f)=R≠(x,y)）。
- 结论：UFD 未必是 PID；与欧几里得环 ⇒ PID ⇒ UFD 链条相对照。
- 参考：Atiyah–Macdonald (1969), Dummit–Foote (2004)。
