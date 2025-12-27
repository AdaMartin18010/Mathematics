# MWE｜高斯整数 Z[i] 为欧几里得整环（最小例）

## 1. 范数

- 定义 N(a+bi)=a^2+b^2（非负整数）。

## 2. 带余除法（就近取整）

- 对任意 α,β∈Z[i], β≠0，存在 q,r∈Z[i] 使 α=βq+r 且 r=0 或 N(r)<N(β)。
- 构造：在 ℂ 中取 q≈α/β 的实部/虚部就近取整，令 r=α−βq，得 N(r)<N(β)。

## 3. 结论

- Z[i] 为欧几里得整环 ⇒ PID ⇒ UFD。

## 4. 参考

- Dummit & Foote, Abstract Algebra.
- Ireland & Rosen, A Classical Introduction to Modern Number Theory.
