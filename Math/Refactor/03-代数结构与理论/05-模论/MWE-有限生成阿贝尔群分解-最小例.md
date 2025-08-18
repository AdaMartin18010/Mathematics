# MWE｜有限生成阿贝尔群分解（最小例）

## 1. 结构定理（PID 上有限生成模）

- 任意有限生成阿贝尔群 G 同构于若干循环群直和：
  - 基本因子分解：G ≅ ⊕_i Z/p_i^{k_i}Z；
  - 不变因子分解：G ≅ ⊕_j Z/d_j Z，且 d_j | d_{j+1}。
- 两种表述等价。 [Dummit–Foote 2004; Hungerford 1974]

## 2. 极小例：Z/12Z

- 素因子：12 = 2^2 · 3。
- 基本因子：Z/12Z ≅ Z/4Z ⊕ Z/3Z（由 CRT 或结构定理）。
- 不变因子：即 Z/12Z 本身；与上式等价（直和下同构）。

## 3. 直和例：Z/8Z ⊕ Z/12Z 的分解

- 基本因子展开：Z/8Z ⊕ Z/12Z ≅ (Z/8Z) ⊕ (Z/4Z ⊕ Z/3Z)
  ≅ Z/8Z ⊕ Z/4Z ⊕ Z/3Z。
- 也可化为不变因子：将 2-主要分量 Z/8 ⊕ Z/4 规整，得到等价不变因子序列（如 Z/24Z ⊕ Z/4Z ⊕ Z/3Z 再与 3-分量整合为 Z/24Z ⊕ Z/12Z 等等；给出一种可能的不变因子序列需按 gcd/lcm 规约）。
- 练习：计算其不变因子标准形（确保整除链）。

## 4. 参考

- Dummit, D. S., & Foote, R. M. (2004). Abstract Algebra (3rd ed.). Wiley.
- Hungerford, T. W. (1974). Algebra. Springer.
- Atiyah, M. F., & Macdonald, I. G. (1969). Introduction to Commutative Algebra.
