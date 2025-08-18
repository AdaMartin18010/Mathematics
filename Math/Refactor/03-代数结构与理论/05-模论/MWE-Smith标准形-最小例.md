# MWE｜Smith 标准形（最小例）

## 1. 定义要点（Z 上整等价）
- 任意整数矩阵 A 可经初等行列变换（整可逆）化为对角矩阵 diag(d1,...,dr)，满足 d_i | d_{i+1}。
- 称该对角形为 Smith 正规形（SNF）。
- 结构意义：coker(A) ≅ ⊕_i Z/d_i Z ⊕ Z^{n-r}（n 为列数）。 [Hungerford 1974]

## 2. 极简例
- A = [ 4  2 ]（1×2 矩阵）。最大公因数 gcd(4,2)=2，SNF 为 [2  0]。
- 因此 coker(A) ≅ Z/2Z ⊕ Z（对应 2 与空余自由度）。

## 3. 2×2 例
- A = \begin{pmatrix} 2 & 4 \\ 6 & 8 \end{pmatrix}
  - 行列初等变换：
    1) R2 := R2 − 3·R1 → \begin{pmatrix} 2 & 4 \\ 0 & -4 \end{pmatrix}
    2) C2 := C2 + C1 → \begin{pmatrix} 2 & 6 \\ 0 & -4 \end{pmatrix}
    3) R1 := R1 − C2 的整合步（此处略去细节），最终得 SNF diag(2,2)。
  - 群解释：coker(A) ≅ Z/2Z ⊕ Z/2Z。

## 4. 阿贝尔群分解关联
- SNF 给出有限生成阿贝尔群的不变因子分解，不变因子即对角元 d_i。 [Dummit–Foote 2004]

## 5. 参考
- Hungerford, Algebra. Springer.
- Dummit & Foote, Abstract Algebra.
- Newman, Integral Matrices. 