# MWE｜正则语言泵引理（最小例）

## 1. 陈述

若 L 为正则语言，则存在 p，使得 |s|≥p 的 s∈L 可写作 s=xyz，满足：

- |xy| ≤ p，|y| ≥ 1；
- 对所有 i≥0，x y^i z ∈ L。

## 2. 例：L={ a^n b^n | n≥0 } 非正则

- 取 p 为泵长。令 s=a^p b^p。
- 因 |xy|≤p，故 y 仅由 a 构成。取 i=0 抽去 y，则 a 与 b 数量失衡，x y^0 z ∉ L，矛盾。
- 故 L 非正则。

## 3. 备注

- 与 Ogden 引理比较：正则泵引理适用范围较弱；当需要精确控制抽取位置（上下文无关语言情形）时使用 Ogden 引理。

## 4. 参考

- Hopcroft–Ullman–Motwani. Introduction to Automata Theory, Languages, and Computation.
- Sipser. Introduction to the Theory of Computation.
