# MWE｜正则语言泵引理（最小例）

## 1. 泵引理（正则语言必要条件，概述）

若语言 L 为正则，则存在泵长 p，使得对任意 |s|≥p 且 s∈L，都可分割 s=xyz，满足：

- |xy| ≤ p，|y| ≥ 1；
- 对所有 i≥0，有 xy^i z ∈ L。
（此为必要条件，非充分条件。）

## 2. 非正则语言示例：L = { 0^n 1^n | n ≥ 0 }

- 目标：证明 L 非正则。
- 反证法：假设 L 正则，取其泵长 p。选串 s = 0^p 1^p ∈ L。
- 根据泵引理，存在分割 s=xyz，|xy|≤p 且 |y|≥1，因此 y 只含 0。
- 抽取 i=0：得到 xz，其 0 的个数 < p，而 1 的个数仍为 p，所以 xz ∉ L。
- 与“对所有 i≥0，xy^i z ∈ L”矛盾。故 L 非正则。

## 3. 备注与练习

- 练习：用泵引理证明 { w ∈ {0,1}* | #0(w) = #1(w) } 非正则。
- 注意：泵引理是必要条件，不能用于证明“正则”。证明正则通常需构造 DFA/NFA 或正则表达式，或使用闭包性质。

## 4. 参考

- Hopcroft–Ullman–Motwani, Introduction to Automata Theory, Languages, and Computation.
- Sipser, Introduction to the Theory of Computation.
