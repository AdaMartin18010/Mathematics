# MWE｜上下文无关语言泵引理（最小例）

## 1. 泵引理（Bar-Hillel 泵引理，必要条件）

若语言 L 为上下文无关（CFL），则存在常数 p，使任意 s∈L 且 |s|≥p，均可分解为 s=uvwxy，满足：

- |vx| ≥ 1；
- |vwx| ≤ p；
- 对所有 i ≥ 0，均有 u v^i w x^i y ∈ L。
（必要条件，非充分条件。）

## 2. 非CFL示例：L = { a^n b^n c^n | n ≥ 0 }

- 目标：证明 L 非上下文无关。
- 反证：设 L 为CFL，取其泵长 p。令 s = a^p b^p c^p ∈ L。
- 由于 |vwx| ≤ p，子串 vwx 最多跨越相邻两段（a块、b块、c块）之和，无法同时覆盖三段。
  - 情况A：v、x 仅含 a 与 b。泵后 a、b 计数同变而 c 不变 ⇒ a、b、c 不再等量 ⇒ 违反 L。
  - 情况B：v、x 仅含 b 与 c。泵后 b、c 同变而 a 不变 ⇒ 不等量 ⇒ 违反 L。
  - 情况C：v、x 仅落在单一块（全 a 或全 b 或全 c）。泵后该块计数改变，其余不变 ⇒ 不等量 ⇒ 违反 L。
- 因为 |vx|≥1，总能使某一块长度改变，从而与“对所有 i≥0，u v^i w x^i y ∈ L”矛盾。
- 结论：L 非上下文无关。

## 3. 练习

- 试用泵引理或 Ogden 引理证明 L' = { a^i b^j c^k | i=j 或 j=k } 非CFL。
- 思考：对 L'' = { a^n b^n c^m | n,m ≥ 0 }，泵引理为何不足以给出“正”的结论？（提示：构造PDA。）

## 4. 参考

- Bar-Hillel, Perles, Shamir (1961). On formal properties of simple phrase structure grammars.
- Hopcroft–Ullman–Motwani. Introduction to Automata Theory, Languages, and Computation.
- Sipser. Introduction to the Theory of Computation.
