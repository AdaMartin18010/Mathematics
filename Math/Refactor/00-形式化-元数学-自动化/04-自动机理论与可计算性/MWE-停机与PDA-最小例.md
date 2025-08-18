# MWE｜停机问题与 PDA（最小例）

## 1. 停机问题不可判定（骨架）

- 假设存在判定器 H(p,x) 判断“程序 p 在输入 x 上是否停机”。
- 构造 D(s)：若 H(s,s)="停机" 则循环，否则立即停机。
- 考察 D(⌜D⌝)：矛盾 ⇒ H 不存在 ⇒ 停机问题不可判定。

## 2. PDA 识别 a^n b^n（n≥1）

- 思路：读入 a 时入栈，读入 b 时出栈；最后栈空且读尽接受。
- 要点：上下文无关语言需下推自动机（带栈）。

## 3. 术语对照

- Halting problem / reduction / PDA / pushdown stack

## 4. 参考

- Turing (1936); Hopcroft & Ullman, Automata Theory
