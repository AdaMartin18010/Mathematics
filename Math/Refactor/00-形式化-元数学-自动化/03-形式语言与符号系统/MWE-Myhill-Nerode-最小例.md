# MWE｜Myhill–Nerode 定理（最小例）

## 1. 陈述（最小可辨识等价）

- 定义等价：x ~_L y ⇔ ∀z, xz∈L ⇔ yz∈L。
- 定理：L 为正则语言 ⇔ ~_L 的等价类个数有限；且最小 DFA 的状态数等于等价类个数。

## 2. 例：L = { a^n b^n | n≥0 } 的等价类无限

- 对任意 m≠n，取区分后缀 z=b^m。则 a^n b^m ∈ L ⇔ n=m。
- 因此 a^n 与 a^m 落在不同 ~_L 等价类，等价类无限，故 L 非正则。

## 3. 用法

- 相比泵引理，Myhill–Nerode 常给出更结构化的“不可正则”证明，且可用于构造最小 DFA。

## 4. 参考

- Hopcroft–Ullman–Motwani. Introduction to Automata Theory, Languages, and Computation.
- Sipser. Introduction to the Theory of Computation.

## 5. 与 Hopcroft 最小化的往返引用

- 参见《DFA 最小化｜Hopcroft 算法骨架》：`DFA最小化-Hopcroft算法-骨架.md`（分割细化、复杂度、witness 构造与可视化）。
- 本文的等价类与最短区分词可直接作为 Hopcroft 细化步骤中的“分裂见证”。

## 构造最小 DFA：二进制被 3 整除

- 语言：L = { w∈{0,1}* | 按二进制解释的值 ≡ 0 (mod 3) }。
- 等价类：按余数 0,1,2 分类，最小 DFA 有 3 个状态。
- 转移：读入位 b∈{0,1} 等价于状态更新 r ↦ (2r+b) mod 3。
- 初态 0 为接受态；其余按更新规则构造即可。由等价类有限知 L 正则，且该 DFA 最小。

### 最小 DFA 转移表（被 3 整除）

- 状态含义：r0=余数0（接受），r1=余数1，r2=余数2；输入字母表 {0,1}。

| 状态 | 0 | 1 |
| ---- |---| ---- |
| r0 (✔) | r0 | r1 |
| r1 | r2 | r0 |
| r2 | r1 | r2 |

- 说明：读 0 相当于乘 2 取模 3；读 1 相当于乘 2 加 1 取模 3。由 Myhill–Nerode 不可区分关系三分得到最小 3 态。

### 等价类与最短区分词示例

- 等价类划分：E0={所有右语言余数为0的前缀}，E1，E2 分别对应余数 1 与 2。
- 最短区分词（示例）：
  - 区分 E0 与 E1：w=ε 时不可区分；取后缀 z=1，可使接受性不同（或取 z=0 也可）。
  - 区分 E1 与 E2：取 z=1；
  - 区分 E0 与 E2：取 z=0。
 以上 witness 可由“配对 BFS”最短区分词算法自动得到（见 Hopcroft 文档）。
