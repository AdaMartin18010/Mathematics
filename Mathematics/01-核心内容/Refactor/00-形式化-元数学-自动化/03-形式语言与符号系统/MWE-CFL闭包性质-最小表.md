# MWE｜上下文无关语言（CFL）闭包性质最小表

## 1. 闭包

- 并、连接、Kleene 星：闭包
- 与正则语言的交：闭包
- 同态与逆同态：闭包

## 2. 非闭包

- 交与补：一般不闭包（但与正则交是闭包）
- 差：一般不闭包

## 3. 典型用法

- 借助与正则交的闭包，将结构约束为“切片”后再用泵引理/Ogden/Myhill–Nerode 等方法判定非CFL。

### 快速对照：操作与证明套路

| 操作 | 是否闭包 | 证明套路 | 典型示例 |
|---|---|---|---|
| 与正则交 | ✅ | 构造交替PDA/语法，或用已知定理 | {a^n b^n c^n} ∩ a*b* c* = {a^n b^n c^n}（用于“切片”） |
| 并/连接/星 | ✅ | 语法级组合或PDA并联/串联/循环 | CFL 基本性质 |
| 补 | ❌ | 反例或由交不闭包推得 | 结合德摩根与交不闭包 |
| 交 | ❌（与正则交为✅） | 反例（两个CFL交为非CFL） | L1={a^i b^j c^k \| i=j} 与 L2={a^i b^j c^k \| j=k}，L1∩L2={a^n b^n c^n} |

### 练习（3）

1) 证明：CFL 在同态与逆同态下闭包（给出语法/自动机视角之一）。
2) 构造两个 CFL 使其交为非CFL，并说明为何与正则交仍闭包。
3) 给出一个利用“与正则交”简化结构后，再用 Ogden 引理判非CFL的完整证明大纲。

## 4. 参考

- Hopcroft–Ullman–Motwani. Introduction to Automata Theory, Languages, and Computation.
- Sipser. Introduction to the Theory of Computation.

## 5. 交叉链接

- `MWE-CFG泵引理-最小例.md`：CFL 泵引理与非CFL判定示例
- `MWE-Ogden引理-最小例.md`：加强版泵引理与标记方法
- `MWE-CFG最小解析-最小例.md`：语法视角与 PDA 构造的互证
- `MWE-PDA构造-最小例.md`：PDA 视角与与正则交的构造套路
