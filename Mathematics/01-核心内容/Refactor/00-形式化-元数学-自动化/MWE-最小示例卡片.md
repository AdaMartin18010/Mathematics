# MWE-最小示例卡片 | 形式化-元数学-自动化

## 1) 公理系统与逻辑基础

- 完备定理（Gödel completeness）：语义有效 ⇒ 句法可证（针对一阶逻辑）
- 紧致定理（Compactness）：若每个有限子集可满足，则整体可满足
- 用例卡：用紧致性从“逐步扩充的有限可满足”构造无穷模型

## 2) 形式语言与符号系统

- BNF（算术表达式原型）：

```text
<expr> ::= <expr> "+" <term> | <term>
<term> ::= <term> "*" <factor> | <factor>
<factor> ::= "(" <expr> ")" | <number>
```

- DFA（偶数个0的二进制串）：状态 {q0,q1}；初态/接受态 q0；在符号“0”上 q0↔q1 互相切换，“1”自环

## 3) 自动机理论与可计算性

- 典例卡：停机问题不可判定（对角化骨架）
- 机器草图：识别偶数个0的 DFA；识别 a^n b^n 的 PDA（栈计数）

## 4) AI与自动证明

- Lean 极小证明片段：

```lean
theorem imp_self (P : Prop) : P → P := fun h => h
```

- 提示：最小证明应可在任一主流定理证明器中复现（Lean/Coq/Isabelle）

---

- 参考：Wikipedia（Completeness/Compactness/Formal language/DFA），Lean/Coq 文档
- 断点：补充“形式化验证 MWE（SMT或Hoare三元组）”与“自动机构造的图示”
