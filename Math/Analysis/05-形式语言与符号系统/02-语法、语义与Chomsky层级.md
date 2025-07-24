# 02-语法、语义与Chomsky层级 | Syntax, Semantics & Chomsky Hierarchy

---

## 1. 主题简介 | Topic Introduction

本节系统梳理形式语言的语法、语义及Chomsky层级，强调其在数学基础、元数学、哲学分析与知识体系创新中的作用。

This section systematically reviews the syntax, semantics, and Chomsky hierarchy of formal languages, emphasizing their roles in mathematical foundations, metamathematics, philosophical analysis, and knowledge system innovation.

---

## 2. 语法与语义 | Syntax & Semantics

- 理论基础：形式语法、生成规则、语义解释。
- 代表人物：乔姆斯基（Chomsky）、蒙塔格（Montague）
- 典型理论：上下文无关文法、语义学派。
- 形式化片段（Lean）：

```lean
-- 上下文无关文法的Lean定义
structure CFG (N T : Type) :=
  (start : N)
  (rules : set (N × list (N ⊕ T)))
```

---

## 3. Chomsky层级 | Chomsky Hierarchy

- 类型：正则语言、上下文无关语言、上下文相关语言、递归可枚举语言。
- 代表人物：乔姆斯基（Chomsky）
- 典型理论：Chomsky层级、自动机模型。
- 伪代码：

```python
# Chomsky层级判别伪代码
class ChomskyClassifier:
    def classify(self, grammar):
        # 判别文法类型
        pass
```

---

## 4. 递归扩展计划 | Recursive Expansion Plan

- 持续细化各类型文法、语义解释、自动机模型等分支。
- 强化多表征内容与国际化标准。

## 概念定义 | Concept Definition

- 中文定义：语法是形式语言中符号串的生成规则，决定了哪些表达式是合法的。语义则规定了符号串的意义或解释。Chomsky层级是Noam Chomsky提出的形式语言分类体系，包括正则语言、上下文无关语言、上下文相关语言和递归可枚举语言，反映了不同语法的表达能力。
- English Definition: Syntax refers to the generative rules of strings in a formal language, determining which expressions are well-formed. Semantics assigns meaning or interpretation to these strings. The Chomsky hierarchy, proposed by Noam Chomsky, is a classification of formal languages into regular, context-free, context-sensitive, and recursively enumerable languages, reflecting the expressive power of different grammars.
- 国际标准/权威来源：
  - ISO 80000-2:2019
  - Stanford Encyclopedia of Philosophy: Formal Language, Syntax, Semantics
  - Encyclopedia of Mathematics: Chomsky hierarchy, Syntax, Semantics
  - Wikipedia: Chomsky hierarchy, Syntax (logic), Semantics (logic)
- 相关批判与哲学反思：
  - 语法与语义的分离是形式化方法的基础，但也导致“语义空洞”与“形式主义危机”。
  - Chomsky层级揭示了自动机与语言的本质联系，但实际自然语言远比形式语言复杂。
  - 结构主义、符号学等流派对语法与语义的本体论和认识论提出多元批判。

---

### Chomsky层级简表 | Table of Chomsky Hierarchy

| 层级 | 英文 | 典型语法 | 典型自动机 | 表达能力 |
|---|---|---|---|---|
| 0型 | Recursively Enumerable | 图灵机 | 图灵机 | 最强，所有可计算语言 |
| 1型 | Context-Sensitive | 上下文相关文法 | 线性有界自动机 | 包含自然语言子集 |
| 2型 | Context-Free | 上下文无关文法 | 下推自动机 | 编译器、程序设计语言 |
| 3型 | Regular | 正则文法 | 有限自动机 | 最弱，正则表达式 |

---

### 理论历史与代表人物

- 乔姆斯基（Noam Chomsky）：Chomsky层级、生成语法
- Kleene、Post：正则语言、递归函数
- Turing：图灵机、可计算性理论

#### 代表性原话（中英对照）

- “Colorless green ideas sleep furiously.”（无色的绿色观念狂暴地沉睡着。）——Noam Chomsky（语法正确但语义荒谬的例子）
- “Syntax is the study of the principles and processes by which sentences are constructed.”（语法是研究句子构造原则与过程的学科。）——Noam Chomsky

---

### 形式化系统与证明片段

- 正则文法BNF示例：

```text
<letter> ::= a | b | ... | z
<word> ::= <letter> | <letter> <word>
```

- 上下文无关文法BNF示例：

```text
<expr> ::= <expr> + <term> | <term>
<term> ::= <term> * <factor> | <factor>
<factor> ::= ( <expr> ) | number
```

- 图灵机的形式定义（LaTeX）：

```latex
M = (Q, \Sigma, \Gamma, \delta, q_0, q_{accept}, q_{reject})
```

---

### 相关性与本地跳转

- 参见 [01-总览.md](./01-总览.md)
- 参见 [01-总览.md](../04-逻辑与公理系统/01-总览.md)
- 参见 [03-编程语言与AI推理.md](./03-编程语言与AI推理.md)
