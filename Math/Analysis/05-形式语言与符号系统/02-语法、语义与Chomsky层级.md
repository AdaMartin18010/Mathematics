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
