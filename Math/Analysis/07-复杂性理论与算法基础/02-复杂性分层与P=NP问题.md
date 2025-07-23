# 02-复杂性分层与P=NP问题 | Complexity Hierarchies & P vs NP Problem

---

## 1. 主题简介 | Topic Introduction

本节系统梳理复杂性分层与P=NP问题，包括P、NP、NP完全、PSPACE、EXPTIME等复杂性类，P=NP问题，强调其在数学基础、元数学、哲学分析与知识体系创新中的作用。

This section systematically reviews complexity hierarchies and the P vs NP problem, including P, NP, NP-complete, PSPACE, EXPTIME, etc., and the P=NP problem, emphasizing their roles in mathematical foundations, metamathematics, philosophical analysis, and knowledge system innovation.

---

## 2. 复杂性分层 | Complexity Hierarchies

- 理论基础：复杂性类、可计算性边界、资源约束。
- 主要类型：P、NP、NP完全、PSPACE、EXPTIME。
- 代表人物：库克（Cook）、卡普（Karp）、萨维奇（Savitch）
- 典型理论：复杂性分层、Savitch定理。
- 形式化片段（Lean）：

```lean
-- 复杂性类的Lean定义（简化）
inductive ComplexityClass
| P | NP | NPC | PSPACE | EXPTIME
```

---

## 3. P=NP问题 | P vs NP Problem

- 理论基础：多项式时间、非确定性、NP完全。
- 代表人物：库克（Cook）、卡普（Karp）、莱文（Levin）
- 典型理论：Cook-Levin定理、NP完全性。
- 伪代码：

```python
# P=NP问题判别伪代码（理论上未解决）
def is_p_equal_np():
    # 理论上未解决，仅作复杂性说明
    raise NotImplementedError
```

---

## 4. 递归扩展计划 | Recursive Expansion Plan

- 持续细化复杂性分层、P=NP、NP完全、PSPACE、EXPTIME等分支，补充代表性案例、历史事件、现代影响。
- 强化多表征内容与国际化标准。
