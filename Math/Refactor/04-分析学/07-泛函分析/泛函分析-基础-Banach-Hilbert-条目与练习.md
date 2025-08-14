# 泛函分析 | 基础：Banach 与 Hilbert（条目与练习）

---

## 1. 学习导引 | Cognitive Primer

- 先修：度量/赋范/内积、线性代数
- 主线：赋范线性空间→完备化（Banach）→内积空间（Hilbert）→三大定理

---

## 2. 基础概念 | Basics

- 赋范线性空间（NLS）：(X,‖·‖)
- 完备性：柯西列极限存在 ⇒ Banach 空间
- 内积空间（IPS）：⟨·,·⟩ 诱导范数 ‖x‖=√⟨x,x⟩；完备 ⇒ Hilbert 空间

---

## 3. 三大基本定理 | Big Three

1) Hahn–Banach 延拓定理：有界线性泛函可延拓，且保持范数
2) Banach–Steinhaus（一致有界原理）：点态有界 ⇒ 一致有界
3) 开映射/闭图像定理：满射有界线性算子为开映射；闭图像 ⇒ 有界

---

## 4. Riesz 表示与投影 | Riesz & Projections

- Riesz 表示（Hilbert）：每个有界线性泛函 f 对应唯一 y，使 f(x)=⟨x,y⟩
- 正交投影：闭子空间 M⊂H 上存在唯一最近点；H = M ⊕ M^⊥

---

## 5. 典例 | Worked Examples

1) 证明：ℓ^p 完备（概念性提纲）
2) L^2[0,1] 是 Hilbert：⟨f,g⟩=∫ f g
3) 证明：有限维 NLS 等价于欧氏空间，范数等价

---

## 6. 练习（8题）| Exercises (8)

1) 证明：C([0,1]) 在一致范数下是 Banach 空间
2) 证明：Hahn–Banach 的分离形式于凸集的应用（概述）
3) 证明：有界线性算子空间 B(X,Y) 在算子范数下是 Banach
4) 给出：Hilbert 空间上最小二乘问题的正交投影解
5) 证明：有限维 NLS 上所有范数等价
6) 证明：闭图像定理的一个应用（解的存在唯一性）
7) 举例：Banach 不等式/Young 不等式在 L^p 上的应用
8) 讨论：弱收敛与弱*收敛的直观区别

---

## 7. 认知提示 | Tips

- 完备性保证极限存在；Hilbert 增加几何结构（角度、投影）
- 三大定理是算子理论与偏微分方程分析的基石

---

## 8. 参考 | References

- `https://en.wikipedia.org/wiki/Functional_analysis`
- `https://en.wikipedia.org/wiki/Hahn%E2%80%93Banach_theorem`
