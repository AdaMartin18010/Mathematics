# MWE｜完备与紧致（最小例）

## 1. 完备定理（Gödel Completeness）

- 陈述（针对一阶逻辑）：若句子 φ 在所有模型中为真（语义有效），则存在形式化证明使得 ⊢ φ（句法可证）。
- 含义：语义真理与句法可证在一阶层面吻合。

## 2. 紧致性定理（Compactness）

- 陈述：若一阶句子族 Σ 的每个有限子集都可满足，则 Σ 可满足。
- 经典应用（构造无穷模型）：
  - Σ = {“存在至少 n 个两两不同元素” | n ∈ ℕ}
  - 任意有限子集只要求到 n=N，可在大小为 N 的有限结构中满足
  - 由紧致性，Σ 整体可满足 ⇒ 存在无穷模型

## 3. 术语对照

- 有效 validity / 可证明 provability / 紧致 compactness / 可满足 satisfiable

## 4. 参考

- Wikipedia: Completeness theorem / Compactness theorem
- Enderton, A Mathematical Introduction to Logic
