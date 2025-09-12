# Stone 表示（布尔代数 ↔ Stone 空间）

## 命题

- 每个布尔代数 B 与某个 Stone 空间 X 的 clopen 集合代数 Clop(X) 同构。

## 构造要点

- 底集：极大滤子空间 X = Ult(B)；拓扑基：Â(a) = { U ∈ X : a ∈ U }。
- 同态：φ: B → Clop(X)，a ↦ Â(a)；保 ∨、∧、补。
- 性质：X 零维、紧、豪斯多夫；Â(a) 为 clopen。

## 证明骨架（提纲）

- 运算保持：Â(a ∨ b) = Â(a) ∪ Â(b)，Â(a ∧ b) = Â(a) ∩ Â(b)，Â(¬a) = X \ Â(a)。
- 单射：a ≠ b ⇒ 存在极大滤子区分（超滤引理/Zorn 引理）。
- 满射：任意 clopen 都由某个 a 的 Â(a) 表示（Stone 基生成）。

## 接口与应用

- 逻辑：经典命题逻辑的代数/拓扑化语义。
- 计算：BDD/布尔函数与谱/超滤的桥接。

## 参考

- Johnstone, Stone Spaces
- Davey & Priestley, Introduction to Lattices and Order

---

返回与相关：

- 返回总览：[表示与对偶总览](./表示与对偶总览.md)
- 相关分支：
  - `03-序结构/02-主要分支/布尔代数.md`
  - `03-序结构/02-主要分支/格论.md`

---

最小示例（新增）

- 取 $B=\mathcal P(\{1\})=\{0,1\}$。极大滤子集 $\text{Ult}(B)=\{\{1\}\}$，拓扑只有 $\emptyset$ 与 $\{\{1\}\}$，显然是 Stone 空间。映射 $a\mapsto \widehat a$ 给出 $0\mapsto\emptyset$，$1\mapsto \{\{1\}\}$，与 Clop(X) 同构。

常见误区（新增）

- 误以为需选择公理的完整形式：有限/可数情形可直接构造；一般情形通常用超滤引理。
- 将“任意拓扑空间的 clopen 代数”都视为布尔代数表示：需 Stone 空间（零维紧豪斯多夫）条件。
