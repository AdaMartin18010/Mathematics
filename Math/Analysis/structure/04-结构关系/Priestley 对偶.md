# Priestley 对偶（分配格 ↔ Priestley 空间）

## 命题

- 分配格与 Priestley 空间范畴对偶。

## 要点

- 结构：偏序 + 紧致拓扑 + 分离（Priestley 分离公理）。
- 映射：保持序的连续映射（order-continuous）。
- 表示：格元素对应上闭且 clopen 的集合；同态对应逆像保持。

## 作用

- 为分配格提供几何/拓扑直观与完备的表示理论，连接概念格与知识表示。

## 参考

- Davey & Priestley, Introduction to Lattices and Order

---

返回与相关：

- 返回总览：[表示与对偶总览](./表示与对偶总览.md)
- 相关分支：
  - `03-序结构/02-主要分支/格论.md`

---

最小示例（新增）

- 取两点分配格 $L=\{0,1\}$。Priestley 空间为两点离散拓扑并取离散序，元素 0 ↦ 空集，1 ↦ 全集（上闭且 clopen）。

常见误区（新增）

- 混淆“上闭 clopen”与任意 clopen：对偶中使用的是与序相容的上闭 clopen。
- 误将任意偏序+紧致拓扑视为 Priestley 空间：需满足分离公理（通过有序分离开点）。
