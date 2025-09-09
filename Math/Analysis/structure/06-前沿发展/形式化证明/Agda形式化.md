# Agda 形式化

---

## 🔄 与三大结构映射

- 拓扑结构：依赖类型与空间直觉的对应
- 代数结构：代数结构与等式推理的可计算编码
- 序结构：归纳-递归、良基归纳与序关系

## 进一步阅读（交叉链接）

- `../../01-拓扑结构/拓扑结构总览.md`
- `../../02-代数结构/代数结构总览.md`
- `../../03-序结构/序结构总览.md`
- `../../04-结构关系/结构关系总览.md`

## 返回导航

- 返回：`../../项目导航系统.md`

## 概述

依类型函数式语言，适合以编程风格发展数学。

## 要点

- 模块/记录与依类型编程
- Cubical Agda 与可计算Univalence

## 安装与快速开始

- 安装：`cabal install Agda` 或 `stack`，建议配合 VSCode + Agda 插件。
- 快速开始：创建 `.agda` 文件，加载标准库，启用 `--cubical` 可用路径等价。

## 最小示例（Agda）

```agda
module AddZero where

open import Agda.Builtin.Nat
open import Agda.Builtin.Equality

addZeroRight : (n : Nat) → n + 0 ≡ n
addZeroRight zero    = refl
addZeroRight (suc n) = cong suc (addZeroRight n)
```

提示：用交互编辑（C-c C-l 加载、C-c C-. 定位空洞）提升效率。

## 参考资料

- Norell, Towards a practical programming language based on dependent type theory
- The Agda Wiki & Documentation (agda.readthedocs.io)
- Vezzosi, Mörtberg, Abel, Cubical Agda: a dependently typed programming language with univalence
