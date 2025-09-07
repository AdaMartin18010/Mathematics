# Exercises 子项目 | Exercises Subproject

本子目录用于集中管理练习与参考解，并以独立 `lake` 项目形式提供一键构建与验证。

## 使用 | Usage

```bash
cd Exercises
lake build
```

## 结构 | Structure

```text
Exercises/
├─ lean-toolchain
├─ lakefile.lean
├─ README.md
├─ Basics/
│   └─ AddComm.lean
├─ Algebra/
│   ├─ Group.lean
│   ├─ Ring.lean
│   └─ Field.lean
├─ Analysis/
│   ├─ Real.lean
│   └─ Complex.lean
├─ Geometry/
│   └─ Euclidean.lean
├─ NumberTheory/
│   └─ Basic.lean
├─ Probability/
│   └─ Basic.lean
├─ Topology/
│   └─ Basic.lean
├─ CategoryTheory/
│   └─ Basic.lean
└─ MeasureTheory/
    └─ Basic.lean
```

## 练习分类 | Exercise Categories

- **Basics/**: 基础语法与类型系统练习
- **Algebra/**: 代数结构练习（群、环、域）
- **Analysis/**: 分析学练习（实数分析、复分析）
- **Geometry/**: 几何学练习（欧几里得几何）
- **NumberTheory/**: 数论练习（基础数论）
- **Probability/**: 概率论练习（基础概率论）
- **Topology/**: 拓扑学练习（基础拓扑）
- **CategoryTheory/**: 范畴论练习（基础范畴论）
- **MeasureTheory/**: 测度论练习（基础测度论）

## 验证 | Verification

所有练习都经过 `lake build` 验证，确保代码可编译运行。

---

更新时间：2025-01-15（新增拓扑学、范畴论、测度论练习，扩展代数和分析练习）
