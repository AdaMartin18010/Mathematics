# Isabelle/HOL 形式化

---

## 🔄 与三大结构映射

- 拓扑结构：拓扑空间、公理化拓扑在Isabelle/HOL中的开发
- 代数结构：群/环/域/模与代数定理自动化证明
- 序结构：偏序/格/布尔代数形式化与序拓扑

## 进一步阅读（交叉链接）

- [../../01-拓扑结构/拓扑结构总览.md](../../01-拓扑结构/拓扑结构总览.md)
- [../../02-代数结构/代数结构总览.md](../../02-代数结构/代数结构总览.md)
- [../../03-序结构/序结构总览.md](../../03-序结构/序结构总览.md)
- [../../04-结构关系/结构关系总览.md](../../04-结构关系/结构关系总览.md)

## 返回导航

- 返回：[../../项目导航系统.md](../../项目导航系统.md)

## 概述

基于高阶逻辑的交互式定理证明器，自动化能力强。

## 要点

- Isar 证明语言
- Sledgehammer 集成自动定理证明

## 安装与快速开始

- 安装：从官网下载安装套件，内含JDK与IDE（Isabelle/jEdit）。
- 启动：打开IDE，新建理论文件 `.thy`，导入 `Main` 并输入命题后使用 `simp/auto`。

## 最小示例（Isar）

```isabelle
theory Add_Zero
  imports Main
begin

lemma add_0_right: "x + 0 = (x::nat)"
proof (induction x)
  case 0
  then show ?case by simp
next
  case (Suc x)
  then show ?case by simp
qed

end
```

提示：`simp`, `auto`, `sledgehammer` 可加速常见证明。

## 参考资料

- Nipkow, Paulson, Wenzel, Isabelle/HOL — A Proof Assistant for Higher-Order Logic
- Isabelle Reference Manual (isabelle.in.tum.de)
- Haddad & colleagues, Isabelle/MMT and mathematical knowledge management
