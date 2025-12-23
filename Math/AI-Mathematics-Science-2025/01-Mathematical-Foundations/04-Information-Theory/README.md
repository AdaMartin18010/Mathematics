# 信息论与编码理论 (Information Theory & Coding Theory)

> **From Entropy to Channel Capacity: The Mathematics of Information**
>
> 从熵到信道容量：信息的数学

---

## 目录

- [信息论与编码理论 (Information Theory & Coding Theory)](#信息论与编码理论-information-theory--coding-theory)
  - [目录](#目录)
  - [📋 模块概览](#-模块概览)
  - [📚 核心内容](#-核心内容)
    - [1. [熵与互信息](./01-Entropy-Mutual-Information.md) ✅](#1-熵与互信息01-entropy-mutual-informationmd-)
    - [2. [信息论应用](./02-Information-Theory-Applications.md) ✅](#2-信息论应用02-information-theory-applicationsmd-)
  - [🔗 与其他模块的联系](#-与其他模块的联系)
  - [📖 学习路径](#-学习路径)
  - [🎓 对标课程](#-对标课程)
  - [📊 模块完成度](#-模块完成度)

---

## 📋 模块概览

**信息论**研究信息的量化、传输和处理，是机器学习和深度学习的理论基础。

**核心问题**:

- 如何量化信息？
- 如何压缩数据？
- 如何可靠地传输信息？
- 信息论在AI中的应用？

---

## 📚 核心内容

### 1. [熵与互信息](./01-Entropy-Mutual-Information.md) ✅

**核心主题**:

- 熵 (Entropy)
- 互信息 (Mutual Information)
- KL散度 (Kullback-Leibler Divergence)
- 交叉熵 (Cross-Entropy)
- 条件熵 (Conditional Entropy)
- 信息增益 (Information Gain)

**关键公式**:

$$
H(X) = -\sum_{x} p(x) \log p(x)
$$

$$
I(X; Y) = H(X) - H(X \mid Y)
$$

$$
D_{KL}(P \| Q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

**AI应用**:

- 损失函数设计（交叉熵损失）
- 特征选择（信息增益）
- 正则化（KL散度）
- 变分推断（ELBO）

---

### 2. [信息论应用](./02-Information-Theory-Applications.md) ✅

**核心主题**:

- 信道容量理论
- 率失真理论
- 数据压缩
- 纠错编码
- 信息论在机器学习中的应用
  - 损失函数设计
  - VAE中的信息瓶颈
  - 模型压缩
  - GAN中的信息论视角

**关键定理**:

- 香农信道容量定理
- 率失真定理
- 数据压缩定理

**AI应用**:

- 变分自编码器 (VAE)
- 信息瓶颈理论
- 模型压缩与量化
- 生成对抗网络 (GAN)

---

## 🔗 与其他模块的联系

### 概率论

```text
概率分布 → 熵 → 信息量
条件概率 → 条件熵 → 互信息
```

### 机器学习

```text
熵 → 交叉熵损失
互信息 → 特征选择
KL散度 → 正则化、变分推断
```

### 优化理论

```text
信息论 → 损失函数设计
信息瓶颈 → 优化目标
```

---

## 📖 学习路径

### 阶段1: 基础概念 (1-2周)

1. **熵与信息量**
   - 理解熵的直观含义
   - 掌握熵的性质
   - 计算常见分布的熵

2. **互信息与KL散度**
   - 理解互信息的含义
   - 掌握KL散度的性质
   - 应用互信息进行特征选择

### 阶段2: 应用 (2-3周)

1. **信道容量理论**
   - 理解信道容量的概念
   - 掌握香农定理
   - 应用信道编码

2. **信息论在ML中的应用**
   - 损失函数设计
   - VAE中的信息论
   - 模型压缩

### 阶段3: 高级主题 (3-4周)

1. **率失真理论**
   - 理解率失真函数
   - 应用数据压缩

2. **信息瓶颈理论**
   - 理解信息瓶颈原理
   - 应用深度学习中

---

## 🎓 对标课程

| 大学 | 课程代码 | 课程名称 | 对应内容 |
 
        $matches[0] -replace '\|[-:]+\|', '| ---- |'
    
| **MIT** | 6.441 | Information Theory | 熵、信道容量、编码理论 |
| **Stanford** | EE376A | Information Theory | 熵、互信息、信道编码 |
| **UC Berkeley** | EECS 229A | Information Theory | 熵、信道容量、率失真 |
| **Cambridge** | Part III | Information Theory | 熵、编码理论、应用 |

---

## 📊 模块完成度

**当前完成度**: 约55-60%

**已完成文档**:

- ✅ 熵与互信息 (约80%完成)
- ✅ 信息论应用 (约75%完成)

**待完善内容**:

- [ ] 补充更多编码理论内容
- [ ] 补充更多率失真理论细节
- [ ] 补充更多应用实例
- [ ] 补充形式化证明

---

**最后更新**: 2025-12-20
**维护**: Mathematics项目团队
