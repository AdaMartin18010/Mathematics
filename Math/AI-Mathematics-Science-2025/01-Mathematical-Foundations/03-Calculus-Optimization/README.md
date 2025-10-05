# 微积分与优化理论 (Calculus & Optimization Theory)

> **The Mathematical Language of Deep Learning**
>
> 深度学习的数学语言

---

## 目录

- [微积分与优化理论 (Calculus \& Optimization Theory)](#微积分与优化理论-calculus--optimization-theory)
  - [目录](#目录)
  - [📋 模块概览](#-模块概览)
  - [📚 子模块结构](#-子模块结构)
    - [1. 多元微积分 (Multivariate Calculus) ✅](#1-多元微积分-multivariate-calculus-)
    - [2. 凸优化基础 (Convex Optimization Fundamentals) \[待补充\]](#2-凸优化基础-convex-optimization-fundamentals-待补充)
    - [3. 非凸优化 (Non-convex Optimization) \[待补充\]](#3-非凸优化-non-convex-optimization-待补充)
  - [💡 核心数学工具](#-核心数学工具)
    - [梯度与Hessian](#梯度与hessian)
    - [优化算法](#优化算法)
  - [🎓 对标世界顶尖大学课程](#-对标世界顶尖大学课程)
    - [MIT](#mit)
    - [Stanford](#stanford)
    - [UC Berkeley](#uc-berkeley)
    - [CMU](#cmu)
  - [📖 核心教材](#-核心教材)
  - [🔗 模块间联系](#-模块间联系)
  - [🛠️ 实践项目建议](#️-实践项目建议)
  - [📊 学习路径](#-学习路径)
    - [初级 (1-2个月)](#初级-1-2个月)
    - [中级 (2-3个月)](#中级-2-3个月)
    - [高级 (3个月以上)](#高级-3个月以上)

## 📋 模块概览

本模块系统介绍微积分与优化理论，这是深度学习的核心数学基础。
从多元微积分到凸优化，从梯度下降到约束优化，我们将建立完整的数学框架。

---

## 📚 子模块结构

### 1. 多元微积分 (Multivariate Calculus) ✅

**核心内容**:

- **偏导数与梯度**
  - 偏导数定义
  - 梯度向量
  - 方向导数
  
- **泰勒展开**
  - 一阶泰勒展开
  - 二阶泰勒展开
  - Hessian矩阵
  
- **链式法则**
  - 标量链式法则
  - 向量链式法则
  - 雅可比矩阵
  
- **梯度下降原理**
  - 最速下降方向
  - 收敛性分析
  - 步长选择
  
- **约束优化**
  - 拉格朗日乘数法
  - KKT条件

**AI应用**:

- 反向传播算法
- 损失函数优化
- 参数更新规则

**对标课程**:

- MIT 18.02 - Multivariable Calculus
- Stanford Math 51

---

### 2. 凸优化基础 (Convex Optimization Fundamentals) [待补充]

**核心内容**:

- **凸集与凸函数**
  - 凸集定义与性质
  - 凸函数定义与判定
  - 强凸性
  
- **凸优化问题**
  - 标准形式
  - 对偶理论
  - 最优性条件
  
- **凸优化算法**
  - 梯度投影法
  - 近端梯度法
  - ADMM

**AI应用**:

- SVM
- Lasso回归
- 神经网络训练

---

### 3. 非凸优化 (Non-convex Optimization) [待补充]

**核心内容**:

- **鞍点与局部极小值**
- **逃逸鞍点理论**
- **损失函数景观**
- **隐式偏差**

**AI应用**:

- 深度神经网络优化
- 过参数化理论

---

## 💡 核心数学工具

### 梯度与Hessian

```python
# 梯度
∇f(x) = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ

# Hessian矩阵
H(x) = [∂²f/∂xᵢ∂xⱼ]

# 泰勒展开
f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½ΔxᵀH(x)Δx
```

### 优化算法

```python
# 梯度下降
x_{t+1} = x_t - η∇f(x_t)

# 牛顿法
x_{t+1} = x_t - H⁻¹∇f(x_t)

# 约束优化 (拉格朗日)
L(x, λ) = f(x) + λg(x)
```

---

## 🎓 对标世界顶尖大学课程

### MIT

- **18.02** - Multivariable Calculus
- **6.255J** - Optimization Methods

### Stanford

- **Math 51** - Linear Algebra & Multivariable Calculus
- **EE364A** - Convex Optimization I
- **EE364B** - Convex Optimization II

### UC Berkeley

- **Math 53** - Multivariable Calculus
- **EECS 127** - Optimization Models in Engineering

### CMU

- **21-259** - Calculus in Three Dimensions
- **10-725** - Convex Optimization

---

## 📖 核心教材

1. **Stewart, J.** *Multivariable Calculus*. Cengage Learning.

2. **Boyd & Vandenberghe.** *Convex Optimization*. Cambridge University Press.

3. **Nocedal & Wright.** *Numerical Optimization*. Springer.

4. **Bertsekas, D.** *Convex Optimization Theory*. Athena Scientific.

---

## 🔗 模块间联系

```text
微积分与优化
    ↓
深度学习优化
    ├─ 梯度下降
    ├─ Adam
    └─ SGD变体
        ↓
    应用
    ├─ 反向传播
    ├─ 参数更新
    └─ 损失函数优化
```

---

## 🛠️ 实践项目建议

1. **实现梯度下降算法**：从零实现GD、Momentum、Adam
2. **可视化优化过程**：在2D函数上可视化不同算法的轨迹
3. **约束优化求解**：实现拉格朗日乘数法
4. **Hessian分析**：分析神经网络损失函数的曲率

---

## 📊 学习路径

### 初级 (1-2个月)

1. 多元微积分基础
2. 梯度与偏导数
3. 简单优化问题

### 中级 (2-3个月)

1. 泰勒展开与Hessian
2. 凸优化基础
3. 约束优化

### 高级 (3个月以上)

1. 非凸优化理论
2. 损失函数景观
3. 前沿优化算法

---

*最后更新：2025年10月*-
