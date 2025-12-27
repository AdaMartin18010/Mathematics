# 熵与互信息

> **Entropy and Mutual Information**
>
> 信息论基础：量化不确定性与信息量

---

## 目录

- [熵与互信息](#熵与互信息)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 熵的定义](#-熵的定义)
    - [1. Shannon熵](#1-shannon熵)
    - [2. 联合熵与条件熵](#2-联合熵与条件熵)
    - [3. 相对熵 (KL散度)](#3-相对熵-kl散度)
  - [📊 互信息](#-互信息)
    - [1. 定义与性质](#1-定义与性质)
    - [2. 数据处理不等式](#2-数据处理不等式)
  - [🤖 AI应用](#-ai应用)
  - [💻 Python实现](#-python实现)
  - [📚 核心定理总结](#-核心定理总结)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**信息论**量化信息的不确定性和信息量。

**核心概念**：

- **熵 (Entropy)**：不确定性的度量
- **互信息 (Mutual Information)**：共享信息的度量
- **KL散度**：分布之间的"距离"

---

## 🎯 熵的定义

### 1. Shannon熵

**定义 1.1 (Shannon熵)**:

对于离散随机变量 $X \sim p(x)$：

$$
H(X) = -\sum_{x} p(x) \log p(x)
$$

**单位**：比特 (bits) 当使用 $\log_2$，奈特 (nats) 当使用 $\ln$。

**直觉**：平均编码长度的下界。

---

**性质**：

1. **非负性**：$H(X) \geq 0$
2. **最大熵**：均匀分布时熵最大
   $$
   H(X) \leq \log |\mathcal{X}|
   $$
3. **确定性**：$H(X) = 0$ 当且仅当 $X$ 确定

---

### 2. 联合熵与条件熵

**定义 2.1 (联合熵)**:

$$
H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)
$$

**定义 2.2 (条件熵)**:

$$
H(Y|X) = \sum_{x} p(x) H(Y|X=x) = -\sum_{x,y} p(x, y) \log p(y|x)
$$

**链式法则**：

$$
H(X, Y) = H(X) + H(Y|X)
$$

---

### 3. 相对熵 (KL散度)

**定义 3.1 (KL散度)**:

$$
D_{KL}(P \| Q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

**性质**：

1. **非负性**：$D_{KL}(P \| Q) \geq 0$，等号成立当且仅当 $P = Q$
2. **非对称性**：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

---

## 📊 互信息

### 1. 定义与性质

**定义 1.1 (互信息)**:

$$
I(X; Y) = \sum_{x,y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

**等价形式**：

$$
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
$$

$$
I(X; Y) = D_{KL}(p(x,y) \| p(x)p(y))
$$

**性质**：

1. **对称性**：$I(X; Y) = I(Y; X)$
2. **非负性**：$I(X; Y) \geq 0$
3. **独立性**：$I(X; Y) = 0$ 当且仅当 $X \perp Y$

---

### 2. 数据处理不等式

**定理 2.1 (数据处理不等式)**:

若 $X \to Y \to Z$ 构成马尔可夫链，则：

$$
I(X; Z) \leq I(X; Y)
$$

**意义**：信息处理不能增加信息量。

---

## 🤖 AI应用

**1. 机器学习**：

- 特征选择（最大化互信息）
- 模型压缩（最小化KL散度）

**2. 深度学习**：

- VAE损失函数（ELBO）
- 信息瓶颈理论

**3. 强化学习**：

- 最大熵RL
- 探索-利用平衡

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def shannon_entropy(p):
    """计算Shannon熵"""
    p = np.array(p)
    p = p[p > 0]  # 移除零概率
    return -np.sum(p * np.log2(p))

def kl_divergence(p, q):
    """计算KL散度"""
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log2(p / q))

def mutual_information(joint_prob):
    """计算互信息"""
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)

    mi = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(
                    joint_prob[i, j] / (p_x[i] * p_y[j])
                )
    return mi

# 示例
p = [0.5, 0.3, 0.2]
print(f"Entropy: {shannon_entropy(p):.4f} bits")

joint = np.array([[0.2, 0.1], [0.1, 0.6]])
print(f"Mutual Information: {mutual_information(joint):.4f} bits")
```

---

## 📚 核心定理总结

| 定理 | 陈述 | 意义 |
| ---- |------| ---- |
| **最大熵** | $H(X) \leq \log \|\mathcal{X}\|$ | 均匀分布最大 |
| **链式法则** | $H(X,Y) = H(X) + H(Y\|X)$ | 熵的分解 |
| **KL非负性** | $D_{KL}(P \| Q) \geq 0$ | 分布差异 |
| **数据处理** | $I(X;Z) \leq I(X;Y)$ | 信息不增 |

---

## 🎓 相关课程

| 大学 | 课程 |
| ---- |------|
| **MIT** | 6.441 Information Theory |
| **Stanford** | EE376A Information Theory |
| **Cambridge** | Information Theory |

---

## 📖 参考文献

1. **Cover & Thomas (2006)**. *Elements of Information Theory* (2nd ed.). Wiley.

2. **MacKay (2003)**. *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.

---

*最后更新：2025年10月*-
