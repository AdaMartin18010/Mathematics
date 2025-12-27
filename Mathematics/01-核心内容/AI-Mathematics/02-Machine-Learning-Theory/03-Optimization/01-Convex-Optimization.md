# 凸优化理论

> **Convex Optimization**
>
> 机器学习优化的理论基础：从梯度下降到加速算法

---

## 目录

- [凸优化理论](#凸优化理论)
  - [目录](#目录)
  - [📋 核心概念](#-核心概念)
  - [🎯 凸集与凸函数](#-凸集与凸函数)
    - [1. 凸集](#1-凸集)
    - [2. 凸函数](#2-凸函数)
    - [3. 强凸性](#3-强凸性)
  - [📊 最优性条件](#-最优性条件)
  - [🔧 优化算法](#-优化算法)
    - [1. 梯度下降](#1-梯度下降)
    - [2. 加速梯度法](#2-加速梯度法)
    - [3. 随机梯度下降 (SGD)](#3-随机梯度下降-sgd)
  - [🤖 AI应用](#-ai应用)
  - [💻 Python实现](#-python实现)
  - [📚 核心定理总结](#-核心定理总结)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心概念

**凸优化**研究形式为：

$$
\min_{x \in \mathcal{C}} f(x)
$$

其中 $f$ 为凸函数，$\mathcal{C}$ 为凸集。

**重要性**：

- 局部最优 = 全局最优
- 高效算法（多项式时间）
- 机器学习中许多问题是凸的（线性回归、SVM、Lasso等）

---

## 🎯 凸集与凸函数

### 1. 凸集

**定义**：集合 $\mathcal{C} \subseteq \mathbb{R}^d$ 是**凸集**，若对任意 $x, y \in \mathcal{C}$ 和 $\theta \in [0,1]$：

$$
\theta x + (1-\theta) y \in \mathcal{C}
$$

**示例**：

- ✅ 超平面：$\{x : a^\top x = b\}$
- ✅ 半空间：$\{x : a^\top x \leq b\}$
- ✅ 球：$\{x : \|x - x_0\| \leq r\}$
- ❌ 非凸：两个分离球的并集

---

### 2. 凸函数

**定义**：函数 $f : \mathcal{C} \to \mathbb{R}$ 是**凸函数**，若对任意 $x, y \in \mathcal{C}$ 和 $\theta \in [0,1]$：

$$
f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y)
$$

**一阶条件**（可微情况）：

$$
f(y) \geq f(x) + \nabla f(x)^\top (y - x)
$$

**二阶条件**（二阶可微）：

$$
\nabla^2 f(x) \succeq 0 \quad \text{(半正定)}
$$

---

### 3. 强凸性

**定义**：$f$ 是 **$\mu$-强凸**，若：

$$
f(y) \geq f(x) + \nabla f(x)^\top (y - x) + \frac{\mu}{2} \|y - x\|^2
$$

**意义**：更强的下界 → 更快的收敛

---

## 📊 最优性条件

**定理（一阶最优性条件）**：

设 $f$ 可微，$x^*$ 是无约束问题的最优解，当且仅当：

$$
\nabla f(x^*) = 0
$$

**带约束问题（KKT条件）**：

对于问题：
$$
\min_x f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \; h_j(x) = 0
$$

最优解 $x^*$ 满足：

1. $\nabla f(x^*) + \sum \lambda_i \nabla g_i(x^*) + \sum \nu_j \nabla h_j(x^*) = 0$
2. $\lambda_i g_i(x^*) = 0$ （互补松弛）
3. $\lambda_i \geq 0$

---

## 🔧 优化算法

### 1. 梯度下降

**算法**：

$$
x_{k+1} = x_k - \eta \nabla f(x_k)
$$

**收敛性（$L$-光滑凸函数）**：

$$
f(x_k) - f(x^*) \leq \frac{\|x_0 - x^*\|^2}{2\eta k}
$$

收敛率：$O(1/k)$

**强凸情况**：指数收敛 $O((1 - \mu/L)^k)$

---

### 2. 加速梯度法

**Nesterov动量**：

$$
\begin{align}
y_k &= x_k + \beta_k (x_k - x_{k-1}) \\
x_{k+1} &= y_k - \eta \nabla f(y_k)
\end{align}
$$

**收敛率**：$O(1/k^2)$ （最优！）

---

### 3. 随机梯度下降 (SGD)

**算法**：

$$
x_{k+1} = x_k - \eta_k \nabla f_i(x_k)
$$

其中 $i$ 随机采样。

**优势**：

- 低内存
- 适合大规模数据

**收敛率**：$O(1/\sqrt{k})$

---

## 🤖 AI应用

1. **线性回归**：$\min \|Ax - b\|^2$ （强凸）
2. **Logistic回归**：$\min \sum \log(1 + \exp(-y_i x_i^\top w))$
3. **SVM**：$\min \frac{1}{2}\|w\|^2 + C \sum \max(0, 1 - y_i(w^\top x_i))$
4. **神经网络训练**：虽然非凸，但凸优化理论提供直觉

---

## 💻 Python实现

```python
import numpy as np

def gradient_descent(f, grad_f, x0, eta=0.01, max_iter=1000, tol=1e-6):
    """梯度下降"""
    x = x0.copy()
    history = [x.copy()]
    
    for k in range(max_iter):
        g = grad_f(x)
        x_new = x - eta * g
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
        history.append(x.copy())
    
    return x, history

# 示例：最小化 f(x) = x^T Q x / 2
Q = np.array([[2, 0.5], [0.5, 1]])
f = lambda x: 0.5 * x @ Q @ x
grad_f = lambda x: Q @ x

x0 = np.array([5.0, 5.0])
x_opt, history = gradient_descent(f, grad_f, x0, eta=0.1, max_iter=100)

print(f"Optimal: {x_opt}")
```

---

## 📚 核心定理总结

| 算法 | 收敛率 | 条件 |
| ---- |--------| ---- |
| **梯度下降** | $O(1/k)$ | $L$-光滑凸 |
| **梯度下降** | $O(\exp(-k\mu/L))$ | $\mu$-强凸 |
| **Nesterov加速** | $O(1/k^2)$ | $L$-光滑凸 |
| **SGD** | $O(1/\sqrt{k})$ | 凸 + 无偏梯度 |

---

## 🎓 相关课程

| 大学 | 课程 |
| ---- |------|
| **Stanford** | EE364A Convex Optimization (Boyd) |
| **MIT** | 6.253 Convex Analysis and Optimization |
| **CMU** | 10-725 Convex Optimization |

---

## 📖 参考文献

1. **Boyd & Vandenberghe (2004)**. *Convex Optimization*. Cambridge University Press.
2. **Nesterov (2018)**. *Lectures on Convex Optimization* (2nd ed.). Springer.
3. **Bubeck (2015)**. "Convex Optimization: Algorithms and Complexity". *Foundations and Trends in ML*.

---

*最后更新：2025年10月*-
