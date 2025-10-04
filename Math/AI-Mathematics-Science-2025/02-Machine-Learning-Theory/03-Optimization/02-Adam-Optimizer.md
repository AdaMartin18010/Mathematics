# Adam优化器

> **Adaptive Moment Estimation (Adam)**
>
> 深度学习中最流行的优化算法

---

## 目录

- [Adam优化器](#adam优化器)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 算法推导](#-算法推导)
    - [1. 动量方法](#1-动量方法)
    - [2. RMSProp](#2-rmsprop)
    - [3. Adam算法](#3-adam算法)
  - [📊 理论分析](#-理论分析)
    - [1. 收敛性](#1-收敛性)
    - [2. 偏差修正](#2-偏差修正)
  - [🔧 变体与改进](#-变体与改进)
    - [1. AdamW](#1-adamw)
    - [2. AMSGrad](#2-amsgrad)
  - [💻 Python实现](#-python实现)
  - [📚 核心要点](#-核心要点)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**Adam** 结合了**动量**和**自适应学习率**的优点。

**核心特性**：

- 为每个参数自适应调整学习率
- 利用一阶和二阶矩估计
- 偏差修正

**优势**：

- 快速收敛
- 对超参数不敏感
- 适用于大规模问题

---

## 🎯 算法推导

### 1. 动量方法

**标准动量 (Momentum)**:

$$
\begin{align}
v_t &= \beta_1 v_{t-1} + (1 - \beta_1) g_t \\
\theta_t &= \theta_{t-1} - \alpha v_t
\end{align}
$$

**直觉**：累积历史梯度，加速收敛。

---

### 2. RMSProp

**Root Mean Square Propagation**:

$$
\begin{align}
s_t &= \beta_2 s_{t-1} + (1 - \beta_2) g_t^2 \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{s_t} + \epsilon} g_t
\end{align}
$$

**直觉**：为每个参数自适应调整学习率。

---

### 3. Adam算法

**算法 3.1 (Adam)**:

**输入**：

- 学习率 $\alpha$ (默认: 0.001)
- 一阶矩衰减率 $\beta_1$ (默认: 0.9)
- 二阶矩衰减率 $\beta_2$ (默认: 0.999)
- 数值稳定项 $\epsilon$ (默认: $10^{-8}$)

**初始化**：

- $m_0 = 0$ (一阶矩)
- $v_0 = 0$ (二阶矩)
- $t = 0$ (时间步)

**更新规则**：

$$
\begin{align}
t &\leftarrow t + 1 \\
g_t &= \nabla_\theta L(\theta_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad \text{(偏差修正)} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \quad \text{(偏差修正)} \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}
$$

---

## 📊 理论分析

### 1. 收敛性

**定理 1.1 (Adam收敛性)**:

在凸情况下，Adam的遗憾界为：

$$
R(T) = \sum_{t=1}^{T} [f(\theta_t) - f(\theta^*)] = O(\sqrt{T})
$$

**注意**：原始Adam在非凸情况下可能不收敛（见AMSGrad）。

---

### 2. 偏差修正

**为什么需要偏差修正？**

初始时 $m_0 = 0, v_0 = 0$，导致估计偏向零。

**修正后的期望**：

$$
\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t], \quad \mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2]
$$

**证明**：

$$
\mathbb{E}[m_t] = \mathbb{E}[g_t](1 - \beta_1^t)
$$

因此：

$$
\mathbb{E}[\hat{m}_t] = \frac{\mathbb{E}[m_t]}{1 - \beta_1^t} = \mathbb{E}[g_t]
$$

---

## 🔧 变体与改进

### 1. AdamW

**核心改进**：解耦权重衰减。

**标准Adam + L2正则**：

$$
\theta_t = \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
$$

**AdamW**：

$$
\theta_t = (1 - \alpha \lambda) \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**优势**：更好的泛化性能。

---

### 2. AMSGrad

**问题**：Adam可能不收敛（指数移动平均可能"忘记"历史信息）。

**解决方案**：保留历史最大二阶矩。

$$
\begin{align}
\hat{v}_t &= \max(\hat{v}_{t-1}, v_t) \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} m_t
\end{align}
$$

**保证**：非凸情况下的收敛性。

---

## 💻 Python实现

```python
import numpy as np

class AdamOptimizer:
    """Adam优化器实现"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步
    
    def update(self, params, grads):
        """
        更新参数
        
        Args:
            params: 参数字典 {name: value}
            grads: 梯度字典 {name: gradient}
        """
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        
        self.t += 1
        
        for key in params:
            # 更新一阶矩
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # 更新二阶矩
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            
            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # 更新参数
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params


class AdamWOptimizer:
    """AdamW优化器（解耦权重衰减）"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        """更新参数（AdamW）"""
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        
        self.t += 1
        
        for key in params:
            # 更新矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            
            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # AdamW更新（解耦权重衰减）
            params[key] = (1 - self.lr * self.weight_decay) * params[key] - \
                          self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params


# 示例：优化Rosenbrock函数
def rosenbrock(x):
    """Rosenbrock函数"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Rosenbrock梯度"""
    dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dx1 = 200 * (x[1] - x[0]**2)
    return np.array([dx0, dx1])

# 优化
params = {'x': np.array([-1.0, 1.0])}
optimizer = AdamOptimizer(lr=0.01)

for i in range(1000):
    grads = {'x': rosenbrock_grad(params['x'])}
    params = optimizer.update(params, grads)
    
    if i % 100 == 0:
        loss = rosenbrock(params['x'])
        print(f"Iteration {i}, Loss: {loss:.6f}, x: {params['x']}")
```

---

## 📚 核心要点

| 概念 | 说明 |
|------|------|
| **一阶矩** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ |
| **二阶矩** | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ |
| **偏差修正** | $\hat{m}_t = m_t / (1 - \beta_1^t)$ |
| **自适应学习率** | $\alpha / \sqrt{\hat{v}_t}$ |

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS231n Deep Learning |
| **MIT** | 6.036 Introduction to ML |
| **CMU** | 10-725 Convex Optimization |

---

## 📖 参考文献

1. **Kingma & Ba (2015)**. "Adam: A Method for Stochastic Optimization". *ICLR*.

2. **Loshchilov & Hutter (2019)**. "Decoupled Weight Decay Regularization". *ICLR*.

3. **Reddi et al. (2018)**. "On the Convergence of Adam and Beyond". *ICLR*.

---

*最后更新：2025年10月*-
