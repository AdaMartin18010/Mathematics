# 反向传播算法

> **Backpropagation Algorithm**
>
> 深度学习的核心：高效计算梯度

---

## 目录

- [反向传播算法](#反向传播算法)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 算法推导](#-算法推导)
    - [1. 前向传播](#1-前向传播)
    - [2. 反向传播](#2-反向传播)
    - [3. 计算复杂度](#3-计算复杂度)
  - [📊 矩阵形式](#-矩阵形式)
  - [🔧 自动微分](#-自动微分)
  - [💻 Python实现](#-python实现)
  - [📚 核心要点](#-核心要点)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**反向传播 (Backpropagation)** 是高效计算神经网络梯度的算法。

**核心**：

- 利用**链式法则**递归计算梯度
- 时间复杂度与前向传播相同
- 使深度学习成为可能

---

## 🎯 算法推导

### 1. 前向传播

考虑 $L$ 层全连接网络：

$$
\begin{align}
z^{(\ell)} &= W^{(\ell)} a^{(\ell-1)} + b^{(\ell)} \\
a^{(\ell)} &= \sigma(z^{(\ell)})
\end{align}
$$

其中 $a^{(0)} = x$ 是输入。

---

### 2. 反向传播

**目标**：计算 $\frac{\partial L}{\partial W^{(\ell)}}$ 和 $\frac{\partial L}{\partial b^{(\ell)}}$。

**定义误差项**：

$$
\delta^{(\ell)} = \frac{\partial L}{\partial z^{(\ell)}}
$$

**递归公式**：

$$
\delta^{(\ell)} = (W^{(\ell+1)})^\top \delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})
$$

**梯度**：

$$
\frac{\partial L}{\partial W^{(\ell)}} = \delta^{(\ell)} (a^{(\ell-1)})^\top
$$

$$
\frac{\partial L}{\partial b^{(\ell)}} = \delta^{(\ell)}
$$

---

### 3. 计算复杂度

- **前向传播**：$O(W)$（$W$ 是参数数量）
- **反向传播**：$O(W)$

**关键**：只需前向传播的2倍时间！

---

## 📊 矩阵形式

**批量处理**：

输入批量 $X \in \mathbb{R}^{n \times d}$（$n$ 个样本）：

$$
Z^{(\ell)} = A^{(\ell-1)} (W^{(\ell)})^\top + \mathbf{1} (b^{(\ell)})^\top
$$

$$
\Delta^{(\ell)} = \Delta^{(\ell+1)} W^{(\ell+1)} \odot \sigma'(Z^{(\ell)})
$$

$$
\frac{\partial L}{\partial W^{(\ell)}} = (\Delta^{(\ell)})^\top A^{(\ell-1)}
$$

---

## 🔧 自动微分

**现代框架** (PyTorch, TensorFlow) 使用**自动微分**：

- **前向模式**：计算方向导数
- **反向模式**：反向传播的泛化

**计算图**：

```text
x → f₁ → y₁ → f₂ → y₂ → ... → L
```

**反向遍历**：从 $L$ 到 $x$ 计算所有梯度。

---

## 💻 Python实现

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.L = len(layer_sizes) - 1
        self.W = []
        self.b = []
        
        for i in range(self.L):
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01
            b = np.zeros((layer_sizes[i+1], 1))
            self.W.append(W)
            self.b.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """前向传播"""
        self.Z = []
        self.A = [X]
        
        for l in range(self.L):
            Z = self.W[l] @ self.A[l] + self.b[l]
            A = self.sigmoid(Z)
            self.Z.append(Z)
            self.A.append(A)
        
        return self.A[-1]
    
    def backward(self, X, Y):
        """反向传播"""
        m = X.shape[1]
        
        # 输出层误差
        dZ = self.A[-1] - Y
        
        # 存储梯度
        dW = []
        db = []
        
        # 反向遍历
        for l in reversed(range(self.L)):
            dW_l = (1/m) * dZ @ self.A[l].T
            db_l = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            dW.insert(0, dW_l)
            db.insert(0, db_l)
            
            if l > 0:
                dZ = (self.W[l].T @ dZ) * self.sigmoid_derivative(self.Z[l-1])
        
        return dW, db
    
    def train(self, X, Y, epochs=1000, lr=0.01):
        """训练"""
        for epoch in range(epochs):
            # 前向
            Y_pred = self.forward(X)
            
            # 反向
            dW, db = self.backward(X, Y)
            
            # 更新
            for l in range(self.L):
                self.W[l] -= lr * dW[l]
                self.b[l] -= lr * db[l]
            
            if epoch % 100 == 0:
                loss = np.mean((Y_pred - Y)**2)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 示例
X = np.random.randn(2, 100)
Y = (X[0] + X[1] > 0).astype(float).reshape(1, -1)

nn = NeuralNetwork([2, 4, 1])
nn.train(X, Y, epochs=1000, lr=0.5)
```

---

## 📚 核心要点

| 概念 | 说明 |
| ---- |------|
| **链式法则** | $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial w}$ |
| **误差项** | $\delta^{(\ell)} = \frac{\partial L}{\partial z^{(\ell)}}$ |
| **递归** | 从输出层到输入层传播误差 |
| **效率** | $O(W)$ 时间复杂度 |

---

## 🎓 相关课程

| 大学 | 课程 |
| ---- |------|
| **Stanford** | CS231n CNN for Visual Recognition |
| **MIT** | 6.036 Introduction to Machine Learning |
| **DeepLearning.AI** | Deep Learning Specialization |

---

## 📖 参考文献

1. **Rumelhart et al. (1986)**. "Learning Representations by Back-propagating Errors". *Nature*.

2. **Goodfellow et al. (2016)**. *Deep Learning*. MIT Press.

---

*最后更新：2025年10月*-
