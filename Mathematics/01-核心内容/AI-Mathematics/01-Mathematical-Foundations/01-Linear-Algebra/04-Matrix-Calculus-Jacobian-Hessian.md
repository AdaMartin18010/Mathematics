# 矩阵微分与Jacobian/Hessian (Matrix Calculus & Jacobian/Hessian)

> **The Mathematics of Backpropagation**
>
> 反向传播的数学基础

---

## 目录

- [矩阵微分与Jacobian/Hessian (Matrix Calculus \& Jacobian/Hessian)](#矩阵微分与jacobianhessian-matrix-calculus--jacobianhessian)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 标量对向量/矩阵的导数](#-标量对向量矩阵的导数)
    - [1. 标量对向量的导数](#1-标量对向量的导数)
    - [2. 标量对矩阵的导数](#2-标量对矩阵的导数)
    - [3. 常见导数公式](#3-常见导数公式)
  - [📊 向量对向量的导数 - Jacobian矩阵](#-向量对向量的导数---jacobian矩阵)
    - [1. Jacobian定义](#1-jacobian定义)
    - [2. Jacobian的性质](#2-jacobian的性质)
    - [3. 链式法则](#3-链式法则)
  - [🔬 二阶导数 - Hessian矩阵](#-二阶导数---hessian矩阵)
    - [1. Hessian定义](#1-hessian定义)
    - [2. Hessian的性质](#2-hessian的性质)
    - [3. 二阶Taylor展开](#3-二阶taylor展开)
  - [💡 矩阵微分技巧](#-矩阵微分技巧)
    - [1. 微分法则](#1-微分法则)
    - [2. 迹技巧](#2-迹技巧)
    - [3. Kronecker积](#3-kronecker积)
  - [🎨 在深度学习中的应用](#-在深度学习中的应用)
    - [1. 反向传播](#1-反向传播)
    - [2. 梯度下降](#2-梯度下降)
    - [3. 二阶优化方法](#3-二阶优化方法)
    - [4. 神经网络中的常见导数](#4-神经网络中的常见导数)
  - [🔧 高级主题](#-高级主题)
    - [1. 向量化技巧](#1-向量化技巧)
    - [2. 自动微分](#2-自动微分)
    - [3. 高阶导数](#3-高阶导数)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：梯度计算](#练习1梯度计算)
    - [练习2：Jacobian矩阵](#练习2jacobian矩阵)
    - [练习3：Hessian矩阵](#练习3hessian矩阵)
    - [练习4：反向传播](#练习4反向传播)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**矩阵微分**是处理多元函数导数的强大工具，是深度学习中反向传播算法的数学基础。**Jacobian矩阵**和**Hessian矩阵**分别表示一阶和二阶导数信息。

**为什么矩阵微分重要**:

```text
深度学习中的应用:
├─ 反向传播: 计算梯度
├─ 梯度下降: 参数更新
├─ 二阶优化: Newton法、拟Newton法
└─ 敏感性分析: 参数重要性

核心工具:
├─ 梯度 (Gradient): ∇f
├─ Jacobian矩阵: J
├─ Hessian矩阵: H
└─ 链式法则: 复合函数求导
```

---

## 🎯 标量对向量/矩阵的导数

### 1. 标量对向量的导数

**定义 1.1 (梯度)**:

设 $f: \mathbb{R}^n \to \mathbb{R}$，则 $f$ 对 $\mathbf{x}$ 的**梯度**定义为：

$$
\nabla f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^n
$$

**布局约定**:

- **分子布局 (Numerator layout)**: 梯度为列向量
- **分母布局 (Denominator layout)**: 梯度为行向量

本文档采用**分子布局**。

---

### 2. 标量对矩阵的导数

**定义 2.1 (矩阵导数)**:

设 $f: \mathbb{R}^{m \times n} \to \mathbb{R}$，则 $f$ 对矩阵 $X$ 的导数定义为：

$$
\frac{\partial f}{\partial X} = \begin{bmatrix}
\frac{\partial f}{\partial X_{11}} & \cdots & \frac{\partial f}{\partial X_{1n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial X_{m1}} & \cdots & \frac{\partial f}{\partial X_{mn}}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

---

### 3. 常见导数公式

**线性函数**:

$$
f(\mathbf{x}) = \mathbf{a}^T \mathbf{x} \quad \Rightarrow \quad \nabla f = \mathbf{a}
$$

**二次型**:

$$
f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} \quad \Rightarrow \quad \nabla f = (A + A^T) \mathbf{x}
$$

如果 $A$ 是对称的，则：

$$
\nabla f = 2A\mathbf{x}
$$

**范数**:

$$
f(\mathbf{x}) = \|\mathbf{x}\|^2 = \mathbf{x}^T \mathbf{x} \quad \Rightarrow \quad \nabla f = 2\mathbf{x}
$$

**矩阵迹**:

$$
f(X) = \text{tr}(AX) \quad \Rightarrow \quad \frac{\partial f}{\partial X} = A^T
$$

$$
f(X) = \text{tr}(X^T AX) \quad \Rightarrow \quad \frac{\partial f}{\partial X} = (A + A^T)X
$$

**行列式**:

$$
f(X) = \log \det(X) \quad \Rightarrow \quad \frac{\partial f}{\partial X} = X^{-T}
$$

---

## 📊 向量对向量的导数 - Jacobian矩阵

### 1. Jacobian定义

**定义 1.1 (Jacobian矩阵)**:

设 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$，$\mathbf{f}(\mathbf{x}) = [f_1(\mathbf{x}), \ldots, f_m(\mathbf{x})]^T$，则**Jacobian矩阵**定义为：

$$
J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

**第 $i$ 行**是 $f_i$ 的梯度的转置：

$$
J_{i,:} = (\nabla f_i)^T
$$

---

### 2. Jacobian的性质

**线性性**:

$$
\frac{\partial (A\mathbf{f} + B\mathbf{g})}{\partial \mathbf{x}} = A \frac{\partial \mathbf{f}}{\partial \mathbf{x}} + B \frac{\partial \mathbf{g}}{\partial \mathbf{x}}
$$

**乘积法则**:

$$
\frac{\partial (\mathbf{f}^T \mathbf{g})}{\partial \mathbf{x}} = \mathbf{f}^T \frac{\partial \mathbf{g}}{\partial \mathbf{x}} + \mathbf{g}^T \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
$$

---

### 3. 链式法则

**定理 3.1 (链式法则)**:

设 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$，$\mathbf{g}: \mathbb{R}^m \to \mathbb{R}^k$，则：

$$
\frac{\partial (\mathbf{g} \circ \mathbf{f})}{\partial \mathbf{x}} = \frac{\partial \mathbf{g}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
$$

其中 $\mathbf{y} = \mathbf{f}(\mathbf{x})$。

**矩阵形式**:

$$
J_{\mathbf{g} \circ \mathbf{f}} = J_{\mathbf{g}} \cdot J_{\mathbf{f}}
$$

**示例**:

设 $\mathbf{y} = A\mathbf{x}$，$\mathbf{z} = B\mathbf{y}$，则：

$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = B \cdot A = BA
$$

---

## 🔬 二阶导数 - Hessian矩阵

### 1. Hessian定义

**定义 1.1 (Hessian矩阵)**:

设 $f: \mathbb{R}^n \to \mathbb{R}$，则**Hessian矩阵**定义为：

$$
H = \nabla^2 f = \frac{\partial^2 f}{\partial \mathbf{x}^2} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix} \in \mathbb{R}^{n \times n}
$$

**元素表示**:

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

---

### 2. Hessian的性质

**对称性 (Schwarz定理)**:

如果 $f$ 是 $C^2$ 函数（二阶连续可微），则：

$$
H_{ij} = H_{ji} \quad \Rightarrow \quad H = H^T
$$

**正定性与凸性**:

- $H \succ 0$ (正定) $\Rightarrow$ $f$ 是**严格凸函数**
- $H \succeq 0$ (半正定) $\Rightarrow$ $f$ 是**凸函数**
- $H \prec 0$ (负定) $\Rightarrow$ $f$ 是**严格凹函数**

**临界点判定**:

设 $\nabla f(\mathbf{x}^*) = \mathbf{0}$，则：

- $H(\mathbf{x}^*) \succ 0$ $\Rightarrow$ $\mathbf{x}^*$ 是**局部极小值**
- $H(\mathbf{x}^*) \prec 0$ $\Rightarrow$ $\mathbf{x}^*$ 是**局部极大值**
- $H(\mathbf{x}^*)$ 不定 $\Rightarrow$ $\mathbf{x}^*$ 是**鞍点**

---

**Hessian矩阵性质的完整证明**:

**证明1：对称性（Schwarz定理）**:

**定理（Schwarz）**: 若 $f: \mathbb{R}^n \to \mathbb{R}$ 是 $C^2$ 函数（二阶偏导数连续），则混合偏导数与求导顺序无关：

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}
$$

**证明**：

考虑二元函数的情况（推广到多元是类似的）。设 $f(x, y)$ 在点 $(a, b)$ 附近二阶连续可微。

定义差商：

$$
\Delta(h, k) = f(a+h, b+k) - f(a+h, b) - f(a, b+k) + f(a, b)
$$

**方法1：先对 $x$ 后对 $y$**

固定 $y = b+k$，定义 $\phi(x) = f(x, b+k) - f(x, b)$，则：

$$
\Delta(h, k) = \phi(a+h) - \phi(a)
$$

由中值定理，存在 $\xi \in (a, a+h)$ 使得：

$$
\Delta(h, k) = h \phi'(\xi) = h \left[\frac{\partial f}{\partial x}(\xi, b+k) - \frac{\partial f}{\partial x}(\xi, b)\right]
$$

再次应用中值定理，存在 $\eta \in (b, b+k)$ 使得：

$$
\Delta(h, k) = hk \frac{\partial^2 f}{\partial y \partial x}(\xi, \eta)
$$

**方法2：先对 $y$ 后对 $x$**

类似地，可以得到：

$$
\Delta(h, k) = hk \frac{\partial^2 f}{\partial x \partial y}(\xi', \eta')
$$

其中 $\xi' \in (a, a+h)$，$\eta' \in (b, b+k)$。

当 $h, k \to 0$ 时，由于二阶偏导数连续：

$$
\frac{\partial^2 f}{\partial y \partial x}(\xi, \eta) \to \frac{\partial^2 f}{\partial y \partial x}(a, b)
$$

$$
\frac{\partial^2 f}{\partial x \partial y}(\xi', \eta') \to \frac{\partial^2 f}{\partial x \partial y}(a, b)
$$

因此：

$$
\frac{\partial^2 f}{\partial x \partial y}(a, b) = \frac{\partial^2 f}{\partial y \partial x}(a, b)
$$

这证明了 Hessian 矩阵的对称性：$H = H^T$。 $\square$

**证明2：正定性与凸性的关系**:

**定理**: 设 $f: \mathbb{R}^n \to \mathbb{R}$ 是 $C^2$ 函数。

(a) 若对所有 $\mathbf{x}$，$H(\mathbf{x}) \succeq 0$（半正定），则 $f$ 是凸函数。

(b) 若对所有 $\mathbf{x}$，$H(\mathbf{x}) \succ 0$（正定），则 $f$ 是严格凸函数。

**证明 (a)**：

需要证明对任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 和 $\lambda \in [0, 1]$：

$$
f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})
$$

定义 $\mathbf{z}(t) = \mathbf{y} + t(\mathbf{x} - \mathbf{y})$，$t \in [0, 1]$，以及：

$$
g(t) = f(\mathbf{z}(t))
$$

我们需要证明 $g$ 是凸函数，即 $g''(t) \geq 0$。

计算一阶导数：

$$
g'(t) = \nabla f(\mathbf{z}(t))^T \mathbf{z}'(t) = \nabla f(\mathbf{z}(t))^T (\mathbf{x} - \mathbf{y})
$$

计算二阶导数：

$$
g''(t) = (\mathbf{x} - \mathbf{y})^T H(\mathbf{z}(t)) (\mathbf{x} - \mathbf{y})
$$

由于 $H(\mathbf{z}(t)) \succeq 0$（半正定），对任意向量 $\mathbf{v} = \mathbf{x} - \mathbf{y}$：

$$
g''(t) = \mathbf{v}^T H(\mathbf{z}(t)) \mathbf{v} \geq 0
$$

因此 $g(t)$ 是凸函数，这意味着 $f$ 是凸函数。 $\square$

**证明 (b)**: 类似，但不等式是严格的。 $\square$

**证明3：临界点的二阶充分条件**:

**定理**: 设 $\nabla f(\mathbf{x}^*) = \mathbf{0}$（$\mathbf{x}^*$ 是临界点）。

(a) 若 $H(\mathbf{x}^*) \succ 0$（正定），则 $\mathbf{x}^*$ 是严格局部极小值点。

(b) 若 $H(\mathbf{x}^*) \prec 0$（负定），则 $\mathbf{x}^*$ 是严格局部极大值点。

(c) 若 $H(\mathbf{x}^*)$ 不定（有正有负特征值），则 $\mathbf{x}^*$ 是鞍点。

**证明 (a)**：

使用二阶 Taylor 展开，对于 $\mathbf{x}^*$ 附近的点 $\mathbf{x}^* + \mathbf{h}$（$\|\mathbf{h}\|$ 很小）：

$$
f(\mathbf{x}^* + \mathbf{h}) = f(\mathbf{x}^*) + \nabla f(\mathbf{x}^*)^T \mathbf{h} + \frac{1}{2} \mathbf{h}^T H(\mathbf{x}^*) \mathbf{h} + o(\|\mathbf{h}\|^2)
$$

由于 $\nabla f(\mathbf{x}^*) = \mathbf{0}$：

$$
f(\mathbf{x}^* + \mathbf{h}) = f(\mathbf{x}^*) + \frac{1}{2} \mathbf{h}^T H(\mathbf{x}^*) \mathbf{h} + o(\|\mathbf{h}\|^2)
$$

因为 $H(\mathbf{x}^*) \succ 0$（正定），存在 $\lambda_{\min} > 0$ 使得：

$$
\mathbf{h}^T H(\mathbf{x}^*) \mathbf{h} \geq \lambda_{\min} \|\mathbf{h}\|^2
$$

因此：

$$
f(\mathbf{x}^* + \mathbf{h}) - f(\mathbf{x}^*) \geq \frac{\lambda_{\min}}{2} \|\mathbf{h}\|^2 + o(\|\mathbf{h}\|^2)
$$

当 $\|\mathbf{h}\|$ 足够小时，右边为正，因此：

$$
f(\mathbf{x}^* + \mathbf{h}) > f(\mathbf{x}^*)
$$

这证明了 $\mathbf{x}^*$ 是严格局部极小值点。 $\square$

**证明 (b)**: 类似，但不等号反向。 $\square$

**证明 (c)**：

若 $H(\mathbf{x}^*)$ 不定，则存在特征值 $\lambda_1 > 0$ 和 $\lambda_2 < 0$，对应的特征向量为 $\mathbf{v}_1$ 和 $\mathbf{v}_2$。

沿 $\mathbf{v}_1$ 方向（取 $\mathbf{h} = t\mathbf{v}_1$，$t$ 很小）：

$$
f(\mathbf{x}^* + t\mathbf{v}_1) - f(\mathbf{x}^*) \approx \frac{t^2}{2} \mathbf{v}_1^T H(\mathbf{x}^*) \mathbf{v}_1 = \frac{t^2}{2} \lambda_1 > 0
$$

沿 $\mathbf{v}_2$ 方向（取 $\mathbf{h} = t\mathbf{v}_2$）：

$$
f(\mathbf{x}^* + t\mathbf{v}_2) - f(\mathbf{x}^*) \approx \frac{t^2}{2} \lambda_2 < 0
$$

因此 $\mathbf{x}^*$ 既不是局部极小值也不是局部极大值，而是鞍点。 $\square$

**实际应用示例**：

考虑函数 $f(x, y) = x^2 - y^2$（马鞍面）。

梯度：

$$
\nabla f = \begin{bmatrix} 2x \\ -2y \end{bmatrix}
$$

临界点：$(0, 0)$，因为 $\nabla f(0, 0) = \mathbf{0}$。

Hessian矩阵：

$$
H = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}
$$

特征值：$\lambda_1 = 2 > 0$，$\lambda_2 = -2 < 0$

由于 $H$ 不定（有正有负特征值），$(0, 0)$ 是鞍点。

---

### 3. 二阶Taylor展开

**定理 3.1 (二阶Taylor展开)**:

$$
f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \mathbf{h} + \frac{1}{2} \mathbf{h}^T H(\mathbf{x}) \mathbf{h}
$$

**应用**:

- 函数逼近
- Newton法
- 信赖域方法

---

## 💡 矩阵微分技巧

### 1. 微分法则

**基本微分**:

$$
d(\mathbf{x}^T \mathbf{a}) = \mathbf{a}^T d\mathbf{x}
$$

$$
d(\mathbf{x}^T A \mathbf{x}) = \mathbf{x}^T (A + A^T) d\mathbf{x}
$$

$$
d(\text{tr}(AX)) = \text{tr}(A \, dX)
$$

**乘积法则**:

$$
d(XY) = (dX)Y + X(dY)
$$

**逆矩阵**:

$$
d(X^{-1}) = -X^{-1} (dX) X^{-1}
$$

**行列式**:

$$
d(\det(X)) = \det(X) \cdot \text{tr}(X^{-1} dX)
$$

---

### 2. 迹技巧

**技巧 2.1 (迹的循环性)**:

$$
\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)
$$

**应用**:

将标量表示为迹，便于求导：

$$
\mathbf{a}^T \mathbf{b} = \text{tr}(\mathbf{a}^T \mathbf{b}) = \text{tr}(\mathbf{b} \mathbf{a}^T)
$$

**示例**:

$$
\frac{\partial (\mathbf{a}^T X \mathbf{b})}{\partial X} = \frac{\partial \text{tr}(\mathbf{a}^T X \mathbf{b})}{\partial X} = \frac{\partial \text{tr}(\mathbf{b} \mathbf{a}^T X)}{\partial X} = \mathbf{a} \mathbf{b}^T
$$

---

### 3. Kronecker积

**定义 3.1 (Kronecker积)**:

$$
A \otimes B = \begin{bmatrix}
a_{11}B & \cdots & a_{1n}B \\
\vdots & \ddots & \vdots \\
a_{m1}B & \cdots & a_{mn}B
\end{bmatrix}
$$

**向量化**:

$$
\text{vec}(AXB) = (B^T \otimes A) \text{vec}(X)
$$

**应用**:

简化矩阵对矩阵的导数计算。

---

## 🎨 在深度学习中的应用

### 1. 反向传播

**前向传播**:

$$
\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$

$$
\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})
$$

**反向传播**:

$$
\delta^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}
$$

**链式法则**:

$$
\delta^{(l-1)} = (W^{(l)})^T \delta^{(l)} \odot \sigma'(\mathbf{z}^{(l-1)})
$$

**梯度**:

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T
$$

$$
\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}
$$

---

### 2. 梯度下降

**参数更新**:

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

**批量梯度下降**:

$$
\nabla L(\theta) = \frac{1}{n} \sum_{i=1}^n \nabla L_i(\theta)
$$

**随机梯度下降**:

$$
\theta_{t+1} = \theta_t - \eta \nabla L_i(\theta_t)
$$

---

### 3. 二阶优化方法

**Newton法**:

$$
\theta_{t+1} = \theta_t - H^{-1} \nabla L(\theta_t)
$$

**拟Newton法 (BFGS)**:

使用近似Hessian $B_t$：

$$
\theta_{t+1} = \theta_t - \eta B_t^{-1} \nabla L(\theta_t)
$$

**Gauss-Newton法**:

对于最小二乘问题 $L = \frac{1}{2} \|\mathbf{r}(\theta)\|^2$：

$$
H \approx J^T J
$$

其中 $J$ 是残差的Jacobian。

---

### 4. 神经网络中的常见导数

**Sigmoid激活**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}} \quad \Rightarrow \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

**Tanh激活**:

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \quad \Rightarrow \quad \tanh'(z) = 1 - \tanh^2(z)
$$

**ReLU激活**:

$$
\text{ReLU}(z) = \max(0, z) \quad \Rightarrow \quad \text{ReLU}'(z) = \begin{cases}
1 & z > 0 \\
0 & z \leq 0
\end{cases}
$$

**Softmax**:

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

$$
\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i (\delta_{ij} - \text{softmax}(\mathbf{z})_j)
$$

**交叉熵损失**:

$$
L = -\sum_i y_i \log \hat{y}_i
$$

$$
\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}
$$

（当使用softmax激活时）

---

## 🔧 高级主题

### 1. 向量化技巧

**批量处理**:

将批量数据组织为矩阵，使用矩阵运算代替循环：

$$
Y = XW + \mathbf{b} \mathbf{1}^T
$$

其中 $X \in \mathbb{R}^{B \times n}$，$W \in \mathbb{R}^{n \times m}$。

**梯度**:

$$
\frac{\partial L}{\partial W} = X^T \delta
$$

---

### 2. 自动微分

**前向模式 (Forward Mode)**:

计算 $\frac{\partial \mathbf{y}}{\partial x_i}$，适合输入维度小的情况。

**反向模式 (Reverse Mode)**:

计算 $\frac{\partial y_i}{\partial \mathbf{x}}$，适合输出维度小的情况（深度学习常用）。

**计算图**:

```text
x → f₁ → y₁ → f₂ → y₂ → ... → L
```

反向传播沿计算图反向计算梯度。

---

### 3. 高阶导数

**三阶张量**:

$$
\frac{\partial^3 f}{\partial x_i \partial x_j \partial x_k}
$$

**应用**:

- 高阶优化方法
- 敏感性分析
- 不确定性量化

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 梯度计算示例
def gradient_examples():
    """梯度计算示例"""
    print("=== 梯度计算 ===\n")
    
    # 线性函数: f(x) = a^T x
    a = np.array([1, 2, 3])
    x = np.array([4, 5, 6])
    
    f = np.dot(a, x)
    grad_f = a  # 梯度就是 a
    
    print(f"线性函数 f(x) = a^T x:")
    print(f"  f = {f}")
    print(f"  ∇f = {grad_f}\n")
    
    # 二次型: f(x) = x^T A x
    A = np.array([[2, 1], [1, 3]])
    x = np.array([1, 2])
    
    f = x @ A @ x
    grad_f = (A + A.T) @ x  # 梯度 = (A + A^T)x
    
    print(f"二次型 f(x) = x^T A x:")
    print(f"  f = {f}")
    print(f"  ∇f = {grad_f}\n")
    
    # 范数: f(x) = ||x||^2
    x = np.array([3, 4])
    f = np.dot(x, x)
    grad_f = 2 * x
    
    print(f"范数平方 f(x) = ||x||^2:")
    print(f"  f = {f}")
    print(f"  ∇f = {grad_f}\n")


# 2. Jacobian矩阵计算
def jacobian_example():
    """Jacobian矩阵计算"""
    print("=== Jacobian矩阵 ===\n")
    
    # 定义函数 f: R^2 -> R^3
    def f(x):
        return np.array([
            x[0]**2 + x[1],
            x[0] * x[1],
            x[0] + x[1]**2
        ])
    
    # 解析Jacobian
    def jacobian_f(x):
        return np.array([
            [2*x[0], 1],
            [x[1], x[0]],
            [1, 2*x[1]]
        ])
    
    # 数值Jacobian (有限差分)
    def numerical_jacobian(f, x, eps=1e-7):
        n = len(x)
        m = len(f(x))
        J = np.zeros((m, n))
        
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            J[:, i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        
        return J
    
    # 测试点
    x = np.array([1.0, 2.0])
    
    J_analytical = jacobian_f(x)
    J_numerical = numerical_jacobian(f, x)
    
    print(f"测试点 x = {x}")
    print(f"\n解析Jacobian:\n{J_analytical}")
    print(f"\n数值Jacobian:\n{J_numerical}")
    print(f"\n误差: {np.max(np.abs(J_analytical - J_numerical)):.2e}\n")


# 3. Hessian矩阵计算
def hessian_example():
    """Hessian矩阵计算"""
    print("=== Hessian矩阵 ===\n")
    
    # 定义函数 f: R^2 -> R
    def f(x):
        return x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2
    
    # 解析Hessian
    def hessian_f(x):
        return np.array([
            [2, 2],
            [2, 6]
        ])
    
    # 数值Hessian
    def numerical_hessian(f, x, eps=1e-5):
        n = len(x)
        H = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                
                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps
                
                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps
                
                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
        
        return H
    
    x = np.array([1.0, 1.0])
    
    H_analytical = hessian_f(x)
    H_numerical = numerical_hessian(f, x)
    
    print(f"测试点 x = {x}")
    print(f"\n解析Hessian:\n{H_analytical}")
    print(f"\n数值Hessian:\n{H_numerical}")
    print(f"\n对称性检查: {np.allclose(H_analytical, H_analytical.T)}")
    
    # 正定性检查
    eigenvalues = np.linalg.eigvalsh(H_analytical)
    print(f"\n特征值: {eigenvalues}")
    if np.all(eigenvalues > 0):
        print("Hessian是正定的 → 函数是严格凸的\n")


# 4. 反向传播示例
class SimpleNN:
    """简单神经网络"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 初始化权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层梯度
        dz2 = output - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        
        # 隐藏层梯度
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        
        return dW1, db1, dW2, db2
    
    def train(self, X, y, epochs=1000, lr=0.1):
        """训练"""
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失
            loss = np.mean((output - y)**2)
            losses.append(loss)
            
            # 反向传播
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            
            # 更新参数
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses


def backpropagation_demo():
    """反向传播演示"""
    print("=== 反向传播 ===\n")
    
    # 生成XOR数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # 创建并训练网络
    nn = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)
    losses = nn.train(X, y, epochs=5000, lr=1.0)
    
    # 测试
    print("\n测试结果:")
    predictions = nn.forward(X)
    for i in range(len(X)):
        print(f"  输入: {X[i]}, 预测: {predictions[i][0]:.4f}, 真实: {y[i][0]}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    # plt.show()


# 5. 梯度下降可视化
def gradient_descent_visualization():
    """梯度下降可视化"""
    # 定义函数: f(x, y) = x^2 + 2y^2
    def f(x, y):
        return x**2 + 2*y**2
    
    def grad_f(x, y):
        return np.array([2*x, 4*y])
    
    # 梯度下降
    x, y = 3.0, 2.0
    lr = 0.1
    trajectory = [(x, y)]
    
    for _ in range(20):
        grad = grad_f(x, y)
        x -= lr * grad[0]
        y -= lr * grad[1]
        trajectory.append((x, y))
    
    trajectory = np.array(trajectory)
    
    # 绘制
    fig = plt.figure(figsize=(12, 5))
    
    # 2D等高线图
    ax1 = fig.add_subplot(121)
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)
    
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2, markersize=6)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Gradient Descent (2D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D曲面图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 
             [f(x, y) for x, y in trajectory], 
             'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x, y)')
    ax2.set_title('Gradient Descent (3D)')
    
    plt.tight_layout()
    # plt.show()


# 6. Newton法 vs 梯度下降
def newton_vs_gradient_descent():
    """Newton法 vs 梯度下降"""
    print("=== Newton法 vs 梯度下降 ===\n")
    
    # 定义函数
    def f(x):
        return x[0]**2 + 2*x[1]**2
    
    def grad_f(x):
        return np.array([2*x[0], 4*x[1]])
    
    def hessian_f(x):
        return np.array([[2, 0], [0, 4]])
    
    # 梯度下降
    x_gd = np.array([3.0, 2.0])
    trajectory_gd = [x_gd.copy()]
    lr = 0.1
    
    for _ in range(20):
        x_gd -= lr * grad_f(x_gd)
        trajectory_gd.append(x_gd.copy())
    
    # Newton法
    x_newton = np.array([3.0, 2.0])
    trajectory_newton = [x_newton.copy()]
    
    for _ in range(5):
        grad = grad_f(x_newton)
        H = hessian_f(x_newton)
        x_newton -= np.linalg.solve(H, grad)
        trajectory_newton.append(x_newton.copy())
    
    print(f"梯度下降 (20步): 终点 = {trajectory_gd[-1]}")
    print(f"Newton法 (5步): 终点 = {trajectory_newton[-1]}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("矩阵微分与Jacobian/Hessian示例")
    print("=" * 60 + "\n")
    
    gradient_examples()
    jacobian_example()
    hessian_example()
    backpropagation_demo()
    
    print("\n可视化...")
    gradient_descent_visualization()
    newton_vs_gradient_descent()
    
    print("\n所有示例完成！")
```

---

## 📚 练习题

### 练习1：梯度计算

计算以下函数的梯度：

1. $f(\mathbf{x}) = \mathbf{a}^T \mathbf{x} + b$
2. $f(\mathbf{x}) = \|\mathbf{x} - \mathbf{a}\|^2$
3. $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x} + c$

### 练习2：Jacobian矩阵

计算以下函数的Jacobian矩阵：

1. $\mathbf{f}(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$
2. $\mathbf{f}(\mathbf{x}) = \text{softmax}(\mathbf{x})$

### 练习3：Hessian矩阵

计算以下函数的Hessian矩阵，并判断凸性：

1. $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$（$A$ 对称）
2. $f(\mathbf{x}) = \log(\sum_i e^{x_i})$

### 练习4：反向传播

手动推导两层神经网络的反向传播公式。

---

## 🎓 相关课程

| 大学 | 课程 |
| ---- |------|
| **MIT** | 18.065 - Matrix Methods in Data Analysis |
| **Stanford** | CS229 - Machine Learning |
| **CMU** | 10-701 - Machine Learning |
| **UC Berkeley** | CS189 - Introduction to Machine Learning |

---

## 📖 参考文献

1. **Petersen & Pedersen (2012)**. *The Matrix Cookbook*. Technical University of Denmark.

2. **Goodfellow et al. (2016)**. *Deep Learning*. MIT Press. (Chapter 4)

3. **Boyd & Vandenberghe (2004)**. *Convex Optimization*. Cambridge University Press.

4. **Magnus & Neudecker (2019)**. *Matrix Differential Calculus with Applications in Statistics and Econometrics*. Wiley.

5. **Griewank & Walther (2008)**. *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM.

---

*最后更新：2025年10月*-
