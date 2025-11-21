# Hilbert空间与再生核Hilbert空间 (Hilbert Spaces & RKHS)

> **The Mathematical Foundation of Kernel Methods**
>
> 核方法的数学基础

---

## 目录

- [Hilbert空间与再生核Hilbert空间 (Hilbert Spaces \& RKHS)](#hilbert空间与再生核hilbert空间-hilbert-spaces--rkhs)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 Hilbert空间](#-hilbert空间)
    - [1. 内积空间](#1-内积空间)
    - [2. Hilbert空间定义](#2-hilbert空间定义)
    - [3. 正交性与投影](#3-正交性与投影)
  - [📊 再生核Hilbert空间 (RKHS)](#-再生核hilbert空间-rkhs)
    - [1. RKHS定义](#1-rkhs定义)
    - [2. 再生核](#2-再生核)
    - [3. Moore-Aronszajn定理](#3-moore-aronszajn定理)
  - [🔬 核函数](#-核函数)
    - [1. 核函数定义](#1-核函数定义)
    - [2. 常见核函数](#2-常见核函数)
    - [3. 核的性质](#3-核的性质)
  - [💡 Representer定理](#-representer定理)
    - [1. 定理陈述](#1-定理陈述)
    - [2. 证明思路](#2-证明思路)
    - [3. 应用](#3-应用)
  - [🎨 核技巧 (Kernel Trick)](#-核技巧-kernel-trick)
    - [1. 核技巧思想](#1-核技巧思想)
    - [2. 特征映射](#2-特征映射)
    - [3. 核矩阵](#3-核矩阵)
  - [🔧 在机器学习中的应用](#-在机器学习中的应用)
    - [1. 支持向量机 (SVM)](#1-支持向量机-svm)
    - [2. 核岭回归](#2-核岭回归)
    - [3. 高斯过程](#3-高斯过程)
    - [4. 核PCA](#4-核pca)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：内积计算](#练习1内积计算)
    - [练习2：核函数验证](#练习2核函数验证)
    - [练习3：Representer定理](#练习3representer定理)
    - [练习4：核SVM](#练习4核svm)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**Hilbert空间**是完备的内积空间，是泛函分析的核心概念。
**再生核Hilbert空间 (RKHS)** 是一类特殊的Hilbert空间，在机器学习中有广泛应用。

**为什么RKHS重要**:

```text
机器学习中的核方法:
├─ 支持向量机 (SVM)
├─ 核岭回归
├─ 高斯过程
└─ 核PCA

核心优势:
├─ 处理非线性问题
├─ 无需显式特征映射
├─ 理论保证 (Representer定理)
└─ 计算高效 (核技巧)
```

---

## 🎯 Hilbert空间

### 1. 内积空间

**定义 1.1 (内积空间)**:

向量空间 $V$ 配备内积 $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ (或 $\mathbb{C}$)，满足：

1. **正定性**: $\langle x, x \rangle \geq 0$，且 $\langle x, x \rangle = 0 \Leftrightarrow x = 0$
2. **线性性**: $\langle ax + by, z \rangle = a\langle x, z \rangle + b\langle y, z \rangle$
3. **对称性**: $\langle x, y \rangle = \langle y, x \rangle$ (实数情况)

**范数**:

$$
\|x\| = \sqrt{\langle x, x \rangle}
$$

**示例**:

- $\mathbb{R}^n$ 配备标准内积：$\langle x, y \rangle = \sum_{i=1}^n x_i y_i$
- $L^2[a, b]$ 配备：$\langle f, g \rangle = \int_a^b f(x)g(x) \, dx$

---

### 2. Hilbert空间定义

**定义 2.1 (Hilbert空间)**:

**完备的内积空间**称为Hilbert空间。即，所有Cauchy序列都收敛。

**示例**:

- $\mathbb{R}^n$ 是有限维Hilbert空间
- $L^2(\mathbb{R})$ 是无限维Hilbert空间
- $\ell^2 = \{(x_1, x_2, \ldots) : \sum_{i=1}^\infty x_i^2 < \infty\}$ 是序列空间

**性质**:

- 有限维内积空间都是Hilbert空间
- Hilbert空间是Banach空间（完备的赋范空间）

---

### 3. 正交性与投影

**定义 3.1 (正交)**:

$x \perp y$ 如果 $\langle x, y \rangle = 0$。

**投影定理**:

设 $M$ 是Hilbert空间 $\mathcal{H}$ 的闭子空间，则对于任意 $x \in \mathcal{H}$，存在唯一的 $y \in M$ 使得：

$$
\|x - y\| = \inf_{z \in M} \|x - z\|
$$

$y$ 称为 $x$ 在 $M$ 上的**正交投影**。

**正交分解**:

$$
\mathcal{H} = M \oplus M^\perp
$$

其中 $M^\perp = \{x \in \mathcal{H} : \langle x, y \rangle = 0, \forall y \in M\}$。

---

## 📊 再生核Hilbert空间 (RKHS)

### 1. RKHS定义

**定义 1.1 (RKHS)**:

设 $\mathcal{H}$ 是定义在集合 $\mathcal{X}$ 上的函数空间。如果对于所有 $x \in \mathcal{X}$，**点评估泛函** $\delta_x: \mathcal{H} \to \mathbb{R}$ 定义为：

$$
\delta_x(f) = f(x)
$$

是**连续的**（有界的），则 $\mathcal{H}$ 称为**再生核Hilbert空间**。

**直觉**：在RKHS中，函数值可以通过内积"再生"。

---

### 2. 再生核

**定理 2.1 (Riesz表示定理)**:

对于RKHS $\mathcal{H}$，对于每个 $x \in \mathcal{X}$，存在唯一的 $k_x \in \mathcal{H}$ 使得：

$$
f(x) = \langle f, k_x \rangle_{\mathcal{H}}, \quad \forall f \in \mathcal{H}
$$

**定义 2.2 (再生核)**:

函数 $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 定义为：

$$
k(x, y) = \langle k_x, k_y \rangle_{\mathcal{H}}
$$

称为 $\mathcal{H}$ 的**再生核**。

**再生性质**:

1. $f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}}$
2. $k(x, y) = \langle k(\cdot, x), k(\cdot, y) \rangle_{\mathcal{H}}$

---

### 3. Moore-Aronszajn定理

**定理 3.1 (Moore-Aronszajn定理)**:

设 $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 是**正定核**（即对于任意 $n$ 和 $x_1, \ldots, x_n \in \mathcal{X}$，矩阵 $K_{ij} = k(x_i, x_j)$ 是半正定的），则存在唯一的RKHS $\mathcal{H}_k$，其再生核为 $k$。

**意义**：正定核与RKHS一一对应。

---

## 🔬 核函数

### 1. 核函数定义

**定义 1.1 (核函数)**:

函数 $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 是**核函数**，如果存在特征映射 $\phi: \mathcal{X} \to \mathcal{H}$ 使得：

$$
k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}
$$

**等价条件** (Mercer定理):

$k$ 是核函数当且仅当 $k$ 是**对称的**且**正定的**。

---

### 2. 常见核函数

**线性核**:

$$
k(x, y) = x^T y
$$

**多项式核**:

$$
k(x, y) = (x^T y + c)^d
$$

**高斯核 (RBF核)**:

$$
k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)
$$

**Laplacian核**:

$$
k(x, y) = \exp\left(-\frac{\|x - y\|}{\sigma}\right)
$$

**Sigmoid核**:

$$
k(x, y) = \tanh(\alpha x^T y + c)
$$

---

### 3. 核的性质

**性质 3.1 (核的封闭性)**:

1. **加法**: 如果 $k_1, k_2$ 是核，则 $k_1 + k_2$ 是核
2. **数乘**: 如果 $k$ 是核，$c > 0$，则 $ck$ 是核
3. **乘法**: 如果 $k_1, k_2$ 是核，则 $k_1 \cdot k_2$ 是核
4. **复合**: 如果 $k$ 是核，$f$ 是正函数，则 $f(k)$ 可能是核

**示例**:

高斯核可以表示为：

$$
k(x, y) = \exp\left(-\frac{\|x\|^2 + \|y\|^2}{2\sigma^2}\right) \exp\left(\frac{x^T y}{\sigma^2}\right)
$$

---

## 💡 Representer定理

### 1. 定理陈述

**定理 1.1 (Representer定理)**:

设 $\mathcal{H}_k$ 是RKHS，给定训练数据 $\{(x_i, y_i)\}_{i=1}^n$，考虑优化问题：

$$
\min_{f \in \mathcal{H}_k} \sum_{i=1}^n L(y_i, f(x_i)) + \lambda \|f\|_{\mathcal{H}_k}^2
$$

其中 $L$ 是损失函数，$\lambda > 0$ 是正则化参数。

则最优解 $f^*$ 可以表示为：

$$
f^*(x) = \sum_{i=1}^n \alpha_i k(x, x_i)
$$

**意义**：最优解位于由训练数据张成的有限维子空间中。

---

### 2. 证明思路

**证明**:

将 $f$ 分解为：

$$
f = f_{\parallel} + f_{\perp}
$$

其中 $f_{\parallel} \in \text{span}\{k(\cdot, x_i)\}_{i=1}^n$，$f_{\perp} \perp f_{\parallel}$。

由再生性质：

$$
f(x_i) = \langle f, k(\cdot, x_i) \rangle = \langle f_{\parallel}, k(\cdot, x_i) \rangle
$$

因此损失项只依赖于 $f_{\parallel}$，而 $\|f\|^2 = \|f_{\parallel}\|^2 + \|f_{\perp}\|^2$。

所以 $f_{\perp} = 0$ 时目标函数最小。

---

### 3. 应用

**核岭回归**:

$$
\min_{f} \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \|f\|^2
$$

解为：

$$
f^*(x) = \sum_{i=1}^n \alpha_i k(x, x_i)
$$

其中 $\alpha = (K + \lambda I)^{-1} y$，$K_{ij} = k(x_i, x_j)$。

---

## 🎨 核技巧 (Kernel Trick)

### 1. 核技巧思想

**核心思想**：

不需要显式计算特征映射 $\phi(x)$，只需要计算核函数 $k(x, y) = \langle \phi(x), \phi(y) \rangle$。

**优势**:

- 避免高维（甚至无限维）特征空间的显式计算
- 计算复杂度只依赖于数据量，而非特征维度

---

### 2. 特征映射

**示例** (多项式核):

对于 $x, y \in \mathbb{R}^2$，考虑 $k(x, y) = (x^T y)^2$。

显式特征映射：

$$
\phi(x) = (x_1^2, \sqrt{2}x_1 x_2, x_2^2)
$$

则：

$$
\langle \phi(x), \phi(y) \rangle = x_1^2 y_1^2 + 2x_1 x_2 y_1 y_2 + x_2^2 y_2^2 = (x^T y)^2
$$

**高斯核**：对应无限维特征空间！

---

### 3. 核矩阵

**定义 3.1 (Gram矩阵/核矩阵)**:

给定数据 $\{x_1, \ldots, x_n\}$，核矩阵 $K \in \mathbb{R}^{n \times n}$ 定义为：

$$
K_{ij} = k(x_i, x_j)
$$

**性质**:

- $K$ 是对称的
- $K$ 是半正定的（如果 $k$ 是正定核）

---

## 🔧 在机器学习中的应用

### 1. 支持向量机 (SVM)

**原始问题** (线性SVM):

$$
\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1
$$

**对偶问题**:

$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j
$$

**核化** (核SVM):

$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j k(x_i, x_j)
$$

**决策函数**:

$$
f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i k(x, x_i) + b\right)
$$

---

### 2. 核岭回归

**问题**:

$$
\min_{f \in \mathcal{H}_k} \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}_k}^2
$$

**解**:

$$
f^*(x) = \sum_{i=1}^n \alpha_i k(x, x_i)
$$

其中：

$$
\alpha = (K + \lambda I)^{-1} y
$$

---

### 3. 高斯过程

**高斯过程**是函数的分布，完全由均值函数和协方差函数（核函数）确定：

$$
f \sim \mathcal{GP}(m(x), k(x, x'))
$$

**预测**:

给定训练数据 $\{(x_i, y_i)\}_{i=1}^n$，在新点 $x_*$ 的预测分布：

$$
p(f_* | x_*, X, y) = \mathcal{N}(\mu_*, \sigma_*^2)
$$

其中：

$$
\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y
$$

$$
\sigma_*^2 = k(x_*, x_*) - k_*^T (K + \sigma_n^2 I)^{-1} k_*
$$

---

### 4. 核PCA

**主成分分析 (PCA)** 在特征空间中：

1. 计算核矩阵 $K$
2. 中心化：$\tilde{K} = K - \mathbf{1}_n K - K \mathbf{1}_n + \mathbf{1}_n K \mathbf{1}_n$
3. 特征值分解：$\tilde{K} = V \Lambda V^T$
4. 主成分：$\alpha_k = V[:, k]$

**投影**:

$$
z_k(x) = \sum_{i=1}^n \alpha_{ki} k(x, x_i)
$$

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

# 1. 核函数实现
def linear_kernel(X, Y):
    """线性核"""
    return X @ Y.T

def polynomial_kernel_custom(X, Y, degree=3, coef0=1):
    """多项式核"""
    return (X @ Y.T + coef0) ** degree

def rbf_kernel_custom(X, Y, gamma=1.0):
    """高斯核 (RBF)"""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^Ty
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    distances_sq = X_norm + Y_norm - 2 * X @ Y.T
    return np.exp(-gamma * distances_sq)


# 2. 核岭回归
class KernelRidgeRegression:
    """核岭回归"""

    def __init__(self, kernel='rbf', gamma=1.0, lambda_=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.lambda_ = lambda_

    def _compute_kernel(self, X, Y):
        """计算核矩阵"""
        if self.kernel == 'linear':
            return linear_kernel(X, Y)
        elif self.kernel == 'rbf':
            return rbf_kernel_custom(X, Y, self.gamma)
        elif self.kernel == 'polynomial':
            return polynomial_kernel_custom(X, Y)

    def fit(self, X, y):
        """训练"""
        self.X_train = X
        K = self._compute_kernel(X, X)
        n = len(y)
        self.alpha = np.linalg.solve(K + self.lambda_ * np.eye(n), y)

    def predict(self, X):
        """预测"""
        K = self._compute_kernel(X, self.X_train)
        return K @ self.alpha


# 3. 核SVM可视化
def visualize_kernel_svm():
    """可视化核SVM"""
    # 生成非线性可分数据
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)

    # 训练不同核的SVM
    kernels = ['linear', 'poly', 'rbf']
    titles = ['Linear Kernel', 'Polynomial Kernel (degree=3)', 'RBF Kernel']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, kernel, title in zip(axes, kernels, titles):
        # 训练SVM
        if kernel == 'poly':
            clf = SVC(kernel=kernel, degree=3, gamma='auto')
        else:
            clf = SVC(kernel=kernel, gamma='auto')
        clf.fit(X, y)

        # 绘制决策边界
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 100),
                             np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 100))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 绘制
        ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.tight_layout()
    # plt.show()


# 4. 核岭回归示例
def kernel_ridge_regression_demo():
    """核岭回归示例"""
    np.random.seed(42)

    # 生成非线性数据
    X_train = np.linspace(0, 10, 50).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.2 * np.random.randn(50)

    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()

    # 训练不同核的模型
    models = {
        'Linear': KernelRidgeRegression(kernel='linear', lambda_=0.1),
        'RBF': KernelRidgeRegression(kernel='rbf', gamma=0.5, lambda_=0.1),
        'Polynomial': KernelRidgeRegression(kernel='polynomial', lambda_=0.1)
    }

    plt.figure(figsize=(15, 5))

    for idx, (name, model) in enumerate(models.items(), 1):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        plt.subplot(1, 3, idx)
        plt.scatter(X_train, y_train, color='red', label='Training data')
        plt.plot(X_test, y_true, 'g-', label='True function', linewidth=2)
        plt.plot(X_test, y_pred, 'b--', label='Prediction', linewidth=2)
        plt.title(f'Kernel Ridge Regression ({name} Kernel)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()


# 5. 核矩阵可视化
def visualize_kernel_matrix():
    """可视化核矩阵"""
    np.random.seed(42)

    # 生成数据
    X = np.random.randn(50, 2)

    # 计算不同核的核矩阵
    K_linear = linear_kernel(X, X)
    K_rbf = rbf_kernel_custom(X, X, gamma=1.0)
    K_poly = polynomial_kernel_custom(X, X, degree=3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 线性核
    im1 = axes[0].imshow(K_linear, cmap='viridis')
    axes[0].set_title('Linear Kernel Matrix')
    plt.colorbar(im1, ax=axes[0])

    # RBF核
    im2 = axes[1].imshow(K_rbf, cmap='viridis')
    axes[1].set_title('RBF Kernel Matrix')
    plt.colorbar(im2, ax=axes[1])

    # 多项式核
    im3 = axes[2].imshow(K_poly, cmap='viridis')
    axes[2].set_title('Polynomial Kernel Matrix')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    print("=== Hilbert空间与RKHS示例 ===\n")

    print("1. 核SVM可视化")
    visualize_kernel_svm()

    print("\n2. 核岭回归")
    kernel_ridge_regression_demo()

    print("\n3. 核矩阵可视化")
    visualize_kernel_matrix()
```

---

## 📚 练习题

### 练习1：内积计算

在 $L^2[0, 1]$ 中，计算 $\langle f, g \rangle$，其中 $f(x) = x$，$g(x) = x^2$。

### 练习2：核函数验证

验证高斯核 $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$ 是正定核。

### 练习3：Representer定理

证明在核岭回归中，最优解可以表示为 $f^*(x) = \sum_{i=1}^n \alpha_i k(x, x_i)$。

### 练习4：核SVM

实现核SVM，并在非线性可分数据上测试。

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **MIT** | 18.102 - Introduction to Functional Analysis |
| **Stanford** | STATS315A - Modern Applied Statistics: Learning |
| **UC Berkeley** | STAT210B - Theoretical Statistics |
| **CMU** | 10-701 - Machine Learning |

---

## 📖 参考文献

1. **Berlinet & Thomas-Agnan (2004)**. *Reproducing Kernel Hilbert Spaces in Probability and Statistics*. Springer.

2. **Schölkopf & Smola (2002)**. *Learning with Kernels*. MIT Press.

3. **Steinwart & Christmann (2008)**. *Support Vector Machines*. Springer.

4. **Rasmussen & Williams (2006)**. *Gaussian Processes for Machine Learning*. MIT Press.

5. **Shawe-Taylor & Cristianini (2004)**. *Kernel Methods for Pattern Analysis*. Cambridge University Press.

---

*最后更新：2025年10月*-
