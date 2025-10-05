# 凸优化进阶 (Advanced Convex Optimization)

> **The Foundation of Efficient Machine Learning Algorithms**
>
> 高效机器学习算法的理论基础

---

## 目录

- [凸优化进阶 (Advanced Convex Optimization)](#凸优化进阶-advanced-convex-optimization)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 凸集与凸函数](#-凸集与凸函数)
    - [1. 凸集](#1-凸集)
    - [2. 凸函数](#2-凸函数)
    - [3. 强凸性](#3-强凸性)
  - [📊 凸优化问题](#-凸优化问题)
    - [1. 标准形式](#1-标准形式)
    - [2. 最优性条件](#2-最优性条件)
    - [3. 对偶理论](#3-对偶理论)
  - [🔬 凸优化算法](#-凸优化算法)
    - [1. 梯度投影法](#1-梯度投影法)
    - [2. 近端梯度法](#2-近端梯度法)
    - [3. 加速梯度法](#3-加速梯度法)
    - [4. ADMM算法](#4-admm算法)
  - [💡 收敛性分析](#-收敛性分析)
    - [1. 梯度下降收敛率](#1-梯度下降收敛率)
    - [2. Nesterov加速](#2-nesterov加速)
    - [3. 强凸情况](#3-强凸情况)
  - [🎨 在机器学习中的应用](#-在机器学习中的应用)
    - [1. 支持向量机 (SVM)](#1-支持向量机-svm)
    - [2. Lasso回归](#2-lasso回归)
    - [3. 逻辑回归](#3-逻辑回归)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：凸性判定](#练习1凸性判定)
    - [练习2：对偶问题](#练习2对偶问题)
    - [练习3：近端算子](#练习3近端算子)
    - [练习4：ADMM应用](#练习4admm应用)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**凸优化**是机器学习中最重要的优化工具，因为凸问题有全局最优解且可以高效求解。

**为什么凸优化重要**:

```text
凸优化的优势:
├─ 局部最优 = 全局最优
├─ 高效算法 (多项式时间)
├─ 理论保证 (收敛性、复杂度)
└─ 广泛应用 (SVM, Lasso, 逻辑回归)

机器学习中的凸问题:
├─ 线性回归 (最小二乘)
├─ 逻辑回归 (凸损失)
├─ SVM (凸二次规划)
└─ Lasso (凸正则化)
```

---

## 🎯 凸集与凸函数

### 1. 凸集

**定义 1.1 (凸集)**:

集合 $C \subseteq \mathbb{R}^n$ 是凸集，如果对于任意 $x, y \in C$ 和 $\theta \in [0, 1]$：

$$
\theta x + (1 - \theta) y \in C
$$

**几何意义**：连接集合中任意两点的线段仍在集合内。

**示例**:

- ✅ **凸集**: 超平面、半空间、球、椭球、多面体
- ❌ **非凸集**: 月牙形、环形

**定理 1.1 (凸集的保持性)**:

- 凸集的交集仍是凸集
- 凸集的仿射变换仍是凸集
- 凸集的笛卡尔积仍是凸集

---

### 2. 凸函数

**定义 2.1 (凸函数)**:

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是凸函数，如果其定义域 $\text{dom}(f)$ 是凸集，且对于任意 $x, y \in \text{dom}(f)$ 和 $\theta \in [0, 1]$：

$$
f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
$$

**几何意义**：函数图像上任意两点之间的弦位于函数图像上方。

**一阶条件** (可微情况):

$f$ 是凸函数当且仅当：

$$
f(y) \geq f(x) + \nabla f(x)^T (y - x) \quad \forall x, y
$$

**二阶条件** (二阶可微情况):

$f$ 是凸函数当且仅当其Hessian矩阵半正定：

$$
\nabla^2 f(x) \succeq 0 \quad \forall x
$$

**示例**:

- ✅ **凸函数**: $\|x\|_2$, $\|x\|_1$, $e^x$, $x^2$, $-\log x$ (x > 0)
- ❌ **非凸函数**: $\sin x$, $x^3$, $\log(1 + e^x)$ (虽然是凸的)

---

### 3. 强凸性

**定义 3.1 (强凸函数)**:

函数 $f$ 是 $\mu$-强凸的，如果对于任意 $x, y$：

$$
f(y) \geq f(x) + \nabla f(x)^T (y - x) + \frac{\mu}{2} \|y - x\|^2
$$

**等价条件**:

$$
\nabla^2 f(x) \succeq \mu I \quad \forall x
$$

**意义**：强凸函数有更好的收敛性质（线性收敛）。

**示例**:

- $f(x) = \frac{1}{2} x^T A x$ 是 $\lambda_{\min}(A)$-强凸的（当 $A \succ 0$）
- $f(x) = \|x\|^2$ 是 2-强凸的

---

## 📊 凸优化问题

### 1. 标准形式

**定义 1.1 (凸优化问题)**:

$$
\begin{align}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{align}
$$

其中 $f, g_i$ 是凸函数，$h_j$ 是仿射函数。

**特殊情况**:

```text
线性规划 (LP):
    f, g_i, h_j 都是仿射函数

二次规划 (QP):
    f 是二次函数，g_i, h_j 是仿射函数

二次约束二次规划 (QCQP):
    f, g_i 是二次函数，h_j 是仿射函数
```

---

### 2. 最优性条件

**定理 2.1 (KKT条件)**:

对于凸优化问题，点 $x^*$ 是最优解当且仅当存在 $\lambda^* \geq 0, \nu^*$ 使得：

1. **平稳性**: $\nabla f(x^*) + \sum_i \lambda_i^* \nabla g_i(x^*) + \sum_j \nu_j^* \nabla h_j(x^*) = 0$
2. **原始可行性**: $g_i(x^*) \leq 0$, $h_j(x^*) = 0$
3. **对偶可行性**: $\lambda_i^* \geq 0$
4. **互补松弛性**: $\lambda_i^* g_i(x^*) = 0$

**无约束情况**:

$$
\nabla f(x^*) = 0
$$

---

### 3. 对偶理论

**拉格朗日函数**:

$$
L(x, \lambda, \nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \nu_j h_j(x)
$$

**对偶函数**:

$$
g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)
$$

**对偶问题**:

$$
\begin{align}
\max_{\lambda, \nu} \quad & g(\lambda, \nu) \\
\text{s.t.} \quad & \lambda \geq 0
\end{align}
$$

**定理 3.1 (弱对偶性)**:

$$
g(\lambda, \nu) \leq f(x^*) \quad \forall \lambda \geq 0, \nu
$$

**定理 3.2 (强对偶性)**:

对于凸优化问题，如果Slater条件成立，则强对偶性成立：

$$
g(\lambda^*, \nu^*) = f(x^*)
$$

---

## 🔬 凸优化算法

### 1. 梯度投影法

**问题**:

$$
\min_{x \in C} f(x)
$$

其中 $C$ 是凸集。

**算法**:

$$
x_{t+1} = \Pi_C(x_t - \eta \nabla f(x_t))
$$

其中 $\Pi_C$ 是投影算子：

$$
\Pi_C(y) = \arg\min_{x \in C} \|x - y\|^2
$$

**收敛性**:

- 凸函数：$O(1/t)$
- 强凸函数：$O(e^{-\mu \eta t})$

---

### 2. 近端梯度法

**问题**:

$$
\min_x f(x) + g(x)
$$

其中 $f$ 光滑，$g$ 可能不光滑但有简单的近端算子。

**近端算子**:

$$
\text{prox}_{\eta g}(y) = \arg\min_x \left\{ g(x) + \frac{1}{2\eta} \|x - y\|^2 \right\}
$$

**算法**:

$$
x_{t+1} = \text{prox}_{\eta g}(x_t - \eta \nabla f(x_t))
$$

**示例** ($\ell_1$ 正则化):

$$
\text{prox}_{\eta \lambda \|\cdot\|_1}(y) = \text{sign}(y) \odot \max(|y| - \eta \lambda, 0)
$$

这就是**软阈值算子** (Soft-thresholding)。

---

### 3. 加速梯度法

**Nesterov加速梯度法**:

$$
\begin{align}
y_t &= x_t + \frac{t - 1}{t + 2} (x_t - x_{t-1}) \\
x_{t+1} &= y_t - \eta \nabla f(y_t)
\end{align}
$$

**收敛率**:

- 标准梯度下降：$O(1/t)$
- Nesterov加速：$O(1/t^2)$ ✅

**直觉**：使用动量项加速收敛。

---

### 4. ADMM算法

**问题** (可分离形式):

$$
\min_{x, z} f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c
$$

**增广拉格朗日函数**:

$$
L_\rho(x, z, y) = f(x) + g(z) + y^T(Ax + Bz - c) + \frac{\rho}{2} \|Ax + Bz - c\|^2
$$

**ADMM迭代**:

$$
\begin{align}
x_{t+1} &= \arg\min_x L_\rho(x, z_t, y_t) \\
z_{t+1} &= \arg\min_z L_\rho(x_{t+1}, z, y_t) \\
y_{t+1} &= y_t + \rho (Ax_{t+1} + Bz_{t+1} - c)
\end{align}
$$

**优势**:

- 可处理大规模问题
- 可并行化
- 收敛性好

---

## 💡 收敛性分析

### 1. 梯度下降收敛率

**定理 1.1 (凸函数)**:

假设 $f$ 是 $L$-光滑的凸函数，使用固定步长 $\eta = 1/L$：

$$
f(x_t) - f^* \leq \frac{L \|x_0 - x^*\|^2}{2t}
$$

**收敛率**: $O(1/t)$

---

### 2. Nesterov加速

**定理 2.1 (Nesterov加速)**:

使用Nesterov加速梯度法：

$$
f(x_t) - f^* \leq \frac{2L \|x_0 - x^*\|^2}{(t+1)^2}
$$

**收敛率**: $O(1/t^2)$ ✅ 比标准梯度下降快！

---

### 3. 强凸情况

**定理 3.1 (强凸函数)**:

假设 $f$ 是 $\mu$-强凸且 $L$-光滑的，使用固定步长 $\eta = 1/L$：

$$
\|x_t - x^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^t \|x_0 - x^*\|^2
$$

**收敛率**: $O(e^{-\mu t / L})$ (线性收敛)

**条件数**:

$$
\kappa = \frac{L}{\mu}
$$

条件数越小，收敛越快。

---

## 🎨 在机器学习中的应用

### 1. 支持向量机 (SVM)

**原始问题**:

$$
\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1
$$

**对偶问题**:

$$
\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j \quad \text{s.t.} \quad \alpha_i \geq 0, \sum_i \alpha_i y_i = 0
$$

**凸二次规划** → 全局最优解

---

### 2. Lasso回归

**问题**:

$$
\min_w \frac{1}{2} \|Xw - y\|^2 + \lambda \|w\|_1
$$

**近端梯度法**:

$$
w_{t+1} = \text{prox}_{\eta \lambda \|\cdot\|_1}(w_t - \eta X^T(Xw_t - y))
$$

其中近端算子是软阈值：

$$
[\text{prox}_{\eta \lambda \|\cdot\|_1}(w)]_i = \text{sign}(w_i) \max(|w_i| - \eta \lambda, 0)
$$

---

### 3. 逻辑回归

**问题**:

$$
\min_w \sum_i \log(1 + e^{-y_i w^T x_i}) + \frac{\lambda}{2} \|w\|^2
$$

**凸优化** → 梯度下降/牛顿法

**梯度**:

$$
\nabla f(w) = -\sum_i \frac{y_i x_i}{1 + e^{y_i w^T x_i}} + \lambda w
$$

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 梯度投影法
def gradient_projection(f, grad_f, project, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """梯度投影法"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {i} iterations")
            break
        
        # 梯度步
        x_new = x - lr * grad
        
        # 投影到可行域
        x = project(x_new)
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


# 2. 近端梯度法
def proximal_gradient(f, grad_f, prox_g, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """近端梯度法"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        # 梯度步
        x_temp = x - lr * grad
        
        # 近端算子
        x = prox_g(x_temp, lr)
        trajectory.append(x.copy())
        
        if np.linalg.norm(x - trajectory[-2]) < tol:
            print(f"Converged in {i} iterations")
            break
    
    return x, np.array(trajectory)


# 3. 软阈值算子 (L1近端算子)
def soft_threshold(x, lambda_):
    """软阈值算子: prox_{lambda ||·||_1}"""
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)


# 4. Nesterov加速梯度法
def nesterov_accelerated_gradient(f, grad_f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """Nesterov加速梯度法"""
    x = x0.copy()
    x_prev = x0.copy()
    trajectory = [x.copy()]
    
    for t in range(1, max_iter):
        # 动量项
        momentum = (t - 1) / (t + 2)
        y = x + momentum * (x - x_prev)
        
        grad = grad_f(y)
        
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {t} iterations")
            break
        
        x_prev = x.copy()
        x = y - lr * grad
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


# 5. ADMM算法 (Lasso示例)
def admm_lasso(X, y, lambda_, rho=1.0, max_iter=100, tol=1e-4):
    """ADMM求解Lasso: min ||Xw - y||^2 + lambda ||w||_1"""
    n, d = X.shape
    
    # 初始化
    w = np.zeros(d)
    z = np.zeros(d)
    u = np.zeros(d)
    
    # 预计算
    XtX = X.T @ X
    Xty = X.T @ y
    L = XtX + rho * np.eye(d)
    
    for i in range(max_iter):
        # w-update (解析解)
        w = np.linalg.solve(L, Xty + rho * (z - u))
        
        # z-update (软阈值)
        z_old = z.copy()
        z = soft_threshold(w + u, lambda_ / rho)
        
        # u-update
        u = u + w - z
        
        # 检查收敛
        if np.linalg.norm(z - z_old) < tol:
            print(f"ADMM converged in {i+1} iterations")
            break
    
    return w


# 示例：Lasso回归
def lasso_example():
    """Lasso回归示例"""
    np.random.seed(42)
    
    # 生成稀疏数据
    n, d = 100, 50
    k = 5  # 真实非零系数数量
    
    X = np.random.randn(n, d)
    w_true = np.zeros(d)
    w_true[:k] = np.random.randn(k)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    # 近端梯度法
    lambda_ = 0.1
    
    def f(w):
        return 0.5 * np.sum((X @ w - y)**2)
    
    def grad_f(w):
        return X.T @ (X @ w - y)
    
    def prox_g(w, eta):
        return soft_threshold(w, eta * lambda_)
    
    w0 = np.zeros(d)
    w_prox, traj_prox = proximal_gradient(f, grad_f, prox_g, w0, lr=0.001, max_iter=1000)
    
    # ADMM
    w_admm = admm_lasso(X, y, lambda_, rho=1.0, max_iter=100)
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 真实系数
    plt.subplot(1, 3, 1)
    plt.stem(w_true)
    plt.title('True Coefficients')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    # 近端梯度法结果
    plt.subplot(1, 3, 2)
    plt.stem(w_prox)
    plt.title('Proximal Gradient')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    # ADMM结果
    plt.subplot(1, 3, 3)
    plt.stem(w_admm)
    plt.title('ADMM')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    plt.tight_layout()
    # plt.show()
    
    print(f"True non-zeros: {np.sum(w_true != 0)}")
    print(f"Prox non-zeros: {np.sum(np.abs(w_prox) > 1e-3)}")
    print(f"ADMM non-zeros: {np.sum(np.abs(w_admm) > 1e-3)}")


# 示例：加速对比
def acceleration_comparison():
    """对比标准梯度下降与Nesterov加速"""
    # 强凸二次函数
    A = np.array([[10, 0], [0, 1]])  # 条件数 = 10
    b = np.array([1, 1])
    
    def f(x):
        return 0.5 * x @ A @ x - b @ x
    
    def grad_f(x):
        return A @ x - b
    
    x0 = np.array([5.0, 5.0])
    
    # 标准梯度下降
    from scipy.optimize import minimize_scalar
    
    def gd(x0, lr, max_iter=1000):
        x = x0.copy()
        traj = [x.copy()]
        for _ in range(max_iter):
            x = x - lr * grad_f(x)
            traj.append(x.copy())
        return np.array(traj)
    
    traj_gd = gd(x0, lr=0.1, max_iter=100)
    
    # Nesterov加速
    _, traj_nag = nesterov_accelerated_gradient(f, grad_f, x0, lr=0.1, max_iter=100)
    
    # 可视化
    x_opt = np.linalg.solve(A, b)
    
    plt.figure(figsize=(15, 5))
    
    # 等高线
    x_range = np.linspace(-1, 6, 100)
    y_range = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    # 标准梯度下降
    plt.subplot(1, 2, 1)
    plt.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)
    plt.plot(traj_gd[:, 0], traj_gd[:, 1], 'r-o', markersize=3, label='GD')
    plt.plot(x_opt[0], x_opt[1], 'g*', markersize=15, label='Optimum')
    plt.title('Standard Gradient Descent')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    
    # Nesterov加速
    plt.subplot(1, 2, 2)
    plt.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)
    plt.plot(traj_nag[:, 0], traj_nag[:, 1], 'b-o', markersize=3, label='NAG')
    plt.plot(x_opt[0], x_opt[1], 'g*', markersize=15, label='Optimum')
    plt.title('Nesterov Accelerated Gradient')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    
    plt.tight_layout()
    # plt.show()
    
    print(f"GD iterations to converge: {len(traj_gd)}")
    print(f"NAG iterations to converge: {len(traj_nag)}")


if __name__ == "__main__":
    print("=== 凸优化进阶示例 ===")
    
    print("\n1. Lasso回归示例")
    lasso_example()
    
    print("\n2. 加速梯度法对比")
    acceleration_comparison()
```

---

## 📚 练习题

### 练习1：凸性判定

判断以下函数是否为凸函数：

1. $f(x) = e^x$
2. $f(x) = x^4$
3. $f(x) = \log(1 + e^x)$
4. $f(x, y) = x^2 + xy + y^2$

### 练习2：对偶问题

求解以下问题的对偶问题：

$$
\min_x \frac{1}{2} x^T Q x + c^T x \quad \text{s.t.} \quad Ax = b, \; x \geq 0
$$

### 练习3：近端算子

计算以下函数的近端算子：

1. $g(x) = \lambda \|x\|_1$
2. $g(x) = I_C(x)$ (指示函数，$C$ 是凸集)

### 练习4：ADMM应用

使用ADMM求解以下问题：

$$
\min_{x, z} \frac{1}{2} \|Ax - b\|^2 + \lambda \|z\|_1 \quad \text{s.t.} \quad x = z
$$

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | EE364A - Convex Optimization I |
| **Stanford** | EE364B - Convex Optimization II |
| **MIT** | 6.255J - Optimization Methods |
| **UC Berkeley** | EECS 127 - Optimization Models |
| **CMU** | 10-725 - Convex Optimization |

---

## 📖 参考文献

1. **Boyd & Vandenberghe (2004)**. *Convex Optimization*. Cambridge University Press.

2. **Nesterov, Y. (2004)**. *Introductory Lectures on Convex Optimization*. Springer.

3. **Bertsekas, D. (2009)**. *Convex Optimization Theory*. Athena Scientific.

4. **Parikh & Boyd (2014)**. "Proximal Algorithms". *Foundations and Trends in Optimization*.

5. **Beck & Teboulle (2009)**. "A Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)". *SIAM Journal on Imaging Sciences*.

---

*最后更新：2025年10月*-
