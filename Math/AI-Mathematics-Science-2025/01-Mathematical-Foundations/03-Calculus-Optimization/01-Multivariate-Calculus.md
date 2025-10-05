# 多元微积分 (Multivariate Calculus)

> **The Mathematical Foundation of Deep Learning Optimization**
>
> 深度学习优化的数学基石

---

## 目录

- [多元微积分 (Multivariate Calculus)](#多元微积分-multivariate-calculus)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 偏导数与梯度](#-偏导数与梯度)
    - [1. 偏导数](#1-偏导数)
    - [2. 梯度向量](#2-梯度向量)
    - [3. 方向导数](#3-方向导数)
  - [📊 多元函数的泰勒展开](#-多元函数的泰勒展开)
    - [1. 一阶泰勒展开](#1-一阶泰勒展开)
    - [2. 二阶泰勒展开](#2-二阶泰勒展开)
    - [3. Hessian矩阵](#3-hessian矩阵)
  - [🔬 链式法则](#-链式法则)
    - [1. 标量链式法则](#1-标量链式法则)
    - [2. 向量链式法则](#2-向量链式法则)
    - [3. 雅可比矩阵](#3-雅可比矩阵)
  - [💡 梯度下降的数学原理](#-梯度下降的数学原理)
    - [1. 最速下降方向](#1-最速下降方向)
    - [2. 收敛性分析](#2-收敛性分析)
    - [3. 步长选择](#3-步长选择)
  - [🎨 约束优化](#-约束优化)
    - [1. 拉格朗日乘数法](#1-拉格朗日乘数法)
    - [2. KKT条件](#2-kkt条件)
  - [🔧 在深度学习中的应用](#-在深度学习中的应用)
    - [1. 反向传播](#1-反向传播)
    - [2. 损失函数的曲率](#2-损失函数的曲率)
    - [3. 优化算法](#3-优化算法)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：梯度计算](#练习1梯度计算)
    - [练习2：Hessian矩阵](#练习2hessian矩阵)
    - [练习3：梯度下降](#练习3梯度下降)
    - [练习4：约束优化](#练习4约束优化)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**多元微积分**是研究多变量函数的微分与积分，是深度学习优化的数学基础。

**核心概念**：

```text
单变量微积分:
    f: ℝ → ℝ
    导数: f'(x)
    
多元微积分:
    f: ℝⁿ → ℝ
    偏导数: ∂f/∂xᵢ
    梯度: ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ
    
深度学习:
    损失函数: L: ℝᵈ → ℝ (d = 参数数量)
    优化: θ* = argmin L(θ)
    梯度下降: θ ← θ - η∇L(θ)
```

---

## 🎯 偏导数与梯度

### 1. 偏导数

**定义 1.1 (偏导数)**:

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 关于第 $i$ 个变量的偏导数：

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}
$$

**直觉**：固定其他变量，只对 $x_i$ 求导。

**示例**：

$$
f(x, y) = x^2 + 3xy + y^2
$$

$$
\frac{\partial f}{\partial x} = 2x + 3y, \quad \frac{\partial f}{\partial y} = 3x + 2y
$$

---

### 2. 梯度向量

**定义 2.1 (梯度)**:

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的梯度是所有偏导数组成的向量：

$$
\nabla f(x) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

**几何意义**：

- 梯度指向函数**增长最快**的方向
- 梯度的模 $\|\nabla f\|$ 是增长的速率

**示例**：

$$
f(x, y) = x^2 + y^2
$$

$$
\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}
$$

在点 $(1, 1)$：$\nabla f(1, 1) = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$

---

### 3. 方向导数

**定义 3.1 (方向导数)**:

函数 $f$ 在点 $x$ 沿方向 $v$ （单位向量）的方向导数：

$$
D_v f(x) = \lim_{t \to 0} \frac{f(x + tv) - f(x)}{t}
$$

**定理 3.1**：

$$
D_v f(x) = \nabla f(x) \cdot v
$$

**推论**：

- 当 $v = \frac{\nabla f}{\|\nabla f\|}$ 时，$D_v f$ 最大（最速上升）
- 当 $v = -\frac{\nabla f}{\|\nabla f\|}$ 时，$D_v f$ 最小（最速下降）

---

## 📊 多元函数的泰勒展开

### 1. 一阶泰勒展开

**定理 1.1 (一阶泰勒展开)**:

$$
f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
$$

**应用**：线性近似

**示例**：

$$
f(x, y) = x^2 + y^2
$$

在点 $(1, 1)$ 附近：

$$
f(1 + \Delta x, 1 + \Delta y) \approx 2 + 2\Delta x + 2\Delta y
$$

---

### 2. 二阶泰勒展开

**定理 2.1 (二阶泰勒展开)**:

$$
f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x
$$

其中 $H(x)$ 是Hessian矩阵。

**应用**：二次近似，牛顿法

---

### 3. Hessian矩阵

**定义 3.1 (Hessian矩阵)**:

$$
H(x) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

**性质**：

- **对称性**：$H_{ij} = H_{ji}$ （Schwarz定理）
- **曲率信息**：描述函数的局部曲率

**特征值与曲率**：

- 所有特征值 > 0：局部极小值（凸）
- 所有特征值 < 0：局部极大值（凹）
- 特征值有正有负：鞍点

**示例**：

$$
f(x, y) = x^2 + y^2
$$

$$
H = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}
$$

特征值：$\lambda_1 = \lambda_2 = 2 > 0$ → 凸函数

---

## 🔬 链式法则

### 1. 标量链式法则

**定理 1.1 (标量链式法则)**:

设 $y = f(u)$，$u = g(x)$，则：

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

**多元情况**：

设 $z = f(x, y)$，$x = g(t)$，$y = h(t)$，则：

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt}
$$

---

### 2. 向量链式法则

**定理 2.1 (向量链式法则)**:

设 $y = f(u)$，$u = g(x)$，其中 $f: \mathbb{R}^m \to \mathbb{R}$，$g: \mathbb{R}^n \to \mathbb{R}^m$，则：

$$
\nabla_x f = J_g^T \nabla_u f
$$

其中 $J_g$ 是 $g$ 的雅可比矩阵。

**反向传播的数学基础**！

---

### 3. 雅可比矩阵

**定义 3.1 (雅可比矩阵)**:

设 $f: \mathbb{R}^n \to \mathbb{R}^m$，$f(x) = [f_1(x), \ldots, f_m(x)]^T$，则雅可比矩阵：

$$
J_f(x) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}_{m \times n}
$$

**示例**：

$$
f(x, y) = \begin{bmatrix} x^2 + y \\ xy \end{bmatrix}
$$

$$
J_f = \begin{bmatrix}
2x & 1 \\
y & x
\end{bmatrix}
$$

---

## 💡 梯度下降的数学原理

### 1. 最速下降方向

**定理 1.1**：

负梯度方向 $-\nabla f(x)$ 是函数 $f$ 在点 $x$ 的**最速下降方向**。

**证明**：

方向导数：$D_v f(x) = \nabla f(x) \cdot v$

最小化 $D_v f$ 等价于最小化 $\nabla f \cdot v = \|\nabla f\| \|v\| \cos \theta$

当 $\theta = \pi$ 时最小，即 $v = -\frac{\nabla f}{\|\nabla f\|}$。

---

### 2. 收敛性分析

**定理 2.1 (梯度下降收敛, 凸情况)**:

假设 $f$ 是 $L$-光滑的凸函数，使用固定步长 $\eta \leq 1/L$：

$$
f(x_t) - f^* \leq \frac{\|x_0 - x^*\|^2}{2\eta t}
$$

**收敛率**：$O(1/t)$

**强凸情况**：

假设 $f$ 是 $\mu$-强凸的，则：

$$
\|x_t - x^*\|^2 \leq (1 - \mu \eta)^t \|x_0 - x^*\|^2
$$

**收敛率**：$O(e^{-\mu \eta t})$ （线性收敛）

---

### 3. 步长选择

**固定步长**：$\eta = \text{const}$

- 简单
- 需要调参

**线搜索** (Line Search)：

在每步选择最优步长：

$$
\eta_t = \arg\min_{\eta > 0} f(x_t - \eta \nabla f(x_t))
$$

**Armijo条件** (Backtracking Line Search)：

选择 $\eta$ 使得：

$$
f(x - \eta \nabla f) \leq f(x) - c \eta \|\nabla f\|^2
$$

其中 $c \in (0, 1)$（通常0.5）。

---

## 🎨 约束优化

### 1. 拉格朗日乘数法

**问题**：

$$
\min_{x} f(x) \quad \text{s.t.} \quad g(x) = 0
$$

**拉格朗日函数**：

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda g(x)
$$

**最优性条件**：

$$
\nabla_x \mathcal{L} = \nabla f(x) + \lambda \nabla g(x) = 0
$$

$$
\nabla_\lambda \mathcal{L} = g(x) = 0
$$

**几何解释**：

在最优点，$\nabla f$ 和 $\nabla g$ 平行。

---

### 2. KKT条件

**问题**：

$$
\min_{x} f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \; h_j(x) = 0
$$

**拉格朗日函数**：

$$
\mathcal{L}(x, \mu, \lambda) = f(x) + \sum_i \mu_i g_i(x) + \sum_j \lambda_j h_j(x)
$$

**KKT条件**：

1. **平稳性**：$\nabla_x \mathcal{L} = 0$
2. **原始可行性**：$g_i(x) \leq 0$，$h_j(x) = 0$
3. **对偶可行性**：$\mu_i \geq 0$
4. **互补松弛性**：$\mu_i g_i(x) = 0$

---

## 🔧 在深度学习中的应用

### 1. 反向传播

**链式法则的应用**：

设神经网络 $f(x; \theta) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$

损失函数：$\mathcal{L}(\theta) = \ell(f(x; \theta), y)$

**梯度计算**：

$$
\frac{\partial \mathcal{L}}{\partial \theta_l} = \frac{\partial \mathcal{L}}{\partial f_L} \frac{\partial f_L}{\partial f_{L-1}} \cdots \frac{\partial f_{l+1}}{\partial f_l} \frac{\partial f_l}{\partial \theta_l}
$$

**反向传播算法**：

从输出层到输入层，逐层计算梯度。

---

### 2. 损失函数的曲率

**Hessian矩阵的作用**：

- **曲率信息**：描述损失函数的局部形状
- **优化难度**：高曲率方向难优化
- **二阶方法**：牛顿法利用Hessian加速收敛

**条件数**：

$$
\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

- 条件数大：优化困难
- 预处理：改善条件数

---

### 3. 优化算法

**一阶方法**：

- 梯度下降
- SGD
- Momentum
- Adam

**二阶方法**：

- 牛顿法：$x_{t+1} = x_t - H^{-1} \nabla f$
- L-BFGS：近似Hessian

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 梯度计算
def compute_gradient(f, x, h=1e-5):
    """数值计算梯度"""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


# 2. Hessian矩阵计算
def compute_hessian(f, x, h=1e-5):
    """数值计算Hessian矩阵"""
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pp[i] += h
            x_pp[j] += h
            
            x_pm = x.copy()
            x_pm[i] += h
            x_pm[j] -= h
            
            x_mp = x.copy()
            x_mp[i] -= h
            x_mp[j] += h
            
            x_mm = x.copy()
            x_mm[i] -= h
            x_mm[j] -= h
            
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
    
    return H


# 3. 梯度下降
def gradient_descent(f, grad_f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """梯度下降算法"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {i} iterations")
            break
        
        x = x - lr * grad
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


# 4. 带Armijo线搜索的梯度下降
def gradient_descent_armijo(f, grad_f, x0, max_iter=1000, tol=1e-6, c=0.5, rho=0.9):
    """带Armijo线搜索的梯度下降"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {i} iterations")
            break
        
        # Armijo线搜索
        lr = 1.0
        while f(x - lr * grad) > f(x) - c * lr * np.dot(grad, grad):
            lr *= rho
        
        x = x - lr * grad
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


# 5. 牛顿法
def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    """牛顿法"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {i} iterations")
            break
        
        H = hess_f(x)
        
        # 求解 H * d = -grad
        try:
            d = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            print("Singular Hessian, using gradient descent step")
            d = -grad
        
        x = x + d
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


# 示例：Rosenbrock函数
def rosenbrock(x):
    """Rosenbrock函数: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Rosenbrock函数的梯度"""
    grad = np.zeros(2)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

def rosenbrock_hess(x):
    """Rosenbrock函数的Hessian"""
    H = np.zeros((2, 2))
    H[0, 0] = 2 - 400 * x[1] + 1200 * x[0]**2
    H[0, 1] = -400 * x[0]
    H[1, 0] = -400 * x[0]
    H[1, 1] = 200
    return H


# 可视化
def visualize_optimization():
    """可视化优化过程"""
    # 初始点
    x0 = np.array([-1.5, 2.0])
    
    # 运行不同算法
    x_gd, traj_gd = gradient_descent(rosenbrock, rosenbrock_grad, x0, lr=0.001, max_iter=5000)
    x_armijo, traj_armijo = gradient_descent_armijo(rosenbrock, rosenbrock_grad, x0, max_iter=1000)
    x_newton, traj_newton = newton_method(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0, max_iter=50)
    
    # 绘制等高线
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))
    
    plt.figure(figsize=(15, 5))
    
    # 梯度下降
    plt.subplot(1, 3, 1)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray', alpha=0.3)
    plt.plot(traj_gd[:, 0], traj_gd[:, 1], 'r-o', markersize=3, label='GD')
    plt.plot(1, 1, 'g*', markersize=15, label='Optimum')
    plt.title('Gradient Descent')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Armijo线搜索
    plt.subplot(1, 3, 2)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray', alpha=0.3)
    plt.plot(traj_armijo[:, 0], traj_armijo[:, 1], 'b-o', markersize=3, label='GD + Armijo')
    plt.plot(1, 1, 'g*', markersize=15, label='Optimum')
    plt.title('GD with Armijo Line Search')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # 牛顿法
    plt.subplot(1, 3, 3)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray', alpha=0.3)
    plt.plot(traj_newton[:, 0], traj_newton[:, 1], 'g-o', markersize=3, label='Newton')
    plt.plot(1, 1, 'g*', markersize=15, label='Optimum')
    plt.title('Newton Method')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    plt.tight_layout()
    # plt.show()
    
    print(f"GD iterations: {len(traj_gd)}")
    print(f"GD + Armijo iterations: {len(traj_armijo)}")
    print(f"Newton iterations: {len(traj_newton)}")


if __name__ == "__main__":
    print("=== 多元微积分示例 ===")
    
    # 测试梯度计算
    x = np.array([1.0, 2.0])
    grad_numerical = compute_gradient(rosenbrock, x)
    grad_analytical = rosenbrock_grad(x)
    
    print(f"\n数值梯度: {grad_numerical}")
    print(f"解析梯度: {grad_analytical}")
    print(f"误差: {np.linalg.norm(grad_numerical - grad_analytical)}")
    
    # 测试Hessian计算
    hess_numerical = compute_hessian(rosenbrock, x)
    hess_analytical = rosenbrock_hess(x)
    
    print(f"\n数值Hessian:\n{hess_numerical}")
    print(f"解析Hessian:\n{hess_analytical}")
    print(f"误差: {np.linalg.norm(hess_numerical - hess_analytical)}")
    
    # 可视化优化
    print("\n=== 优化算法对比 ===")
    visualize_optimization()
```

---

## 📚 练习题

### 练习1：梯度计算

计算以下函数的梯度：

1. $f(x, y) = x^2 + 2xy + 3y^2$
2. $f(x, y) = e^{x+y}$
3. $f(x, y, z) = x^2 y + y^2 z + z^2 x$

### 练习2：Hessian矩阵

计算 $f(x, y) = x^3 + y^3 - 3xy$ 的Hessian矩阵，并判断点 $(1, 1)$ 的性质。

### 练习3：梯度下降

使用梯度下降最小化 $f(x, y) = x^2 + 4y^2$，初始点 $(2, 2)$，学习率 $\eta = 0.1$。

### 练习4：约束优化

使用拉格朗日乘数法求解：

$$
\min_{x, y} x^2 + y^2 \quad \text{s.t.} \quad x + y = 1
$$

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **MIT** | 18.02 Multivariable Calculus |
| **Stanford** | Math 51 Linear Algebra & Multivariable Calculus |
| **UC Berkeley** | Math 53 Multivariable Calculus |
| **CMU** | 21-259 Calculus in Three Dimensions |

---

## 📖 参考文献

1. **Stewart, J. (2015)**. *Multivariable Calculus*. Cengage Learning.

2. **Nocedal & Wright (2006)**. *Numerical Optimization*. Springer.

3. **Boyd & Vandenberghe (2004)**. *Convex Optimization*. Cambridge University Press.

4. **Goodfellow et al. (2016)**. *Deep Learning*. MIT Press. (Chapter 4: Numerical Computation)

---

*最后更新：2025年10月*-
