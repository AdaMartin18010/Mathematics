# 二阶优化方法 (Second-Order Optimization Methods)

> **Beyond Gradient Descent: Leveraging Curvature Information**
>
> 超越梯度下降：利用曲率信息

---

## 目录

- [二阶优化方法 (Second-Order Optimization Methods)](#二阶优化方法-second-order-optimization-methods)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 Newton法](#-newton法)
    - [1. 基本Newton法](#1-基本newton法)
    - [2. 阻尼Newton法](#2-阻尼newton法)
    - [3. 信赖域Newton法](#3-信赖域newton法)
  - [📊 拟Newton法](#-拟newton法)
    - [1. BFGS算法](#1-bfgs算法)
    - [2. L-BFGS算法](#2-l-bfgs算法)
    - [3. DFP算法](#3-dfp算法)
  - [🔬 共轭梯度法](#-共轭梯度法)
    - [1. 线性共轭梯度](#1-线性共轭梯度)
    - [2. 非线性共轭梯度](#2-非线性共轭梯度)
    - [3. 预条件共轭梯度](#3-预条件共轭梯度)
  - [💡 Gauss-Newton法](#-gauss-newton法)
    - [1. 基本Gauss-Newton法](#1-基本gauss-newton法)
    - [2. Levenberg-Marquardt算法](#2-levenberg-marquardt算法)
  - [🎨 在深度学习中的应用](#-在深度学习中的应用)
    - [1. 自然梯度](#1-自然梯度)
    - [2. K-FAC](#2-k-fac)
    - [3. Shampoo](#3-shampoo)
  - [🔧 实用技巧](#-实用技巧)
    - [1. Hessian近似](#1-hessian近似)
    - [2. 线搜索策略](#2-线搜索策略)
    - [3. 收敛性分析](#3-收敛性分析)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：Newton法](#练习1newton法)
    - [练习2：BFGS](#练习2bfgs)
    - [练习3：共轭梯度](#练习3共轭梯度)
    - [练习4：Gauss-Newton](#练习4gauss-newton)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**二阶优化方法**利用目标函数的二阶导数信息（Hessian矩阵），相比一阶方法（如梯度下降）具有更快的收敛速度。

**为什么使用二阶方法**:

```text
优势:
├─ 收敛速度快: 二次收敛 vs 线性收敛
├─ 自适应步长: 自动调整学习率
├─ 曲率信息: 利用Hessian矩阵
└─ 理论保证: 强凸函数局部二次收敛

挑战:
├─ 计算成本: O(n³) Hessian求逆
├─ 存储需求: O(n²) Hessian矩阵
├─ 非凸问题: 可能收敛到鞍点
└─ 深度学习: 参数量巨大
```

---

## 🎯 Newton法

### 1. 基本Newton法

**算法思想**:

在当前点 $x_k$ 处，用二阶Taylor展开逼近目标函数：

$$
f(x) \approx f(x_k) + \nabla f(x_k)^T (x - x_k) + \frac{1}{2} (x - x_k)^T H_k (x - x_k)
$$

其中 $H_k = \nabla^2 f(x_k)$ 是Hessian矩阵。

**最优化条件**:

$$
\nabla f(x) \approx \nabla f(x_k) + H_k (x - x_k) = 0
$$

**Newton方向**:

$$
d_k = -H_k^{-1} \nabla f(x_k)
$$

**更新规则**:

$$
x_{k+1} = x_k + d_k = x_k - H_k^{-1} \nabla f(x_k)
$$

---

**算法 1.1 (Newton法)**:

```text
输入: 初始点 x₀, 容差 ε
输出: 最优解 x*

1. for k = 0, 1, 2, ... do
2.     计算梯度 g_k = ∇f(x_k)
3.     if ||g_k|| < ε then
4.         return x_k
5.     计算Hessian H_k = ∇²f(x_k)
6.     求解 H_k d_k = -g_k  (Newton方程)
7.     x_{k+1} = x_k + d_k
8. end for
```

---

**收敛性**:

**定理 1.1 (Newton法收敛性)**:

设 $f$ 是强凸函数，$\nabla^2 f$ 是Lipschitz连续的。如果初始点 $x_0$ 足够接近最优解 $x^*$，则Newton法**二次收敛**：

$$
\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2
$$

---

### 2. 阻尼Newton法

**问题**: 基本Newton法可能不收敛（远离最优解时）。

**解决**: 引入步长 $\alpha_k$：

$$
x_{k+1} = x_k - \alpha_k H_k^{-1} \nabla f(x_k)
$$

**Armijo线搜索**:

选择 $\alpha_k$ 使得：

$$
f(x_k + \alpha_k d_k) \leq f(x_k) + c_1 \alpha_k \nabla f(x_k)^T d_k
$$

其中 $c_1 \in (0, 1)$（通常取0.0001）。

---

### 3. 信赖域Newton法

**思想**: 在信赖域内最小化二次模型。

**子问题**:

$$
\min_{d} \quad m_k(d) = f(x_k) + \nabla f(x_k)^T d + \frac{1}{2} d^T H_k d
$$

$$
\text{s.t.} \quad \|d\| \leq \Delta_k
$$

其中 $\Delta_k$ 是信赖域半径。

**更新规则**:

根据实际下降与预测下降的比值调整信赖域：

$$
\rho_k = \frac{f(x_k) - f(x_k + d_k)}{m_k(0) - m_k(d_k)}
$$

- 如果 $\rho_k > 0.75$：增大 $\Delta_{k+1}$
- 如果 $\rho_k < 0.25$：减小 $\Delta_{k+1}$

---

## 📊 拟Newton法

**核心思想**: 避免计算Hessian矩阵，用近似矩阵 $B_k \approx H_k$ 代替。

### 1. BFGS算法

**Broyden-Fletcher-Goldfarb-Shanno算法**:

**拟Newton条件** (Secant equation):

$$
B_{k+1} s_k = y_k
$$

其中：

- $s_k = x_{k+1} - x_k$
- $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$

**BFGS更新公式**:

$$
B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{y_k y_k^T}{y_k^T s_k}
$$

或者直接更新逆矩阵 $H_k = B_k^{-1}$：

$$
H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}
$$

---

**算法 1.2 (BFGS)**:

```text
输入: 初始点 x₀, 初始Hessian逆 H₀ = I
输出: 最优解 x*

1. for k = 0, 1, 2, ... do
2.     计算梯度 g_k = ∇f(x_k)
3.     计算搜索方向 d_k = -H_k g_k
4.     线搜索: 选择 α_k 满足Wolfe条件
5.     x_{k+1} = x_k + α_k d_k
6.     s_k = x_{k+1} - x_k
7.     y_k = ∇f(x_{k+1}) - ∇f(x_k)
8.     更新 H_{k+1} (BFGS公式)
9. end for
```

---

### 2. L-BFGS算法

**Limited-memory BFGS**:

**问题**: BFGS需要存储 $n \times n$ 矩阵，对于大规模问题不可行。

**解决**: 只存储最近 $m$ 次迭代的 $(s_i, y_i)$ 对（通常 $m = 5 \sim 20$）。

**两循环递归**:

不显式存储 $H_k$，而是通过递归计算 $H_k g_k$。

**存储需求**: $O(mn)$ vs $O(n^2)$

---

**算法 1.3 (L-BFGS两循环递归)**:

```text
输入: 梯度 g, 历史 {(s_i, y_i)}_{i=k-m}^{k-1}
输出: H_k g

1. q = g
2. for i = k-1, k-2, ..., k-m do
3.     α_i = ρ_i s_i^T q, where ρ_i = 1/(y_i^T s_i)
4.     q = q - α_i y_i
5. end for
6. r = H_0 q  (通常 H_0 = γI, γ = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1})
7. for i = k-m, k-m+1, ..., k-1 do
8.     β = ρ_i y_i^T r
9.     r = r + s_i (α_i - β)
10. end for
11. return r
```

---

### 3. DFP算法

**Davidon-Fletcher-Powell算法**:

**更新公式**:

$$
H_{k+1} = H_k - \frac{H_k y_k y_k^T H_k}{y_k^T H_k y_k} + \frac{s_k s_k^T}{y_k^T s_k}
$$

**与BFGS的关系**: DFP是BFGS的对偶形式。

---

## 🔬 共轭梯度法

### 1. 线性共轭梯度

**问题**: 求解线性系统 $Ax = b$，其中 $A$ 是对称正定矩阵。

等价于最小化二次函数：

$$
f(x) = \frac{1}{2} x^T A x - b^T x
$$

**共轭方向**:

方向 $d_i$ 和 $d_j$ 关于 $A$ 共轭，如果：

$$
d_i^T A d_j = 0, \quad i \neq j
$$

---

**算法 1.4 (线性共轭梯度)**:

```text
输入: A, b, x₀
输出: 解 x

1. r₀ = b - Ax₀
2. d₀ = r₀
3. for k = 0, 1, 2, ... do
4.     α_k = (r_k^T r_k) / (d_k^T A d_k)
5.     x_{k+1} = x_k + α_k d_k
6.     r_{k+1} = r_k - α_k A d_k
7.     if ||r_{k+1}|| < ε then
8.         return x_{k+1}
9.     β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
10.    d_{k+1} = r_{k+1} + β_k d_k
11. end for
```

**收敛性**: 最多 $n$ 步精确收敛（理论上）。

---

### 2. 非线性共轭梯度

**扩展到非线性优化**:

$$
x_{k+1} = x_k + \alpha_k d_k
$$

$$
d_k = -\nabla f(x_k) + \beta_k d_{k-1}
$$

**$\beta_k$ 的选择**:

- **Fletcher-Reeves**:
  $$
  \beta_k^{FR} = \frac{\|\nabla f(x_k)\|^2}{\|\nabla f(x_{k-1})\|^2}
  $$

- **Polak-Ribière**:
  $$
  \beta_k^{PR} = \frac{\nabla f(x_k)^T (\nabla f(x_k) - \nabla f(x_{k-1}))}{\|\nabla f(x_{k-1})\|^2}
  $$

- **Hestenes-Stiefel**:
  $$
  \beta_k^{HS} = \frac{\nabla f(x_k)^T (\nabla f(x_k) - \nabla f(x_{k-1}))}{d_{k-1}^T (\nabla f(x_k) - \nabla f(x_{k-1}))}
  $$

---

### 3. 预条件共轭梯度

**思想**: 用预条件矩阵 $M$ 加速收敛。

求解 $M^{-1} A x = M^{-1} b$ 而不是 $Ax = b$。

**选择 $M$**:

- $M \approx A$（易于求逆）
- 改善条件数 $\kappa(M^{-1}A) < \kappa(A)$

---

## 💡 Gauss-Newton法

### 1. 基本Gauss-Newton法

**问题**: 非线性最小二乘

$$
\min_x \quad f(x) = \frac{1}{2} \|r(x)\|^2 = \frac{1}{2} \sum_{i=1}^m r_i(x)^2
$$

其中 $r(x) = [r_1(x), \ldots, r_m(x)]^T$ 是残差向量。

**梯度**:

$$
\nabla f(x) = J(x)^T r(x)
$$

其中 $J(x)$ 是残差的Jacobian矩阵。

**Hessian近似**:

$$
\nabla^2 f(x) = J(x)^T J(x) + \sum_{i=1}^m r_i(x) \nabla^2 r_i(x)
$$

忽略二阶项：

$$
H \approx J(x)^T J(x)
$$

**Gauss-Newton方向**:

$$
d = -(J^T J)^{-1} J^T r
$$

---

### 2. Levenberg-Marquardt算法

**改进**: 结合梯度下降和Gauss-Newton。

**更新规则**:

$$
(J^T J + \lambda I) d = -J^T r
$$

- $\lambda$ 大：接近梯度下降
- $\lambda$ 小：接近Gauss-Newton

**自适应调整 $\lambda$**:

- 如果迭代成功：减小 $\lambda$
- 如果迭代失败：增大 $\lambda$

---

## 🎨 在深度学习中的应用

### 1. 自然梯度

**Fisher信息矩阵**:

$$
F = \mathbb{E}[\nabla \log p(y|x; \theta) \nabla \log p(y|x; \theta)^T]
$$

**自然梯度**:

$$
\tilde{\nabla} L = F^{-1} \nabla L
$$

**更新规则**:

$$
\theta_{t+1} = \theta_t - \eta F^{-1} \nabla L(\theta_t)
$$

**优势**: 参数空间不变性。

---

### 2. K-FAC

**Kronecker-Factored Approximate Curvature**:

**思想**: 用Kronecker积近似Fisher信息矩阵。

对于层 $l$：

$$
F_l \approx A_l \otimes G_l
$$

其中：

- $A_l$: 激活的二阶统计量
- $G_l$: 梯度的二阶统计量

**计算复杂度**: $O(n^{3/2})$ vs $O(n^3)$

---

### 3. Shampoo

**Scalable Higher-order Adaptive Methods for Parallel and Distributed Optimization**:

**思想**: 对每层使用独立的预条件矩阵。

**更新规则**:

$$
\theta_{t+1} = \theta_t - \eta (L_t \otimes R_t)^{-1/4} \nabla L(\theta_t)
$$

其中 $L_t$ 和 $R_t$ 是左右预条件矩阵。

---

## 🔧 实用技巧

### 1. Hessian近似

**有限差分**:

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} \approx \frac{f(x + \epsilon e_i + \epsilon e_j) - f(x + \epsilon e_i) - f(x + \epsilon e_j) + f(x)}{\ epsilon^2}
$$

**自动微分**: 使用反向模式自动微分计算Hessian-向量积。

---

### 2. 线搜索策略

**Wolfe条件**:

1. **充分下降条件** (Armijo):
   $$
   f(x_k + \alpha d_k) \leq f(x_k) + c_1 \alpha \nabla f(x_k)^T d_k
   $$

2. **曲率条件**:
   $$
   \nabla f(x_k + \alpha d_k)^T d_k \geq c_2 \nabla f(x_k)^T d_k
   $$

其中 $0 < c_1 < c_2 < 1$（通常 $c_1 = 10^{-4}$, $c_2 = 0.9$）。

---

### 3. 收敛性分析

**收敛速度比较**:

| 方法 | 收敛速度 | 每步成本 |
|------|----------|----------|
| 梯度下降 | 线性 | $O(n)$ |
| 共轭梯度 | 超线性 | $O(n)$ |
| BFGS | 超线性 | $O(n^2)$ |
| L-BFGS | 超线性 | $O(mn)$ |
| Newton | 二次 | $O(n^3)$ |

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search
from scipy.linalg import cho_factor, cho_solve

# 1. Newton法
def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    """Newton法"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for k in range(max_iter):
        g = grad_f(x)
        
        if np.linalg.norm(g) < tol:
            break
        
        H = hess_f(x)
        
        # 求解Newton方程: H * d = -g
        try:
            d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("Hessian is singular, using gradient descent")
            d = -g
        
        # 线搜索
        alpha = backtracking_line_search(f, grad_f, x, d)
        
        x = x + alpha * d
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


def backtracking_line_search(f, grad_f, x, d, alpha=1.0, rho=0.5, c=1e-4):
    """Armijo回溯线搜索"""
    f_x = f(x)
    grad_f_x = grad_f(x)
    
    while f(x + alpha * d) > f_x + c * alpha * np.dot(grad_f_x, d):
        alpha *= rho
        if alpha < 1e-10:
            break
    
    return alpha


# 2. BFGS算法
def bfgs(f, grad_f, x0, max_iter=100, tol=1e-6):
    """BFGS算法"""
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # 初始Hessian逆近似
    trajectory = [x.copy()]
    
    for k in range(max_iter):
        g = grad_f(x)
        
        if np.linalg.norm(g) < tol:
            break
        
        # 搜索方向
        d = -H @ g
        
        # 线搜索
        alpha = backtracking_line_search(f, grad_f, x, d)
        
        # 更新
        s = alpha * d
        x_new = x + s
        y = grad_f(x_new) - g
        
        # BFGS更新H
        rho = 1.0 / (y @ s)
        if rho > 0:  # 确保正定性
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        x = x_new
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


# 3. L-BFGS算法
class LBFGS:
    """L-BFGS算法"""
    
    def __init__(self, m=10):
        self.m = m  # 历史大小
        self.s_list = []
        self.y_list = []
        
    def two_loop_recursion(self, g):
        """两循环递归计算H*g"""
        q = g.copy()
        alpha_list = []
        
        # 第一个循环
        for s, y in zip(reversed(self.s_list), reversed(self.y_list)):
            rho = 1.0 / (y @ s)
            alpha = rho * (s @ q)
            q = q - alpha * y
            alpha_list.append(alpha)
        
        alpha_list.reverse()
        
        # 初始Hessian逆近似
        if len(self.s_list) > 0:
            s = self.s_list[-1]
            y = self.y_list[-1]
            gamma = (s @ y) / (y @ y)
        else:
            gamma = 1.0
        
        r = gamma * q
        
        # 第二个循环
        for s, y, alpha in zip(self.s_list, self.y_list, alpha_list):
            rho = 1.0 / (y @ s)
            beta = rho * (y @ r)
            r = r + s * (alpha - beta)
        
        return r
    
    def update(self, s, y):
        """更新历史"""
        if len(self.s_list) >= self.m:
            self.s_list.pop(0)
            self.y_list.pop(0)
        
        self.s_list.append(s)
        self.y_list.append(y)
    
    def optimize(self, f, grad_f, x0, max_iter=100, tol=1e-6):
        """优化"""
        x = x0.copy()
        trajectory = [x.copy()]
        
        for k in range(max_iter):
            g = grad_f(x)
            
            if np.linalg.norm(g) < tol:
                break
            
            # 计算搜索方向
            d = -self.two_loop_recursion(g)
            
            # 线搜索
            alpha = backtracking_line_search(f, grad_f, x, d)
            
            # 更新
            s = alpha * d
            x_new = x + s
            y = grad_f(x_new) - g
            
            # 更新历史
            if y @ s > 0:  # 确保正定性
                self.update(s, y)
            
            x = x_new
            trajectory.append(x.copy())
        
        return x, np.array(trajectory)


# 4. 共轭梯度法
def conjugate_gradient(A, b, x0=None, max_iter=None, tol=1e-6):
    """线性共轭梯度法"""
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    if max_iter is None:
        max_iter = n
    
    r = b - A @ x
    d = r.copy()
    
    trajectory = [x.copy()]
    
    for k in range(max_iter):
        if np.linalg.norm(r) < tol:
            break
        
        Ad = A @ d
        alpha = (r @ r) / (d @ Ad)
        
        x = x + alpha * d
        r_new = r - alpha * Ad
        
        beta = (r_new @ r_new) / (r @ r)
        d = r_new + beta * d
        
        r = r_new
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


# 5. Gauss-Newton法
def gauss_newton(residual, jacobian, x0, max_iter=100, tol=1e-6):
    """Gauss-Newton法"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for k in range(max_iter):
        r = residual(x)
        J = jacobian(x)
        
        if np.linalg.norm(r) < tol:
            break
        
        # 求解正规方程: (J^T J) d = -J^T r
        d = -np.linalg.solve(J.T @ J, J.T @ r)
        
        # 线搜索
        def f(x):
            return 0.5 * np.sum(residual(x)**2)
        
        def grad_f(x):
            return jacobian(x).T @ residual(x)
        
        alpha = backtracking_line_search(f, grad_f, x, d)
        
        x = x + alpha * d
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)


# 6. 可视化比较
def compare_methods():
    """比较不同优化方法"""
    # Rosenbrock函数
    def f(x):
        return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    
    def grad_f(x):
        return np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])
    
    def hess_f(x):
        return np.array([
            [2 - 400*(x[1] - x[0]**2) + 800*x[0]**2, -400*x[0]],
            [-400*x[0], 200]
        ])
    
    x0 = np.array([-1.0, 1.0])
    
    # 运行不同方法
    methods = {
        'Newton': lambda: newton_method(f, grad_f, hess_f, x0, max_iter=50),
        'BFGS': lambda: bfgs(f, grad_f, x0, max_iter=50),
        'L-BFGS': lambda: LBFGS(m=5).optimize(f, grad_f, x0, max_iter=50)
    }
    
    # 绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 等高线图
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-0.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([x, y])) for x in x] for y in y])
    
    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    
    colors = ['red', 'blue', 'green']
    for (name, method), color in zip(methods.items(), colors):
        x_opt, traj = method()
        ax1.plot(traj[:, 0], traj[:, 1], 'o-', color=color, 
                label=f'{name} ({len(traj)} iters)', markersize=4, linewidth=2)
        
        # 收敛曲线
        f_vals = [f(x) for x in traj]
        ax2.semilogy(f_vals, 'o-', color=color, label=name, linewidth=2)
    
    ax1.plot(1, 1, 'r*', markersize=15, label='Optimum')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('Optimization Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value')
    ax2.set_title('Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("二阶优化方法示例")
    print("=" * 60 + "\n")
    
    print("比较不同优化方法...")
    compare_methods()
    
    print("\n所有示例完成！")
```

---

## 📚 练习题

### 练习1：Newton法

实现Newton法求解 $f(x) = x^4 - 3x^3 + 2$ 的最小值。

### 练习2：BFGS

使用BFGS算法最小化Rosenbrock函数。

### 练习3：共轭梯度

用共轭梯度法求解线性系统 $Ax = b$，其中 $A$ 是大型稀疏矩阵。

### 练习4：Gauss-Newton

使用Gauss-Newton法拟合非线性模型。

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | EE364B - Convex Optimization II |
| **MIT** | 6.255J - Optimization Methods |
| **CMU** | 10-725 - Convex Optimization |
| **UC Berkeley** | EECS227C - Convex Optimization |

---

## 📖 参考文献

1. **Nocedal & Wright (2006)**. *Numerical Optimization*. Springer.

2. **Boyd & Vandenberghe (2004)**. *Convex Optimization*. Cambridge University Press.

3. **Martens & Grosse (2015)**. *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*. ICML.

4. **Gupta et al. (2018)**. *Shampoo: Preconditioned Stochastic Tensor Optimization*. ICML.

5. **Amari (1998)**. *Natural Gradient Works Efficiently in Learning*. Neural Computation.

---

*最后更新：2025年10月*-
