# 最优传输理论 (Optimal Transport Theory)

> **Moving Probability Distributions Optimally**
>
> 概率分布的最优移动

---

## 目录

- [最优传输理论 (Optimal Transport Theory)](#最优传输理论-optimal-transport-theory)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 Monge问题](#-monge问题)
    - [1. 经典Monge问题](#1-经典monge问题)
    - [2. Monge问题的困难](#2-monge问题的困难)
  - [📊 Kantorovich松弛](#-kantorovich松弛)
    - [1. Kantorovich问题](#1-kantorovich问题)
    - [2. 对偶问题](#2-对偶问题)
    - [3. Kantorovich-Rubinstein定理](#3-kantorovich-rubinstein定理)
  - [🔬 Wasserstein距离](#-wasserstein距离)
    - [1. Wasserstein-p距离](#1-wasserstein-p距离)
    - [2. Wasserstein-1距离](#2-wasserstein-1距离)
    - [3. Wasserstein-2距离](#3-wasserstein-2距离)
  - [💡 最优传输映射](#-最优传输映射)
    - [1. Brenier定理](#1-brenier定理)
    - [2. 凸势函数](#2-凸势函数)
    - [3. McCann插值](#3-mccann插值)
  - [🎨 Wasserstein梯度流](#-wasserstein梯度流)
    - [1. 概率测度空间上的梯度流](#1-概率测度空间上的梯度流)
    - [2. JKO格式](#2-jko格式)
    - [3. 偏微分方程与梯度流](#3-偏微分方程与梯度流)
  - [🔧 计算方法](#-计算方法)
    - [1. Sinkhorn算法](#1-sinkhorn算法)
    - [2. 熵正则化](#2-熵正则化)
    - [3. 离散最优传输](#3-离散最优传输)
  - [💻 Python实现](#-python实现)
  - [🎓 在AI中的应用](#-在ai中的应用)
    - [1. Wasserstein GAN](#1-wasserstein-gan)
    - [2. 域适应](#2-域适应)
    - [3. 生成模型评估](#3-生成模型评估)
  - [📚 练习题](#-练习题)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**最优传输理论**研究如何以最小代价将一个概率分布转换为另一个。

**核心问题**:

```text
给定两个概率分布 μ 和 ν，如何以最优方式将 μ 转换为 ν？

关键要素:
├─ 源分布: μ (初始分布)
├─ 目标分布: ν (目标分布)
├─ 代价函数: c(x, y) (从x移动到y的代价)
└─ 传输方案: π 或 T (如何移动)

应用:
├─ 机器学习: Wasserstein GAN, 域适应
├─ 计算机视觉: 图像处理, 形状匹配
├─ 经济学: 资源分配
└─ 物理学: 流体力学, 热传导
```

---

## 🎯 Monge问题

### 1. 经典Monge问题

**问题** (Monge, 1781):

给定两个概率测度 $\mu, \nu$ 在 $\mathbb{R}^d$ 上，以及代价函数 $c: \mathbb{R}^d \times \mathbb{R}^d \to [0, \infty)$。

找传输映射 $T: \mathbb{R}^d \to \mathbb{R}^d$ 使得：

$$
T_\# \mu = \nu
$$

即 $\mu(T^{-1}(B)) = \nu(B)$ 对所有可测集 $B$。

**目标**: 最小化总代价

$$
\min_{T: T_\# \mu = \nu} \int_{\mathbb{R}^d} c(x, T(x)) d\mu(x)
$$

---

**例 1.1 (一维情况)**:

设 $\mu, \nu$ 是 $\mathbb{R}$ 上的概率测度，$c(x, y) = |x - y|$。

**最优传输映射**: $T(x) = F_\nu^{-1}(F_\mu(x))$

其中 $F_\mu, F_\nu$ 是累积分布函数。

---

### 2. Monge问题的困难

**问题**:

1. **存在性**: 传输映射 $T$ 可能不存在
   - 例如: $\mu = \delta_0$ (点质量), $\nu = \frac{1}{2}(\delta_{-1} + \delta_1)$

2. **唯一性**: 即使存在，也可能不唯一

3. **非凸性**: 可行集不是凸集

---

## 📊 Kantorovich松弛

### 1. Kantorovich问题

**Kantorovich松弛** (1942):

不要求传输由映射给出，而是考虑**传输计划** $\pi \in \Pi(\mu, \nu)$，其中

$$
\Pi(\mu, \nu) = \{\pi \in \mathcal{P}(\mathbb{R}^d \times \mathbb{R}^d): \pi_1 = \mu, \pi_2 = \nu\}
$$

这里 $\pi_1, \pi_2$ 是边缘分布。

**Kantorovich问题**:

$$
\min_{\pi \in \Pi(\mu, \nu)} \int_{\mathbb{R}^d \times \mathbb{R}^d} c(x, y) d\pi(x, y)
$$

---

**优势**:

1. **存在性**: 在温和条件下，最优 $\pi$ 总是存在
2. **凸性**: $\Pi(\mu, \nu)$ 是凸集
3. **包含Monge**: 如果 $T$ 是Monge最优，则 $\pi = (id \times T)_\# \mu$ 是Kantorovich最优

---

### 2. 对偶问题

**Kantorovich对偶**:

$$
\sup_{\phi, \psi} \left\{\int \phi d\mu + \int \psi d\nu: \phi(x) + \psi(y) \leq c(x, y)\right\}
$$

**定理 2.1 (强对偶性)**:

在温和条件下，

$$
\min_{\pi \in \Pi(\mu, \nu)} \int c d\pi = \sup_{\phi, \psi} \left\{\int \phi d\mu + \int \psi d\nu: \phi \oplus \psi \leq c\right\}
$$

---

**c-变换**:

对于 $\phi: \mathbb{R}^d \to \mathbb{R}$，定义 **c-变换**:

$$
\phi^c(y) = \inf_{x} \{c(x, y) - \phi(x)\}
$$

**性质**: $(\phi^c)^c \geq \phi$，且 $\phi^c \oplus \phi \leq c$。

---

### 3. Kantorovich-Rubinstein定理

**定理 2.2 (Kantorovich-Rubinstein)**:

对于 $c(x, y) = \|x - y\|$，

$$
W_1(\mu, \nu) = \sup_{\|f\|_L \leq 1} \left\{\int f d\mu - \int f d\nu\right\}
$$

其中 $\|f\|_L = \sup_{x \neq y} \frac{|f(x) - f(y)|}{\|x - y\|}$ 是Lipschitz常数。

**意义**: Wasserstein-1距离可以通过Lipschitz函数计算。

---

## 🔬 Wasserstein距离

### 1. Wasserstein-p距离

**定义 3.1 (Wasserstein-p距离)**:

对于 $p \geq 1$，

$$
W_p(\mu, \nu) = \left(\inf_{\pi \in \Pi(\mu, \nu)} \int \|x - y\|^p d\pi(x, y)\right)^{1/p}
$$

**性质**:

1. **度量**: $W_p$ 是 $\mathcal{P}_p(\mathbb{R}^d)$ 上的度量
2. **弱收敛**: $W_p(\mu_n, \mu) \to 0 \Leftrightarrow \mu_n \rightharpoonup \mu$ 且 $\int \|x\|^p d\mu_n \to \int \|x\|^p d\mu$

---

### 2. Wasserstein-1距离

**Earth Mover's Distance (EMD)**:

$$
W_1(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \int \|x - y\| d\pi(x, y)
$$

**对偶形式** (Kantorovich-Rubinstein):

$$
W_1(\mu, \nu) = \sup_{\|f\|_L \leq 1} \left|\int f d\mu - \int f d\nu\right|
$$

---

**例 3.1 (离散分布)**:

设 $\mu = \sum_{i=1}^n a_i \delta_{x_i}$, $\nu = \sum_{j=1}^m b_j \delta_{y_j}$，其中 $\sum a_i = \sum b_j = 1$。

$$
W_1(\mu, \nu) = \min_{\pi_{ij}} \sum_{i,j} \pi_{ij} \|x_i - y_j\|
$$

约束: $\sum_j \pi_{ij} = a_i$, $\sum_i \pi_{ij} = b_j$, $\pi_{ij} \geq 0$。

这是**线性规划**问题。

---

### 3. Wasserstein-2距离

**定义**:

$$
W_2(\mu, \nu) = \left(\inf_{\pi \in \Pi(\mu, \nu)} \int \|x - y\|^2 d\pi(x, y)\right)^{1/2}
$$

**特殊性质**:

- 与**黎曼几何**联系紧密
- 存在唯一最优传输映射 (在绝对连续情况下)

---

**定理 3.1 (Wasserstein-2的性质)**:

1. **三角不等式**: $W_2(\mu, \nu) \leq W_2(\mu, \rho) + W_2(\rho, \nu)$

2. **位移凸性**: 泛函 $F[\mu] = \int f d\mu$ 在 $W_2$ 意义下是凸的，如果 $f$ 是凸的

3. **测地线**: $\mu_t = ((1-t)id + tT)_\# \mu$ 是 $\mu$ 到 $\nu = T_\# \mu$ 的测地线

---

## 💡 最优传输映射

### 1. Brenier定理

**定理 4.1 (Brenier定理)**:

设 $\mu, \nu$ 是 $\mathbb{R}^d$ 上的概率测度，$\mu$ 绝对连续。

则存在唯一的凸函数 $\phi: \mathbb{R}^d \to \mathbb{R}$ (差一个常数) 使得：

$$
T(x) = \nabla \phi(x)
$$

是从 $\mu$ 到 $\nu$ 的最优传输映射 (对于代价 $c(x, y) = \|x - y\|^2$)。

**意义**: 最优传输映射是梯度映射！

---

**证明思路**:

1. **存在性**: 通过对偶问题
2. **唯一性**: 利用严格凸性
3. **梯度结构**: 利用最优性条件

---

### 2. 凸势函数

**Monge-Ampère方程**:

设 $\mu = \rho dx$, $\nu = \sigma dy$，$T = \nabla \phi$。

则 $T_\# \mu = \nu$ 等价于：

$$
\rho(x) = \sigma(\nabla \phi(x)) \det(D^2 \phi(x))
$$

这是**Monge-Ampère方程**。

---

### 3. McCann插值

**定义 4.2 (位移插值)**:

设 $T$ 是从 $\mu$ 到 $\nu$ 的最优传输映射。

**McCann插值**:

$$
\mu_t = ((1-t)id + tT)_\# \mu, \quad t \in [0, 1]
$$

**性质**:

- $\mu_0 = \mu$, $\mu_1 = \nu$
- $W_2(\mu_t, \mu_s) = |t - s| W_2(\mu, \nu)$
- $\mu_t$ 是 $\mu$ 到 $\nu$ 的测地线

---

## 🎨 Wasserstein梯度流

### 1. 概率测度空间上的梯度流

**泛函的梯度流**:

考虑泛函 $F: \mathcal{P}_2(\mathbb{R}^d) \to \mathbb{R}$。

**Wasserstein梯度流**:

$$
\frac{\partial \mu_t}{\partial t} = -\nabla_{W_2} F[\mu_t]
$$

**形式化**:

$$
\mu_t = \lim_{h \to 0} \frac{1}{h} \arg\min_{\nu} \left\{W_2^2(\mu_t, \nu) + 2h F[\nu]\right\}
$$

---

### 2. JKO格式

**Jordan-Kinderlehrer-Otto (JKO) 格式**:

离散化Wasserstein梯度流：

$$
\mu^{k+1} = \arg\min_{\nu} \left\{\frac{1}{2\tau} W_2^2(\mu^k, \nu) + F[\nu]\right\}
$$

其中 $\tau > 0$ 是时间步长。

**收敛性**: 当 $\tau \to 0$ 时，$\mu^k$ 收敛到梯度流。

---

### 3. 偏微分方程与梯度流

**例 5.1 (热方程)**:

$$
\frac{\partial \rho}{\partial t} = \Delta \rho
$$

是泛函 $F[\rho] = \int \rho \log \rho dx$ (熵) 的Wasserstein梯度流。

---

**例 5.2 (Fokker-Planck方程)**:

$$
\frac{\partial \rho}{\partial t} = \Delta \rho + \nabla \cdot (\rho \nabla V)
$$

是泛函 $F[\rho] = \int (\rho \log \rho + \rho V) dx$ 的Wasserstein梯度流。

---

**例 5.3 (多孔介质方程)**:

$$
\frac{\partial \rho}{\partial t} = \Delta \rho^m
$$

是泛函 $F[\rho] = \int \frac{\rho^m}{m-1} dx$ 的Wasserstein梯度流。

---

## 🔧 计算方法

### 1. Sinkhorn算法

**熵正则化最优传输**:

$$
\min_{\pi \in \Pi(\mu, \nu)} \left\{\int c d\pi + \epsilon H(\pi | \mu \otimes \nu)\right\}
$$

其中 $H(\pi | \mu \otimes \nu) = \int \log \frac{d\pi}{d(\mu \otimes \nu)} d\pi$ 是相对熵。

---

**Sinkhorn算法**:

对于离散分布 $\mu = \sum a_i \delta_{x_i}$, $\nu = \sum b_j \delta_{y_j}$：

1. 初始化 $u^{(0)} = \mathbf{1}$, $v^{(0)} = \mathbf{1}$
2. 迭代:
   $$
   u^{(k+1)}_i = \frac{a_i}{\sum_j K_{ij} v^{(k)}_j}
   $$
   $$
   v^{(k+1)}_j = \frac{b_j}{\sum_i K_{ij} u^{(k+1)}_i}
   $$

其中 $K_{ij} = e^{-c(x_i, y_j)/\epsilon}$。

**收敛性**: 指数收敛到最优解。

---

### 2. 熵正则化

**优势**:

1. **平滑性**: 最优 $\pi$ 绝对连续
2. **计算效率**: Sinkhorn算法快速
3. **可微性**: 对参数可微

**劣势**:

- 引入偏差 (bias)
- 需要选择 $\epsilon$

---

### 3. 离散最优传输

**线性规划**:

$$
\min_{\pi} \sum_{i,j} c_{ij} \pi_{ij}
$$

约束: $\sum_j \pi_{ij} = a_i$, $\sum_i \pi_{ij} = b_j$, $\pi_{ij} \geq 0$。

**求解器**: 网络单纯形法、内点法。

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

# 1. 精确最优传输 (线性规划)
def optimal_transport_lp(a, b, C):
    """
    精确最优传输 (线性规划)

    a: 源分布 (n,)
    b: 目标分布 (m,)
    C: 代价矩阵 (n, m)
    """
    n, m = C.shape

    # 线性规划: min c^T x, s.t. A_eq x = b_eq, x >= 0
    c = C.flatten()

    # 约束: 行和 = a, 列和 = b
    A_eq = np.zeros((n + m, n * m))
    b_eq = np.concatenate([a, b])

    for i in range(n):
        A_eq[i, i*m:(i+1)*m] = 1

    for j in range(m):
        A_eq[n+j, j::m] = 1

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

    if result.success:
        return result.x.reshape(n, m), result.fun
    else:
        raise ValueError("Optimization failed")


# 2. Sinkhorn算法
def sinkhorn(a, b, C, epsilon=0.1, max_iter=1000, tol=1e-9):
    """
    Sinkhorn算法 (熵正则化最优传输)

    a: 源分布 (n,)
    b: 目标分布 (m,)
    C: 代价矩阵 (n, m)
    epsilon: 熵正则化参数
    """
    n, m = C.shape

    # K = exp(-C/epsilon)
    K = np.exp(-C / epsilon)

    # 初始化
    u = np.ones(n)
    v = np.ones(m)

    for _ in range(max_iter):
        u_old = u.copy()

        # 更新 u
        u = a / (K @ v)

        # 更新 v
        v = b / (K.T @ u)

        # 检查收敛
        if np.linalg.norm(u - u_old) < tol:
            break

    # 计算传输计划
    pi = np.diag(u) @ K @ np.diag(v)

    # 计算代价
    cost = np.sum(pi * C)

    return pi, cost


# 3. Wasserstein距离计算
def wasserstein_distance(X, Y, a=None, b=None, p=2, method='sinkhorn', **kwargs):
    """
    计算Wasserstein距离

    X: 源样本 (n, d)
    Y: 目标样本 (m, d)
    a: 源权重 (n,), 默认均匀
    b: 目标权重 (m,), 默认均匀
    p: Wasserstein-p距离
    method: 'lp' 或 'sinkhorn'
    """
    n, m = len(X), len(Y)

    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m

    # 计算代价矩阵
    C = cdist(X, Y, metric='euclidean') ** p

    # 求解最优传输
    if method == 'lp':
        pi, cost = optimal_transport_lp(a, b, C)
    elif method == 'sinkhorn':
        pi, cost = sinkhorn(a, b, C, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    return cost ** (1/p), pi


# 4. Wasserstein重心
def wasserstein_barycenter(distributions, weights=None, epsilon=0.1, max_iter=100):
    """
    计算Wasserstein重心

    distributions: 分布列表 [(X_1, a_1), (X_2, a_2), ...]
    weights: 权重 (k,)
    """
    k = len(distributions)

    if weights is None:
        weights = np.ones(k) / k

    # 初始化重心 (使用第一个分布)
    X_bar, a_bar = distributions[0]

    for _ in range(max_iter):
        # 计算到每个分布的传输计划
        plans = []
        for (X, a), w in zip(distributions, weights):
            _, pi = wasserstein_distance(X_bar, X, a_bar, a, method='sinkhorn', epsilon=epsilon)
            plans.append(pi)

        # 更新重心
        # (简化版: 这里应该用更复杂的算法)
        break

    return X_bar, a_bar


# 5. 可视化
def visualize_optimal_transport():
    """可视化最优传输"""
    np.random.seed(42)

    # 生成两个2D分布
    n, m = 20, 30
    X = np.random.randn(n, 2)
    Y = np.random.randn(m, 2) + np.array([3, 0])

    a = np.ones(n) / n
    b = np.ones(m) / m

    # 计算最优传输
    C = cdist(X, Y, metric='euclidean') ** 2
    pi_lp, cost_lp = optimal_transport_lp(a, b, C)
    pi_sink, cost_sink = sinkhorn(a, b, C, epsilon=0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 精确最优传输
    ax1.scatter(X[:, 0], X[:, 1], c='blue', s=100, alpha=0.6, label='Source')
    ax1.scatter(Y[:, 0], Y[:, 1], c='red', s=100, alpha=0.6, label='Target')

    for i in range(n):
        for j in range(m):
            if pi_lp[i, j] > 1e-6:
                ax1.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]],
                        'k-', alpha=pi_lp[i, j] * n * 2, linewidth=1)

    ax1.set_title(f'Exact OT (LP)\nCost: {cost_lp:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sinkhorn算法
    ax2.scatter(X[:, 0], X[:, 1], c='blue', s=100, alpha=0.6, label='Source')
    ax2.scatter(Y[:, 0], Y[:, 1], c='red', s=100, alpha=0.6, label='Target')

    for i in range(n):
        for j in range(m):
            if pi_sink[i, j] > 1e-6:
                ax2.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]],
                        'k-', alpha=pi_sink[i, j] * n * 2, linewidth=1)

    ax2.set_title(f'Sinkhorn OT (ε=0.1)\nCost: {cost_sink:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()


# 6. Wasserstein GAN损失
def wasserstein_gan_loss(real_samples, fake_samples, critic):
    """
    Wasserstein GAN损失

    real_samples: 真实样本
    fake_samples: 生成样本
    critic: 判别器 (1-Lipschitz函数)
    """
    # W1距离的对偶形式
    real_scores = critic(real_samples)
    fake_scores = critic(fake_samples)

    # Wasserstein距离估计
    w_distance = np.mean(real_scores) - np.mean(fake_scores)

    return w_distance


def demo_optimal_transport():
    """最优传输示例"""
    print("=" * 60)
    print("最优传输理论示例")
    print("=" * 60 + "\n")

    # 1. 一维情况
    print("1. 一维Wasserstein距离")
    X = np.array([[0], [1], [2]])
    Y = np.array([[0.5], [1.5], [2.5]])

    w_dist, _ = wasserstein_distance(X, Y, p=1, method='lp')
    print(f"   W1距离: {w_dist:.4f}")

    # 2. 二维情况
    print("\n2. 二维Wasserstein距离")
    np.random.seed(42)
    X = np.random.randn(10, 2)
    Y = np.random.randn(10, 2) + 2

    w2_lp, _ = wasserstein_distance(X, Y, p=2, method='lp')
    w2_sink, _ = wasserstein_distance(X, Y, p=2, method='sinkhorn', epsilon=0.1)

    print(f"   W2距离 (LP): {w2_lp:.4f}")
    print(f"   W2距离 (Sinkhorn): {w2_sink:.4f}")

    # 3. 可视化
    print("\n3. 生成可视化...")
    visualize_optimal_transport()

    print("\n所有示例完成！")


if __name__ == "__main__":
    demo_optimal_transport()
```

---

## 🎓 在AI中的应用

### 1. Wasserstein GAN

**问题**: 传统GAN训练不稳定。

**Wasserstein GAN (WGAN)**:

**生成器损失**:

$$
\min_G -\mathbb{E}_{z \sim p_z}[D(G(z))]
$$

**判别器损失**:

$$
\max_{D: \|D\|_L \leq 1} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

**优势**:

- 训练稳定
- 有意义的损失曲线
- 不需要平衡生成器和判别器

---

### 2. 域适应

**问题**: 源域和目标域分布不同。

**最优传输域适应**:

最小化源域和目标域之间的Wasserstein距离：

$$
\min_\theta W_2(p_{source}^{(f_\theta)}, p_{target}^{(f_\theta)})
$$

其中 $f_\theta$ 是特征提取器。

---

### 3. 生成模型评估

**Fréchet Inception Distance (FID)**:

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

这是高斯分布之间的Wasserstein-2距离。

---

### 4. 最优传输在图像处理中的应用

**颜色迁移**：

使用最优传输将一幅图像的颜色分布迁移到另一幅图像：

$$
T^* = \arg\min_T \int c(x, T(x)) d\mu(x) \quad \text{s.t.} \quad T_\# \mu = \nu
$$

其中 $\mu$ 是源图像的颜色分布，$\nu$ 是目标颜色分布。

**图像修复**：

使用最优传输填充图像缺失区域，保持统计一致性。

**风格迁移**：

将一幅图像的风格（纹理分布）传输到另一幅图像，同时保持内容。

**Python实现示例**：

```python
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
import numpy as np

def optimal_transport_1d(source, target):
    """
    一维最优传输（精确解）

    参数:
        source: 源分布（直方图）
        target: 目标分布（直方图）
    """
    n = len(source)
    m = len(target)

    # 成本矩阵（L2距离）
    C = cdist(source.reshape(-1, 1), target.reshape(-1, 1))**2

    # 线性规划：最小化 <C, P>
    # 约束：P 1 = source, P^T 1 = target
    c = C.flatten()

    # 约束矩阵
    A_eq = []
    b_eq = []

    # 行约束：每行和为source
    for i in range(n):
        row = np.zeros(n * m)
        row[i*m:(i+1)*m] = 1
        A_eq.append(row)
        b_eq.append(source[i])

    # 列约束：每列和为target
    for j in range(m):
        col = np.zeros(n * m)
        col[j::m] = 1
        A_eq.append(col)
        b_eq.append(target[j])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # 求解
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
    P = result.x.reshape(n, m)

    # 计算Wasserstein距离
    W2 = np.sum(P * C)

    return P, W2
```

---

### 5. 最优传输在强化学习中的应用

**分布强化学习**：

在分布强化学习中，值函数不再是标量，而是分布。使用Wasserstein距离比较值分布：

$$
Z(s, a) \sim \text{Value Distribution}
$$

**目标**：最小化预测分布和真实分布之间的Wasserstein距离。

**优势**：

- **风险敏感**：可以建模风险偏好
- **不确定性量化**：分布提供不确定性信息
- **样本效率**：更好的样本利用

**Python实现示例**：

```python
import torch
import torch.nn as nn

class DistributionalDQN(nn.Module):
    """分布DQN"""
    def __init__(self, state_dim, action_dim, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta = (v_max - v_min) / (n_atoms - 1)

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * n_atoms)
        )

    def forward(self, state):
        logits = self.network(state)
        logits = logits.view(-1, self.action_dim, self.n_atoms)
        return torch.softmax(logits, dim=-1)

    def wasserstein_loss(self, pred_dist, target_dist, rewards, dones, gamma=0.99):
        """Wasserstein损失"""
        # 投影目标分布
        target_support = rewards + gamma * (1 - dones) * self.get_support()

        # 计算Wasserstein距离
        # 这里使用简化的1-Wasserstein距离（累积分布函数差）
        loss = 0
        for i in range(self.n_atoms):
            cdf_pred = torch.cumsum(pred_dist, dim=-1)
            cdf_target = torch.cumsum(target_dist, dim=-1)
            loss += torch.abs(cdf_pred - cdf_target).mean()

        return loss

    def get_support(self):
        """获取支持点"""
        return torch.linspace(self.v_min, self.v_max, self.n_atoms)
```

---

### 6. 最优传输在数据增强中的应用

**语义数据增强**：

使用最优传输生成语义一致的数据增强样本：

$$
\min_T \int c(x, T(x)) d\mu(x) + \lambda \mathcal{L}_{\text{task}}(T_\# \mu)
$$

其中 $\mathcal{L}_{\text{task}}$ 是任务损失（如分类损失）。

**优势**：

- **保持语义**：增强样本保持原始语义
- **多样性**：生成多样化的训练样本
- **可控性**：通过成本函数控制增强程度

**应用场景**：

- **小样本学习**：在数据稀缺时生成更多样本
- **域适应**：生成目标域风格的样本
- **对抗训练**：生成对抗样本

---

### 7. 最优传输在公平机器学习中的应用

**公平性约束**：

使用最优传输确保不同群体的预测分布相似：

$$
\min_f \mathcal{L}(f) \quad \text{s.t.} \quad W_1(P_{Y|A=a}, P_{Y|A=b}) \leq \epsilon
$$

其中 $A$ 是敏感属性（如性别、种族），$\epsilon$ 是公平性容忍度。

**优势**：

- **形式化公平性**：将公平性转化为数学约束
- **可解释性**：Wasserstein距离有明确的几何意义
- **灵活性**：可以控制不同公平性定义

**实现方法**：

1. **后处理**：训练后调整预测分布
2. **训练时约束**：在损失函数中加入Wasserstein惩罚项
3. **预处理**：使用最优传输预处理数据

**Python实现示例**：

```python
def fair_classification_loss(predictions, labels, sensitive_attr, lambda_fair=0.1):
    """
    公平分类损失

    参数:
        predictions: 模型预测
        labels: 真实标签
        sensitive_attr: 敏感属性
        lambda_fair: 公平性权重
    """
    # 标准分类损失
    classification_loss = F.cross_entropy(predictions, labels)

    # 公平性损失（Wasserstein距离）
    group_0 = predictions[sensitive_attr == 0]
    group_1 = predictions[sensitive_attr == 1]

    # 计算预测分布
    dist_0 = torch.softmax(group_0, dim=-1).mean(dim=0)
    dist_1 = torch.softmax(group_1, dim=-1).mean(dim=0)

    # Wasserstein-1距离（简化版：L1距离）
    wasserstein_dist = torch.abs(dist_0 - dist_1).sum()

    # 总损失
    total_loss = classification_loss + lambda_fair * wasserstein_dist

    return total_loss
```

---

### 8. 最优传输在推荐系统中的应用

**用户-物品分布对齐**：

使用最优传输对齐不同用户群体的物品分布，提高推荐公平性。

**冷启动问题**：

对于新用户，使用最优传输将相似用户的偏好分布传输过来。

**多样性推荐**：

使用最优传输确保推荐结果的多样性，避免推荐过于集中。

**Python实现示例**：

```python
def optimal_transport_recommendation(user_embeddings, item_embeddings,
                                     user_distribution, item_distribution):
    """
    使用最优传输进行推荐

    参数:
        user_embeddings: 用户嵌入
        item_embeddings: 物品嵌入
        user_distribution: 用户分布（偏好）
        item_distribution: 物品分布（流行度）
    """
    # 计算成本矩阵（用户-物品相似度）
    cost_matrix = cdist(user_embeddings, item_embeddings, metric='cosine')

    # 求解最优传输
    from scipy.optimize import linprog

    # 这里使用Sinkhorn算法（更高效）
    from ot import sinkhorn

    # 熵正则化最优传输
    transport_plan = sinkhorn(user_distribution, item_distribution,
                              cost_matrix, reg=0.1)

    # 推荐：选择传输质量高的物品
    recommendations = transport_plan.argmax(axis=1)

    return recommendations, transport_plan
```

---

## 📚 练习题

**练习1**: 证明Wasserstein-1距离满足三角不等式。

**练习2**: 实现一维情况的精确最优传输。

**练习3**: 比较Sinkhorn算法在不同 $\epsilon$ 下的性能。

**练习4**: 实现简单的Wasserstein GAN。

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | STATS385 - Theories of Deep Learning |
| **MIT** | 18.S096 - Topics in Mathematics with Applications |
| **ENS Paris** | Optimal Transport (Villani) |
| **UC Berkeley** | STAT260 - Mean Field Asymptotics |

---

## 📖 参考文献

1. **Villani, C. (2009)**. *Optimal Transport: Old and New*. Springer.

2. **Santambrogio, F. (2015)**. *Optimal Transport for Applied Mathematicians*. Birkhäuser.

3. **Peyré, G. & Cuturi, M. (2019)**. *Computational Optimal Transport*. Foundations and Trends in Machine Learning.

4. **Arjovsky, M. et al. (2017)**. *Wasserstein Generative Adversarial Networks*. ICML.

5. **Cuturi, M. (2013)**. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NIPS.

---

*最后更新：2025年10月*-
