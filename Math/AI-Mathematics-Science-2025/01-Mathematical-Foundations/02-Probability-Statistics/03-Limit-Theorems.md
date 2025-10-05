# 极限定理 (Limit Theorems)

> **The Foundation of Statistical Inference**
>
> 统计推断的理论基础

---

## 目录

- [极限定理 (Limit Theorems)](#极限定理-limit-theorems)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 收敛性概念](#-收敛性概念)
    - [1. 依概率收敛](#1-依概率收敛)
    - [2. 几乎必然收敛](#2-几乎必然收敛)
    - [3. 依分布收敛](#3-依分布收敛)
    - [4. 收敛性之间的关系](#4-收敛性之间的关系)
  - [📊 大数定律 (Law of Large Numbers)](#-大数定律-law-of-large-numbers)
    - [1. 弱大数定律 (WLLN)](#1-弱大数定律-wlln)
    - [2. 强大数定律 (SLLN)](#2-强大数定律-slln)
    - [3. 应用](#3-应用)
  - [🔬 中心极限定理 (Central Limit Theorem)](#-中心极限定理-central-limit-theorem)
    - [1. 经典中心极限定理](#1-经典中心极限定理)
    - [2. Lindeberg-Lévy定理](#2-lindeberg-lévy定理)
    - [3. Lyapunov定理](#3-lyapunov定理)
    - [4. Berry-Esseen定理](#4-berry-esseen定理)
  - [💡 多元中心极限定理](#-多元中心极限定理)
    - [1. 多元CLT](#1-多元clt)
    - [2. Delta方法](#2-delta方法)
  - [🎨 在机器学习中的应用](#-在机器学习中的应用)
    - [1. 经验风险最小化](#1-经验风险最小化)
    - [2. 参数估计](#2-参数估计)
    - [3. 置信区间](#3-置信区间)
    - [4. 假设检验](#4-假设检验)
    - [5. Bootstrap方法](#5-bootstrap方法)
  - [🔧 高级主题](#-高级主题)
    - [1. 大偏差理论](#1-大偏差理论)
    - [2. 函数型中心极限定理](#2-函数型中心极限定理)
    - [3. 依赖数据的极限定理](#3-依赖数据的极限定理)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：大数定律验证](#练习1大数定律验证)
    - [练习2：中心极限定理](#练习2中心极限定理)
    - [练习3：置信区间](#练习3置信区间)
    - [练习4：Bootstrap](#练习4bootstrap)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**极限定理**描述了大量随机变量的和或平均的渐近行为，是统计推断和机器学习的理论基础。

**为什么极限定理重要**:

```text
统计推断的基础:
├─ 大数定律: 样本均值收敛到总体均值
├─ 中心极限定理: 样本均值的分布趋于正态
├─ 置信区间: 参数估计的不确定性
└─ 假设检验: 统计显著性

机器学习中的应用:
├─ 经验风险最小化
├─ 泛化误差估计
├─ 模型选择
└─ 不确定性量化
```

---

## 🎯 收敛性概念

### 1. 依概率收敛

**定义 1.1 (依概率收敛)**:

随机变量序列 $\{X_n\}$ **依概率收敛**到 $X$，记作 $X_n \xrightarrow{P} X$，如果：

$$
\forall \epsilon > 0, \quad \lim_{n \to \infty} P(|X_n - X| > \epsilon) = 0
$$

**等价表述**:

$$
\lim_{n \to \infty} P(|X_n - X| \leq \epsilon) = 1
$$

**直觉**：$X_n$ 以很大的概率接近 $X$。

---

### 2. 几乎必然收敛

**定义 2.1 (几乎必然收敛)**:

随机变量序列 $\{X_n\}$ **几乎必然收敛**（或**以概率1收敛**）到 $X$，记作 $X_n \xrightarrow{a.s.} X$，如果：

$$
P\left(\lim_{n \to \infty} X_n = X\right) = 1
$$

**等价表述**:

$$
P\left(\{\omega : \lim_{n \to \infty} X_n(\omega) = X(\omega)\}\right) = 1
$$

**直觉**：除了零概率事件外，$X_n$ 的每个样本路径都收敛到 $X$。

---

### 3. 依分布收敛

**定义 3.1 (依分布收敛)**:

随机变量序列 $\{X_n\}$ **依分布收敛**到 $X$，记作 $X_n \xrightarrow{d} X$，如果：

$$
\lim_{n \to \infty} F_n(x) = F(x)
$$

对于 $F$ 的所有连续点 $x$，其中 $F_n$ 和 $F$ 分别是 $X_n$ 和 $X$ 的累积分布函数。

**记号**：也记作 $X_n \Rightarrow X$。

---

### 4. 收敛性之间的关系

**定理 4.1 (收敛性层次)**:

$$
X_n \xrightarrow{a.s.} X \quad \Rightarrow \quad X_n \xrightarrow{P} X \quad \Rightarrow \quad X_n \xrightarrow{d} X
$$

**反向不成立**：

- 依分布收敛 $\not\Rightarrow$ 依概率收敛
- 依概率收敛 $\not\Rightarrow$ 几乎必然收敛

**特殊情况**：如果 $X$ 是常数 $c$，则：

$$
X_n \xrightarrow{d} c \quad \Leftrightarrow \quad X_n \xrightarrow{P} c
$$

---

## 📊 大数定律 (Law of Large Numbers)

### 1. 弱大数定律 (WLLN)

**定理 1.1 (Khinchin弱大数定律)**:

设 $X_1, X_2, \ldots$ 是独立同分布的随机变量，且 $E[X_i] = \mu$ 存在。令：

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i
$$

则：

$$
\bar{X}_n \xrightarrow{P} \mu
$$

**证明思路** (Chebyshev不等式):

假设 $\text{Var}(X_i) = \sigma^2 < \infty$，则：

$$
\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}
$$

由Chebyshev不等式：

$$
P(|\bar{X}_n - \mu| > \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2} \to 0
$$

---

### 2. 强大数定律 (SLLN)

**定理 2.1 (Kolmogorov强大数定律)**:

设 $X_1, X_2, \ldots$ 是独立同分布的随机变量，且 $E[|X_i|] < \infty$，$E[X_i] = \mu$。则：

$$
\bar{X}_n \xrightarrow{a.s.} \mu
$$

**意义**：样本均值以概率1收敛到总体均值。

---

### 3. 应用

**蒙特卡洛积分**:

估计 $\theta = E[g(X)]$：

$$
\hat{\theta}_n = \frac{1}{n} \sum_{i=1}^n g(X_i) \xrightarrow{a.s.} \theta
$$

**经验风险**:

$$
\hat{R}_n(f) = \frac{1}{n} \sum_{i=1}^n L(f(x_i), y_i) \xrightarrow{P} R(f) = E[L(f(X), Y)]
$$

---

## 🔬 中心极限定理 (Central Limit Theorem)

### 1. 经典中心极限定理

**定理 1.1 (经典CLT)**:

设 $X_1, X_2, \ldots$ 是独立同分布的随机变量，$E[X_i] = \mu$，$\text{Var}(X_i) = \sigma^2 < \infty$。令：

$$
Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} = \frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}}
$$

则：

$$
Z_n \xrightarrow{d} N(0, 1)
$$

**等价表述**:

$$
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)
$$

或：

$$
\bar{X}_n \sim \text{AN}\left(\mu, \frac{\sigma^2}{n}\right)
$$

其中 AN 表示"渐近正态"(Asymptotically Normal)。

---

### 2. Lindeberg-Lévy定理

**定理 2.1 (Lindeberg-Lévy CLT)**:

设 $X_1, X_2, \ldots$ 独立同分布，$E[X_i] = \mu$，$\text{Var}(X_i) = \sigma^2 < \infty$。则对于任意 $x \in \mathbb{R}$：

$$
\lim_{n \to \infty} P\left(\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \leq x\right) = \Phi(x)
$$

其中 $\Phi$ 是标准正态分布的累积分布函数。

---

### 3. Lyapunov定理

**定理 3.1 (Lyapunov CLT)**:

设 $X_1, X_2, \ldots$ 独立（不必同分布），$E[X_i] = \mu_i$，$\text{Var}(X_i) = \sigma_i^2$。令：

$$
s_n^2 = \sum_{i=1}^n \sigma_i^2
$$

如果存在 $\delta > 0$ 使得：

$$
\lim_{n \to \infty} \frac{1}{s_n^{2+\delta}} \sum_{i=1}^n E[|X_i - \mu_i|^{2+\delta}] = 0
$$

则：

$$
\frac{\sum_{i=1}^n (X_i - \mu_i)}{s_n} \xrightarrow{d} N(0, 1)
$$

**意义**：允许非同分布的情况。

---

### 4. Berry-Esseen定理

**定理 4.1 (Berry-Esseen定理)**:

设 $X_1, \ldots, X_n$ 独立同分布，$E[X_i] = \mu$，$\text{Var}(X_i) = \sigma^2$，$E[|X_i - \mu|^3] = \rho < \infty$。则：

$$
\sup_x \left|P\left(\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \leq x\right) - \Phi(x)\right| \leq \frac{C\rho}{\sigma^3 \sqrt{n}}
$$

其中 $C$ 是绝对常数（$C \leq 0.4748$）。

**意义**：给出了收敛速度的界，通常是 $O(n^{-1/2})$。

---

## 💡 多元中心极限定理

### 1. 多元CLT

**定理 1.1 (多元CLT)**:

设 $\mathbf{X}_1, \mathbf{X}_2, \ldots$ 是独立同分布的 $d$ 维随机向量，$E[\mathbf{X}_i] = \boldsymbol{\mu}$，$\text{Cov}(\mathbf{X}_i) = \Sigma$（正定）。则：

$$
\sqrt{n}(\bar{\mathbf{X}}_n - \boldsymbol{\mu}) \xrightarrow{d} N(\mathbf{0}, \Sigma)
$$

其中 $\bar{\mathbf{X}}_n = \frac{1}{n} \sum_{i=1}^n \mathbf{X}_i$。

---

### 2. Delta方法

**定理 2.1 (Delta方法)**:

设 $\sqrt{n}(X_n - \theta) \xrightarrow{d} N(0, \sigma^2)$，$g$ 是可微函数且 $g'(\theta) \neq 0$。则：

$$
\sqrt{n}(g(X_n) - g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2)
$$

**多元Delta方法**:

设 $\sqrt{n}(\mathbf{X}_n - \boldsymbol{\theta}) \xrightarrow{d} N(\mathbf{0}, \Sigma)$，$g: \mathbb{R}^d \to \mathbb{R}$ 可微。则：

$$
\sqrt{n}(g(\mathbf{X}_n) - g(\boldsymbol{\theta})) \xrightarrow{d} N(0, \nabla g(\boldsymbol{\theta})^T \Sigma \nabla g(\boldsymbol{\theta}))
$$

**应用**：非线性变换的渐近分布。

---

## 🎨 在机器学习中的应用

### 1. 经验风险最小化

**泛化误差**:

训练误差：

$$
\hat{R}_n(f) = \frac{1}{n} \sum_{i=1}^n L(f(x_i), y_i)
$$

真实误差：

$$
R(f) = E[L(f(X), Y)]
$$

**大数定律**：$\hat{R}_n(f) \xrightarrow{P} R(f)$

**中心极限定理**：

$$
\sqrt{n}(\hat{R}_n(f) - R(f)) \xrightarrow{d} N(0, \sigma^2)
$$

其中 $\sigma^2 = \text{Var}(L(f(X), Y))$。

---

### 2. 参数估计

**最大似然估计 (MLE)**:

$$
\hat{\theta}_n = \arg\max_{\theta} \frac{1}{n} \sum_{i=1}^n \log p(x_i; \theta)
$$

**渐近正态性**:

在正则条件下：

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})
$$

其中 $I(\theta)$ 是Fisher信息矩阵。

---

### 3. 置信区间

**构造置信区间**:

由CLT，对于样本均值：

$$
P\left(\mu - z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \leq \bar{X}_n \leq \mu + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right) \approx 1 - \alpha
$$

**95%置信区间**:

$$
\bar{X}_n \pm 1.96 \frac{\sigma}{\sqrt{n}}
$$

**实践中**：用样本标准差 $s$ 估计 $\sigma$。

---

### 4. 假设检验

**Z检验**:

检验 $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$。

检验统计量：

$$
Z = \frac{\bar{X}_n - \mu_0}{\sigma / \sqrt{n}}
$$

在 $H_0$ 下，$Z \sim N(0, 1)$（渐近地）。

**拒绝域**：$|Z| > z_{\alpha/2}$。

---

### 5. Bootstrap方法

**Bootstrap原理**:

从样本 $\{x_1, \ldots, x_n\}$ 中有放回地抽取 $n$ 个样本，重复 $B$ 次。

**Bootstrap分布**:

$$
\hat{F}_n^* \approx F
$$

**应用**：估计统计量的分布、构造置信区间。

---

## 🔧 高级主题

### 1. 大偏差理论

**Cramér定理**:

设 $X_1, X_2, \ldots$ 独立同分布，矩母函数 $M(t) = E[e^{tX_i}]$ 存在。则对于 $a > \mu$：

$$
\lim_{n \to \infty} \frac{1}{n} \log P(\bar{X}_n \geq a) = -I(a)
$$

其中 $I(a) = \sup_t \{ta - \log M(t)\}$ 是**速率函数**。

**意义**：描述尾部概率的指数衰减速率。

---

### 2. 函数型中心极限定理

**Donsker定理**:

经验过程：

$$
G_n(t) = \sqrt{n}(\hat{F}_n(t) - F(t))
$$

其中 $\hat{F}_n$ 是经验分布函数。

**Donsker定理**：

$$
G_n \Rightarrow B \circ F
$$

其中 $B$ 是布朗桥（Brownian bridge）。

---

### 3. 依赖数据的极限定理

**时间序列**:

对于弱依赖的时间序列（如混合序列），在适当条件下，CLT仍然成立，但方差需要调整：

$$
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2 + 2\sum_{k=1}^\infty \gamma(k))
$$

其中 $\gamma(k) = \text{Cov}(X_0, X_k)$ 是自协方差函数。

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, expon, uniform, chi2

# 1. 大数定律演示
def law_of_large_numbers_demo():
    """大数定律演示"""
    print("=== 大数定律 ===\n")
    
    # 参数
    mu = 5  # 真实均值
    n_max = 10000
    n_trials = 5
    
    plt.figure(figsize=(12, 5))
    
    # 不同分布
    distributions = {
        'Uniform(0, 10)': lambda n: np.random.uniform(0, 10, n),
        'Exponential(λ=0.2)': lambda n: np.random.exponential(5, n),
        'Normal(5, 2)': lambda n: np.random.normal(5, 2, n)
    }
    
    for idx, (name, dist) in enumerate(distributions.items(), 1):
        plt.subplot(1, 3, idx)
        
        for _ in range(n_trials):
            samples = dist(n_max)
            cumulative_mean = np.cumsum(samples) / np.arange(1, n_max + 1)
            plt.plot(cumulative_mean, alpha=0.5, linewidth=0.8)
        
        plt.axhline(y=mu, color='r', linestyle='--', linewidth=2, label=f'True mean = {mu}')
        plt.xlabel('Sample size n')
        plt.ylabel('Sample mean')
        plt.title(f'LLN: {name}')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    # plt.show()
    
    print("大数定律：样本均值收敛到真实均值\n")


# 2. 中心极限定理演示
def central_limit_theorem_demo():
    """中心极限定理演示"""
    print("=== 中心极限定理 ===\n")
    
    # 不同分布
    distributions = {
        'Uniform(0, 1)': (lambda n: np.random.uniform(0, 1, n), 0.5, 1/12),
        'Exponential(λ=1)': (lambda n: np.random.exponential(1, n), 1, 1),
        'Chi-square(df=2)': (lambda n: np.random.chisquare(2, n), 2, 4)
    }
    
    sample_sizes = [1, 5, 30, 100]
    n_samples = 10000
    
    fig, axes = plt.subplots(len(distributions), len(sample_sizes), 
                             figsize=(16, 10))
    
    for i, (dist_name, (dist_func, mu, var)) in enumerate(distributions.items()):
        for j, n in enumerate(sample_sizes):
            # 生成样本均值
            sample_means = np.array([
                np.mean(dist_func(n)) for _ in range(n_samples)
            ])
            
            # 标准化
            z_scores = (sample_means - mu) / np.sqrt(var / n)
            
            # 绘制直方图
            ax = axes[i, j]
            ax.hist(z_scores, bins=50, density=True, alpha=0.7, 
                   edgecolor='black', label='Sample')
            
            # 叠加标准正态分布
            x = np.linspace(-4, 4, 100)
            ax.plot(x, norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
            
            ax.set_xlim(-4, 4)
            ax.set_title(f'{dist_name}\nn={n}')
            ax.grid(True, alpha=0.3)
            
            if j == 0:
                ax.set_ylabel('Density')
            if i == len(distributions) - 1:
                ax.set_xlabel('Standardized mean')
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout()
    # plt.show()
    
    print("中心极限定理：标准化的样本均值趋于标准正态分布\n")


# 3. 收敛速度 - Berry-Esseen
def berry_esseen_demo():
    """Berry-Esseen定理：收敛速度"""
    print("=== Berry-Esseen定理 ===\n")
    
    sample_sizes = [10, 30, 100, 300, 1000]
    n_samples = 10000
    
    # 使用指数分布
    mu = 1
    sigma = 1
    
    max_errors = []
    theoretical_bounds = []
    
    for n in sample_sizes:
        # 生成样本均值
        sample_means = np.array([
            np.mean(np.random.exponential(1, n)) for _ in range(n_samples)
        ])
        
        # 标准化
        z_scores = (sample_means - mu) / (sigma / np.sqrt(n))
        
        # 计算经验CDF
        z_sorted = np.sort(z_scores)
        empirical_cdf = np.arange(1, len(z_sorted) + 1) / len(z_sorted)
        
        # 理论CDF
        theoretical_cdf = norm.cdf(z_sorted)
        
        # 最大误差
        max_error = np.max(np.abs(empirical_cdf - theoretical_cdf))
        max_errors.append(max_error)
        
        # Berry-Esseen界
        rho = 2  # E[|X - mu|^3] for Exp(1)
        C = 0.4748
        bound = C * rho / (sigma**3 * np.sqrt(n))
        theoretical_bounds.append(bound)
        
        print(f"n={n:4d}: 最大误差={max_error:.4f}, Berry-Esseen界={bound:.4f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.loglog(sample_sizes, max_errors, 'bo-', linewidth=2, 
              markersize=8, label='Observed error')
    plt.loglog(sample_sizes, theoretical_bounds, 'r--', linewidth=2, 
              label='Berry-Esseen bound')
    plt.loglog(sample_sizes, [1/np.sqrt(n) for n in sample_sizes], 
              'g:', linewidth=2, label='O(1/√n)')
    plt.xlabel('Sample size n')
    plt.ylabel('Maximum error')
    plt.title('Berry-Esseen Theorem: Convergence Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    print()


# 4. 置信区间
def confidence_interval_demo():
    """置信区间演示"""
    print("=== 置信区间 ===\n")
    
    # 真实参数
    mu_true = 10
    sigma_true = 2
    
    # 样本大小
    n = 30
    
    # 置信水平
    confidence_level = 0.95
    alpha = 1 - confidence_level
    z_critical = norm.ppf(1 - alpha/2)
    
    # 生成多个样本并计算置信区间
    n_experiments = 100
    coverage = 0
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_experiments):
        # 生成样本
        sample = np.random.normal(mu_true, sigma_true, n)
        
        # 样本统计量
        x_bar = np.mean(sample)
        s = np.std(sample, ddof=1)
        
        # 置信区间
        margin = z_critical * s / np.sqrt(n)
        ci_lower = x_bar - margin
        ci_upper = x_bar + margin
        
        # 检查是否覆盖真实值
        covers = ci_lower <= mu_true <= ci_upper
        coverage += covers
        
        # 绘制
        color = 'green' if covers else 'red'
        plt.plot([ci_lower, ci_upper], [i, i], color=color, alpha=0.6)
        plt.plot(x_bar, i, 'o', color=color, markersize=3)
    
    # 真实均值
    plt.axvline(mu_true, color='blue', linestyle='--', linewidth=2, 
               label=f'True mean = {mu_true}')
    
    plt.xlabel('Value')
    plt.ylabel('Experiment')
    plt.title(f'95% Confidence Intervals\nCoverage: {coverage}/{n_experiments} = {coverage/n_experiments:.2%}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    # plt.show()
    
    print(f"理论覆盖率: {confidence_level:.2%}")
    print(f"实际覆盖率: {coverage/n_experiments:.2%}\n")


# 5. Bootstrap方法
def bootstrap_demo():
    """Bootstrap方法演示"""
    print("=== Bootstrap方法 ===\n")
    
    # 原始样本
    np.random.seed(42)
    n = 50
    original_sample = np.random.exponential(2, n)
    
    # 统计量：中位数
    def statistic(data):
        return np.median(data)
    
    observed_stat = statistic(original_sample)
    
    # Bootstrap
    n_bootstrap = 10000
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        bootstrap_sample = np.random.choice(original_sample, size=n, replace=True)
        bootstrap_stats.append(statistic(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Bootstrap置信区间
    ci_lower = np.percentile(bootstrap_stats, 2.5)
    ci_upper = np.percentile(bootstrap_stats, 97.5)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 原始样本
    plt.subplot(1, 2, 1)
    plt.hist(original_sample, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(observed_stat, color='r', linestyle='--', linewidth=2,
               label=f'Median = {observed_stat:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Original Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bootstrap分布
    plt.subplot(1, 2, 2)
    plt.hist(bootstrap_stats, bins=50, edgecolor='black', alpha=0.7, density=True)
    plt.axvline(observed_stat, color='r', linestyle='--', linewidth=2,
               label=f'Observed = {observed_stat:.2f}')
    plt.axvline(ci_lower, color='g', linestyle=':', linewidth=2,
               label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    plt.axvline(ci_upper, color='g', linestyle=':', linewidth=2)
    plt.xlabel('Median')
    plt.ylabel('Density')
    plt.title('Bootstrap Distribution of Median')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print(f"观测中位数: {observed_stat:.2f}")
    print(f"Bootstrap均值: {np.mean(bootstrap_stats):.2f}")
    print(f"Bootstrap标准误: {np.std(bootstrap_stats):.2f}")
    print(f"95% Bootstrap CI: [{ci_lower:.2f}, {ci_upper:.2f}]\n")


# 6. Delta方法
def delta_method_demo():
    """Delta方法演示"""
    print("=== Delta方法 ===\n")
    
    # 参数
    n = 100
    n_samples = 10000
    mu = 2
    sigma = 1
    
    # 非线性变换: g(x) = x^2
    def g(x):
        return x**2
    
    def g_prime(x):
        return 2*x
    
    # 生成样本均值
    sample_means = np.array([
        np.mean(np.random.normal(mu, sigma, n)) for _ in range(n_samples)
    ])
    
    # 变换后的统计量
    transformed = g(sample_means)
    
    # Delta方法的渐近分布
    asymptotic_mean = g(mu)
    asymptotic_var = (g_prime(mu)**2) * (sigma**2 / n)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 样本均值分布
    plt.subplot(1, 2, 1)
    plt.hist(sample_means, bins=50, density=True, alpha=0.7, 
            edgecolor='black', label='Sample')
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    plt.plot(x, norm.pdf(x, mu, sigma/np.sqrt(n)), 'r-', 
            linewidth=2, label='Theoretical')
    plt.xlabel('Sample mean')
    plt.ylabel('Density')
    plt.title('Distribution of Sample Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 变换后的分布
    plt.subplot(1, 2, 2)
    plt.hist(transformed, bins=50, density=True, alpha=0.7, 
            edgecolor='black', label='Sample')
    x = np.linspace(transformed.min(), transformed.max(), 100)
    plt.plot(x, norm.pdf(x, asymptotic_mean, np.sqrt(asymptotic_var)), 
            'r-', linewidth=2, label='Delta method')
    plt.xlabel('g(Sample mean)')
    plt.ylabel('Density')
    plt.title('Distribution of g(Sample Mean)\ng(x) = x²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    print(f"变换后的均值:")
    print(f"  观测: {np.mean(transformed):.4f}")
    print(f"  理论: {asymptotic_mean:.4f}")
    print(f"\n变换后的方差:")
    print(f"  观测: {np.var(transformed):.4f}")
    print(f"  Delta方法: {asymptotic_var:.4f}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("极限定理示例")
    print("=" * 60 + "\n")
    
    law_of_large_numbers_demo()
    central_limit_theorem_demo()
    berry_esseen_demo()
    confidence_interval_demo()
    bootstrap_demo()
    delta_method_demo()
    
    print("\n所有示例完成！")
```

---

## 📚 练习题

### 练习1：大数定律验证

使用蒙特卡洛方法估计 $\pi$，验证大数定律。

### 练习2：中心极限定理

对于 $\text{Uniform}(0, 1)$ 分布，验证不同样本大小下的CLT。

### 练习3：置信区间

构造总体均值的95%置信区间，验证覆盖率。

### 练习4：Bootstrap

使用Bootstrap方法估计样本中位数的标准误和置信区间。

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **MIT** | 18.650 - Statistics for Applications |
| **Stanford** | STATS200 - Introduction to Statistical Inference |
| **UC Berkeley** | STAT134 - Concepts of Probability |
| **CMU** | 36-705 - Intermediate Statistics |

---

## 📖 参考文献

1. **Billingsley, P. (1995)**. *Probability and Measure*. Wiley.

2. **Durrett, R. (2019)**. *Probability: Theory and Examples*. Cambridge University Press.

3. **Van der Vaart, A. W. (1998)**. *Asymptotic Statistics*. Cambridge University Press.

4. **Casella & Berger (2002)**. *Statistical Inference*. Duxbury Press.

5. **Efron & Tibshirani (1994)**. *An Introduction to the Bootstrap*. Chapman & Hall.

---

*最后更新：2025年10月*
