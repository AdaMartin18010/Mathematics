# 统计推断 (Statistical Inference)

> **From Data to Knowledge: Estimation, Testing, and Decision Making**
>
> 从数据到知识：估计、检验与决策

---

## 目录

- [统计推断 (Statistical Inference)](#统计推断-statistical-inference)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 点估计](#-点估计)
    - [1. 最大似然估计 (MLE)](#1-最大似然估计-mle)
    - [2. 矩估计 (Method of Moments)](#2-矩估计-method-of-moments)
    - [3. 估计量的性质](#3-估计量的性质)
  - [📊 区间估计](#-区间估计)
    - [1. 置信区间](#1-置信区间)
    - [2. 渐近置信区间](#2-渐近置信区间)
    - [3. Bootstrap置信区间](#3-bootstrap置信区间)
  - [🔬 假设检验](#-假设检验)
    - [1. 基本概念](#1-基本概念)
    - [2. 经典检验](#2-经典检验)
    - [3. p值与多重检验](#3-p值与多重检验)
  - [💡 贝叶斯推断](#-贝叶斯推断)
    - [1. 贝叶斯定理](#1-贝叶斯定理)
    - [2. 共轭先验](#2-共轭先验)
    - [3. 后验计算](#3-后验计算)
  - [🎨 变分推断](#-变分推断)
    - [1. 证据下界 (ELBO)](#1-证据下界-elbo)
    - [2. 平均场变分](#2-平均场变分)
    - [3. 变分自编码器 (VAE)](#3-变分自编码器-vae)
  - [🔧 蒙特卡洛方法](#-蒙特卡洛方法)
    - [1. 重要性采样](#1-重要性采样)
    - [2. Markov链蒙特卡洛 (MCMC)](#2-markov链蒙特卡洛-mcmc)
    - [3. Hamiltonian蒙特卡洛 (HMC)](#3-hamiltonian蒙特卡洛-hmc)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：MLE](#练习1mle)
    - [练习2：假设检验](#练习2假设检验)
    - [练习3：贝叶斯推断](#练习3贝叶斯推断)
    - [练习4：变分推断](#练习4变分推断)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**统计推断**是从数据中提取信息、做出决策的科学。

**两大学派**:

```text
频率学派 (Frequentist):
├─ 参数是固定但未知的
├─ 数据是随机的
├─ 方法: MLE, 假设检验, 置信区间
└─ 解释: 长期频率

贝叶斯学派 (Bayesian):
├─ 参数是随机的 (有先验分布)
├─ 数据是观测到的
├─ 方法: 后验分布, 贝叶斯因子
└─ 解释: 主观概率
```

---

## 🎯 点估计

### 1. 最大似然估计 (MLE)

**定义**:

给定数据 $X_1, \ldots, X_n \sim p(x; \theta)$，**似然函数**为：

$$
L(\theta) = \prod_{i=1}^n p(X_i; \theta)
$$

**对数似然**:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(X_i; \theta)
$$

**最大似然估计**:

$$
\hat{\theta}_{MLE} = \arg\max_{\theta} \ell(\theta)
$$

---

**例 1.1 (高斯分布的MLE)**:

设 $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$。

**对数似然**:

$$
\ell(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2
$$

**MLE**:

$$
\hat{\mu}_{MLE} = \frac{1}{n} \sum_{i=1}^n X_i = \bar{X}
$$

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2
$$

---

**定理 1.1 (MLE的渐近性质)**:

在正则条件下，MLE具有以下性质：

1. **相合性** (Consistency):
   $$
   \hat{\theta}_n \xrightarrow{P} \theta_0
   $$

2. **渐近正态性** (Asymptotic Normality):
   $$
   \sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0)^{-1})
   $$
   其中 $I(\theta)$ 是Fisher信息矩阵。

3. **渐近有效性** (Asymptotic Efficiency):
   MLE达到Cramér-Rao下界。

---

**Fisher信息矩阵**:

$$
I(\theta) = -\mathbb{E}\left[\frac{\partial^2 \ell(\theta)}{\partial \theta \partial \theta^T}\right] = \mathbb{E}\left[\left(\frac{\partial \ell(\theta)}{\partial \theta}\right)\left(\frac{\partial \ell(\theta)}{\partial \theta}\right)^T\right]
$$

**Cramér-Rao下界**:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n I(\theta)}
$$

---

### 2. 矩估计 (Method of Moments)

**思想**: 用样本矩估计总体矩。

**$k$阶样本矩**:

$$
\hat{m}_k = \frac{1}{n} \sum_{i=1}^n X_i^k
$$

**$k$阶总体矩**:

$$
m_k(\theta) = \mathbb{E}[X^k]
$$

**矩估计**:

解方程组 $\hat{m}_k = m_k(\theta)$ 得到 $\hat{\theta}_{MM}$。

---

**例 1.2 (指数分布的矩估计)**:

设 $X_1, \ldots, X_n \sim \text{Exp}(\lambda)$，则 $\mathbb{E}[X] = 1/\lambda$。

**矩估计**:

$$
\hat{\lambda}_{MM} = \frac{1}{\bar{X}}
$$

---

### 3. 估计量的性质

**无偏性** (Unbiasedness):

$$
\mathbb{E}[\hat{\theta}] = \theta
$$

**相合性** (Consistency):

$$
\hat{\theta}_n \xrightarrow{P} \theta
$$

**有效性** (Efficiency):

在所有无偏估计中，方差最小。

**均方误差** (MSE):

$$
\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2
$$

---

## 📊 区间估计

### 1. 置信区间

**定义**:

$[L(X), U(X)]$ 是 $\theta$ 的 $1-\alpha$ **置信区间**，如果：

$$
P_{\theta}(\theta \in [L(X), U(X)]) \geq 1 - \alpha, \quad \forall \theta
$$

**解释**: 在重复抽样中，约 $100(1-\alpha)\%$ 的区间包含真实参数。

---

**例 2.1 (正态均值的置信区间)**:

设 $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$，$\sigma^2$ 已知。

$$
\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
$$

$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0, 1)
$$

**$1-\alpha$ 置信区间**:

$$
\left[\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}}, \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right]
$$

其中 $z_{\alpha/2}$ 是标准正态分布的 $1-\alpha/2$ 分位数。

---

**$\sigma^2$ 未知时**:

使用 $t$ 分布：

$$
\frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}
$$

其中 $S^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2$。

**置信区间**:

$$
\left[\bar{X} - t_{\alpha/2, n-1} \frac{S}{\sqrt{n}}, \bar{X} + t_{\alpha/2, n-1} \frac{S}{\sqrt{n}}\right]
$$

---

### 2. 渐近置信区间

利用MLE的渐近正态性：

$$
\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})
$$

**渐近 $1-\alpha$ 置信区间**:

$$
\left[\hat{\theta}_n - z_{\alpha/2} \sqrt{\frac{I(\hat{\theta}_n)^{-1}}{n}}, \hat{\theta}_n + z_{\alpha/2} \sqrt{\frac{I(\hat{\theta}_n)^{-1}}{n}}\right]
$$

---

### 3. Bootstrap置信区间

**Bootstrap方法**:

1. 从样本 $\{X_1, \ldots, X_n\}$ 中有放回抽取 $n$ 个样本，得到 $\{X_1^*, \ldots, X_n^*\}$
2. 计算 $\hat{\theta}^* = \hat{\theta}(X_1^*, \ldots, X_n^*)$
3. 重复 $B$ 次，得到 $\{\hat{\theta}_1^*, \ldots, \hat{\theta}_B^*\}$

**百分位Bootstrap置信区间**:

$$
[\hat{\theta}_{\alpha/2}^*, \hat{\theta}_{1-\alpha/2}^*]
$$

其中 $\hat{\theta}_{\alpha}^*$ 是 $\{\hat{\theta}_1^*, \ldots, \hat{\theta}_B^*\}$ 的 $\alpha$ 分位数。

---

## 🔬 假设检验

### 1. 基本概念

**假设**:

- **原假设** (Null Hypothesis): $H_0$
- **备择假设** (Alternative Hypothesis): $H_1$ 或 $H_a$

**两类错误**:

| | $H_0$ 为真 | $H_0$ 为假 |
|---|---|---|
| **拒绝 $H_0$** | 第一类错误 (Type I) | 正确 |
| **接受 $H_0$** | 正确 | 第二类错误 (Type II) |

**显著性水平**: $\alpha = P(\text{Type I error})$

**功效** (Power): $1 - \beta = 1 - P(\text{Type II error})$

---

**检验统计量**:

基于数据计算的统计量 $T(X)$。

**拒绝域**:

$R = \{x: T(x) > c\}$，其中 $c$ 是临界值。

**p值**:

在 $H_0$ 下，观测到当前或更极端结果的概率。

$$
p = P_{H_0}(T(X) \geq T(x_{obs}))
$$

**决策规则**: 如果 $p < \alpha$，拒绝 $H_0$。

---

### 2. 经典检验

**例 3.1 (单样本 $t$ 检验)**:

设 $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$，$\sigma^2$ 未知。

**假设**:

- $H_0: \mu = \mu_0$
- $H_1: \mu \neq \mu_0$

**检验统计量**:

$$
T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t_{n-1} \quad \text{(under } H_0\text{)}
$$

**拒绝域**: $|T| > t_{\alpha/2, n-1}$

---

**例 3.2 (两样本 $t$ 检验)**:

设 $X_1, \ldots, X_m \sim \mathcal{N}(\mu_1, \sigma^2)$，$Y_1, \ldots, Y_n \sim \mathcal{N}(\mu_2, \sigma^2)$。

**假设**:

- $H_0: \mu_1 = \mu_2$
- $H_1: \mu_1 \neq \mu_2$

**检验统计量** (等方差):

$$
T = \frac{\bar{X} - \bar{Y}}{S_p \sqrt{1/m + 1/n}} \sim t_{m+n-2}
$$

其中 $S_p^2 = \frac{(m-1)S_X^2 + (n-1)S_Y^2}{m+n-2}$ 是合并方差。

---

**例 3.3 (似然比检验)**:

**检验统计量**:

$$
\Lambda = \frac{\sup_{\theta \in \Theta_0} L(\theta)}{\sup_{\theta \in \Theta} L(\theta)}
$$

**Wilks定理**:

在正则条件下，

$$
-2 \log \Lambda \xrightarrow{d} \chi^2_k
$$

其中 $k = \dim(\Theta) - \dim(\Theta_0)$。

---

### 3. p值与多重检验

**多重检验问题**:

同时检验 $m$ 个假设 $H_1, \ldots, H_m$。

**家族错误率** (FWER):

$$
\text{FWER} = P(\text{至少一个Type I错误})
$$

**Bonferroni校正**:

对每个检验使用显著性水平 $\alpha/m$。

$$
\text{FWER} \leq m \cdot \frac{\alpha}{m} = \alpha
$$

---

**错误发现率** (FDR):

$$
\text{FDR} = \mathbb{E}\left[\frac{\text{假阳性数}}{\text{拒绝数}}\right]
$$

**Benjamini-Hochberg过程**:

1. 对 $p$ 值排序: $p_{(1)} \leq \cdots \leq p_{(m)}$
2. 找最大的 $k$ 使得 $p_{(k)} \leq \frac{k}{m} \alpha$
3. 拒绝对应于 $p_{(1)}, \ldots, p_{(k)}$ 的假设

**控制**: $\text{FDR} \leq \alpha$

---

## 💡 贝叶斯推断

### 1. 贝叶斯定理

**贝叶斯公式**:

$$
p(\theta | X) = \frac{p(X | \theta) p(\theta)}{p(X)}
$$

其中：

- $p(\theta)$: **先验分布** (Prior)
- $p(X | \theta)$: **似然** (Likelihood)
- $p(\theta | X)$: **后验分布** (Posterior)
- $p(X) = \int p(X | \theta) p(\theta) d\theta$: **边际似然** (Evidence)

---

**后验正比于先验乘以似然**:

$$
p(\theta | X) \propto p(X | \theta) p(\theta)
$$

---

### 2. 共轭先验

**定义**: 如果先验和后验属于同一分布族，称先验为**共轭先验**。

**例 4.1 (Beta-Binomial共轭)**:

- **似然**: $X \sim \text{Binomial}(n, \theta)$
- **先验**: $\theta \sim \text{Beta}(\alpha, \beta)$

$$
p(\theta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1}
$$

- **后验**: $\theta | X \sim \text{Beta}(\alpha + X, \beta + n - X)$

---

**例 4.2 (Gamma-Poisson共轭)**:

- **似然**: $X_1, \ldots, X_n \sim \text{Poisson}(\lambda)$
- **先验**: $\lambda \sim \text{Gamma}(\alpha, \beta)$

$$
p(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta\lambda}
$$

- **后验**: $\lambda | X \sim \text{Gamma}\left(\alpha + \sum X_i, \beta + n\right)$

---

**例 4.3 (Normal-Normal共轭)**:

- **似然**: $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$，$\sigma^2$ 已知
- **先验**: $\mu \sim \mathcal{N}(\mu_0, \tau^2)$

- **后验**: $\mu | X \sim \mathcal{N}(\mu_n, \tau_n^2)$

其中：

$$
\tau_n^{-2} = \tau^{-2} + n\sigma^{-2}
$$

$$
\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau^2} + \frac{n\bar{X}}{\sigma^2}\right)
$$

---

### 3. 后验计算

**点估计**:

- **后验均值**: $\hat{\theta}_{Bayes} = \mathbb{E}[\theta | X]$
- **后验中位数**: $\text{Median}(\theta | X)$
- **最大后验估计** (MAP): $\hat{\theta}_{MAP} = \arg\max_{\theta} p(\theta | X)$

**区间估计**:

**可信区间** (Credible Interval): $[L, U]$ 使得

$$
P(L \leq \theta \leq U | X) = 1 - \alpha
$$

---

## 🎨 变分推断

### 1. 证据下界 (ELBO)

**问题**: 后验 $p(\theta | X)$ 难以计算。

**解决**: 用简单分布 $q(\theta)$ 近似 $p(\theta | X)$。

**KL散度**:

$$
\text{KL}(q \| p) = \int q(\theta) \log \frac{q(\theta)}{p(\theta | X)} d\theta
$$

**ELBO** (Evidence Lower Bound):

$$
\log p(X) = \text{ELBO}(q) + \text{KL}(q \| p)
$$

其中：

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(X, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

**优化目标**: $\max_q \text{ELBO}(q) \Leftrightarrow \min_q \text{KL}(q \| p)$

---

### 2. 平均场变分

**假设**: $q(\theta) = \prod_{i=1}^m q_i(\theta_i)$

**坐标上升**:

$$
q_j^*(\theta_j) \propto \exp\left(\mathbb{E}_{q_{-j}}[\log p(X, \theta)]\right)
$$

其中 $q_{-j} = \prod_{i \neq j} q_i$。

---

### 3. 变分自编码器 (VAE)

**生成模型**:

$$
p(x) = \int p(x | z) p(z) dz
$$

**变分近似**:

$$
q_\phi(z | x) \approx p(z | x)
$$

**ELBO**:

$$
\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)] - \text{KL}(q_\phi(z|x) \| p(z))
$$

**重参数化技巧**:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

---

## 🔧 蒙特卡洛方法

### 1. 重要性采样

**目标**: 计算 $\mathbb{E}_p[f(X)]$，但从 $p$ 采样困难。

**重要性采样**:

从 $q$ 采样，计算：

$$
\mathbb{E}_p[f(X)] = \mathbb{E}_q\left[f(X) \frac{p(X)}{q(X)}\right]
$$

**估计**:

$$
\hat{\mu} = \frac{1}{n} \sum_{i=1}^n f(X_i) w(X_i), \quad X_i \sim q, \quad w(X) = \frac{p(X)}{q(X)}
$$

---

### 2. Markov链蒙特卡洛 (MCMC)

**Metropolis-Hastings算法**:

1. 初始化 $\theta_0$
2. 对 $t = 0, 1, 2, \ldots$:
   - 从提议分布 $q(\theta' | \theta_t)$ 采样 $\theta'$
   - 计算接受概率:
     $$
     \alpha = \min\left(1, \frac{p(\theta') q(\theta_t | \theta')}{p(\theta_t) q(\theta' | \theta_t)}\right)
     $$
   - 以概率 $\alpha$ 接受 $\theta_{t+1} = \theta'$，否则 $\theta_{t+1} = \theta_t$

---

**Gibbs采样**:

对于 $\theta = (\theta_1, \ldots, \theta_m)$:

1. 初始化 $\theta^{(0)}$
2. 对 $t = 0, 1, 2, \ldots$:
   - $\theta_1^{(t+1)} \sim p(\theta_1 | \theta_2^{(t)}, \ldots, \theta_m^{(t)}, X)$
   - $\theta_2^{(t+1)} \sim p(\theta_2 | \theta_1^{(t+1)}, \theta_3^{(t)}, \ldots, \theta_m^{(t)}, X)$
   - $\vdots$
   - $\theta_m^{(t+1)} \sim p(\theta_m | \theta_1^{(t+1)}, \ldots, \theta_{m-1}^{(t+1)}, X)$

---

### 3. Hamiltonian蒙特卡洛 (HMC)

**思想**: 利用梯度信息，引入动量变量。

**Hamiltonian**:

$$
H(\theta, r) = -\log p(\theta | X) + \frac{1}{2} r^T M^{-1} r
$$

其中 $r$ 是动量，$M$ 是质量矩阵。

**Hamilton方程**:

$$
\frac{d\theta}{dt} = M^{-1} r, \quad \frac{dr}{dt} = \nabla_\theta \log p(\theta | X)
$$

**Leapfrog积分器**:

$$
r_{t+\epsilon/2} = r_t + \frac{\epsilon}{2} \nabla_\theta \log p(\theta_t | X)
$$

$$
\theta_{t+\epsilon} = \theta_t + \epsilon M^{-1} r_{t+\epsilon/2}
$$

$$
r_{t+\epsilon} = r_{t+\epsilon/2} + \frac{\epsilon}{2} \nabla_\theta \log p(\theta_{t+\epsilon} | X)
$$

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# 1. 最大似然估计
def mle_normal(data):
    """正态分布的MLE"""
    mu_mle = np.mean(data)
    sigma2_mle = np.var(data, ddof=0)  # MLE (有偏)
    return mu_mle, sigma2_mle


def mle_general(data, log_likelihood, theta0):
    """一般MLE (数值优化)"""
    def neg_log_lik(theta):
        return -log_likelihood(data, theta)
    
    result = minimize(neg_log_lik, theta0)
    return result.x


# 2. 置信区间
def confidence_interval_normal(data, alpha=0.05):
    """正态均值的置信区间 (方差未知)"""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # 标准误
    
    # t分布临界值
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)
    
    ci = (mean - t_crit * se, mean + t_crit * se)
    return ci


def bootstrap_ci(data, statistic, n_bootstrap=10000, alpha=0.05):
    """Bootstrap置信区间"""
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # 百分位法
    ci = np.percentile(bootstrap_stats, [100*alpha/2, 100*(1-alpha/2)])
    return ci


# 3. 假设检验
def t_test_one_sample(data, mu0, alpha=0.05):
    """单样本t检验"""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    
    # 检验统计量
    t_stat = (mean - mu0) / se
    
    # p值 (双侧)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    
    # 决策
    reject = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_H0': reject,
        'confidence_interval': confidence_interval_normal(data, alpha)
    }


def t_test_two_sample(data1, data2, alpha=0.05):
    """两样本t检验 (等方差)"""
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    
    # 合并方差
    sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    # 检验统计量
    t_stat = (mean1 - mean2) / np.sqrt(sp2 * (1/n1 + 1/n2))
    
    # p值
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n2 - 2))
    
    reject = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_H0': reject
    }


# 4. 贝叶斯推断
class BayesianInference:
    """贝叶斯推断"""
    
    @staticmethod
    def beta_binomial(x, n, alpha_prior=1, beta_prior=1):
        """Beta-Binomial共轭"""
        # 后验参数
        alpha_post = alpha_prior + x
        beta_post = beta_prior + n - x
        
        # 后验均值
        mean_post = alpha_post / (alpha_post + beta_post)
        
        # 可信区间
        ci = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
        
        return {
            'posterior_alpha': alpha_post,
            'posterior_beta': beta_post,
            'posterior_mean': mean_post,
            'credible_interval': ci
        }
    
    @staticmethod
    def normal_normal(data, mu0=0, tau2=1, sigma2=1):
        """Normal-Normal共轭"""
        n = len(data)
        xbar = np.mean(data)
        
        # 后验参数
        tau2_post = 1 / (1/tau2 + n/sigma2)
        mu_post = tau2_post * (mu0/tau2 + n*xbar/sigma2)
        
        # 可信区间
        ci = stats.norm.ppf([0.025, 0.975], mu_post, np.sqrt(tau2_post))
        
        return {
            'posterior_mean': mu_post,
            'posterior_variance': tau2_post,
            'credible_interval': ci
        }


# 5. MCMC
class MetropolisHastings:
    """Metropolis-Hastings算法"""
    
    def __init__(self, log_target, proposal_std=1.0):
        self.log_target = log_target
        self.proposal_std = proposal_std
    
    def sample(self, n_samples, theta0, burn_in=1000):
        """采样"""
        theta = theta0
        samples = []
        n_accept = 0
        
        for i in range(n_samples + burn_in):
            # 提议
            theta_prop = theta + np.random.normal(0, self.proposal_std, size=theta.shape)
            
            # 接受概率
            log_alpha = self.log_target(theta_prop) - self.log_target(theta)
            
            if np.log(np.random.rand()) < log_alpha:
                theta = theta_prop
                n_accept += 1
            
            if i >= burn_in:
                samples.append(theta.copy())
        
        accept_rate = n_accept / (n_samples + burn_in)
        return np.array(samples), accept_rate


class GibbsSampler:
    """Gibbs采样"""
    
    def __init__(self, conditional_samplers):
        """
        conditional_samplers: 列表，每个元素是条件采样函数
        conditional_samplers[i](theta, i) 返回 theta_i | theta_{-i} 的样本
        """
        self.conditional_samplers = conditional_samplers
    
    def sample(self, n_samples, theta0, burn_in=1000):
        """采样"""
        theta = theta0.copy()
        samples = []
        
        for i in range(n_samples + burn_in):
            # 逐个更新
            for j, sampler in enumerate(self.conditional_samplers):
                theta[j] = sampler(theta, j)
            
            if i >= burn_in:
                samples.append(theta.copy())
        
        return np.array(samples)


# 6. 变分推断
class VariationalInference:
    """变分推断"""
    
    @staticmethod
    def elbo(data, q_params, log_likelihood, log_prior, log_q):
        """计算ELBO"""
        # 从q采样
        z_samples = log_q['sample'](q_params, n_samples=1000)
        
        # E_q[log p(x, z)]
        log_joint = log_likelihood(data, z_samples) + log_prior(z_samples)
        
        # E_q[log q(z)]
        log_q_z = log_q['log_prob'](z_samples, q_params)
        
        elbo = np.mean(log_joint - log_q_z)
        return elbo
    
    @staticmethod
    def mean_field_vi(data, log_likelihood, log_prior, q_family, 
                      n_iter=1000, lr=0.01):
        """平均场变分推断"""
        # 初始化变分参数
        q_params = q_family['init']()
        
        for i in range(n_iter):
            # 计算ELBO梯度 (使用自动微分或重参数化)
            grad = q_family['grad_elbo'](data, q_params, log_likelihood, log_prior)
            
            # 梯度上升
            q_params = q_family['update'](q_params, grad, lr)
            
            if i % 100 == 0:
                elbo = VariationalInference.elbo(data, q_params, 
                                                log_likelihood, log_prior, q_family)
                print(f"Iter {i}, ELBO: {elbo:.4f}")
        
        return q_params


# 7. 可视化示例
def demo_inference():
    """统计推断示例"""
    np.random.seed(42)
    
    # 生成数据
    true_mu = 5.0
    true_sigma = 2.0
    n = 100
    data = np.random.normal(true_mu, true_sigma, n)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MLE
    mu_mle, sigma2_mle = mle_normal(data)
    ax = axes[0, 0]
    ax.hist(data, bins=30, density=True, alpha=0.7, label='Data')
    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu_mle, np.sqrt(sigma2_mle)), 
            'r-', linewidth=2, label=f'MLE: μ={mu_mle:.2f}, σ²={sigma2_mle:.2f}')
    ax.axvline(true_mu, color='g', linestyle='--', label=f'True μ={true_mu}')
    ax.set_title('Maximum Likelihood Estimation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 置信区间
    ax = axes[0, 1]
    ci = confidence_interval_normal(data)
    ax.errorbar(1, mu_mle, yerr=[[mu_mle-ci[0]], [ci[1]-mu_mle]], 
                fmt='o', markersize=10, capsize=10, label='95% CI')
    ax.axhline(true_mu, color='g', linestyle='--', label=f'True μ={true_mu}')
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(true_mu-1, true_mu+1)
    ax.set_xticks([1])
    ax.set_xticklabels(['Sample Mean'])
    ax.set_title('Confidence Interval')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Bootstrap
    ax = axes[1, 0]
    bootstrap_means = []
    for _ in range(1000):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    ax.hist(bootstrap_means, bins=50, density=True, alpha=0.7, label='Bootstrap')
    ax.axvline(mu_mle, color='r', linewidth=2, label=f'Sample Mean={mu_mle:.2f}')
    ax.axvline(true_mu, color='g', linestyle='--', label=f'True μ={true_mu}')
    ax.set_title('Bootstrap Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 贝叶斯推断
    ax = axes[1, 1]
    result = BayesianInference.normal_normal(data, mu0=0, tau2=100, 
                                             sigma2=true_sigma**2)
    theta = np.linspace(true_mu-2, true_mu+2, 100)
    
    # 先验
    prior = stats.norm.pdf(theta, 0, 10)
    ax.plot(theta, prior, 'b--', label='Prior', linewidth=2)
    
    # 后验
    posterior = stats.norm.pdf(theta, result['posterior_mean'], 
                               np.sqrt(result['posterior_variance']))
    ax.plot(theta, posterior, 'r-', label='Posterior', linewidth=2)
    
    ax.axvline(true_mu, color='g', linestyle='--', label=f'True μ={true_mu}')
    ax.axvline(result['posterior_mean'], color='r', linestyle=':', 
              label=f"Posterior Mean={result['posterior_mean']:.2f}")
    ax.set_title('Bayesian Inference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("统计推断示例")
    print("=" * 60 + "\n")
    
    demo_inference()
    
    print("\n所有示例完成！")
```

---

## 📚 练习题

### 练习1：MLE

求指数分布 $\text{Exp}(\lambda)$ 的MLE，并验证其渐近正态性。

### 练习2：假设检验

设计并实现似然比检验，比较两个正态分布的方差是否相等。

### 练习3：贝叶斯推断

实现Poisson-Gamma共轭，并可视化先验、似然和后验。

### 练习4：变分推断

实现简单的变分自编码器 (VAE)，并在MNIST数据集上训练。

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **MIT** | 18.650 - Statistics for Applications |
| **Stanford** | STATS200 - Introduction to Statistical Inference |
| **CMU** | 36-705 - Intermediate Statistics |
| **UC Berkeley** | STAT210A - Theoretical Statistics |

---

## 📖 参考文献

1. **Casella & Berger (2002)**. *Statistical Inference*. Duxbury Press.

2. **Gelman et al. (2013)**. *Bayesian Data Analysis*. CRC Press.

3. **Bishop (2006)**. *Pattern Recognition and Machine Learning*. Springer.

4. **Murphy (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press.

5. **Blei et al. (2017)**. *Variational Inference: A Review for Statisticians*. JASA.

---

*最后更新：2025年10月*-
