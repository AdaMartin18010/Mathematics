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

**定理 1.1 的完整证明**:

我们将分别证明MLE的三个渐近性质。为简化记号，考虑标量参数 $\theta \in \mathbb{R}$（多维情况类似）。

**正则条件**:

1. 参数空间 $\Theta$ 包含真实参数 $\theta_0$ 的一个邻域
2. 支撑集 $\{x : p(x; \theta) > 0\}$ 不依赖于 $\theta$
3. $\log p(x; \theta)$ 关于 $\theta$ 三次可微
4. 可以交换求导和积分的顺序
5. Fisher信息 $I(\theta_0) > 0$ 且有限

---

**证明 (1): 相合性**:

**证明思路**: 证明 $\hat{\theta}_n$ 最大化样本对数似然，而样本对数似然依概率收敛到期望对数似然，后者在 $\theta_0$ 处唯一最大。

**第一步：定义目标函数**:

令：

- $\ell_n(\theta) = \frac{1}{n} \sum_{i=1}^n \log p(X_i; \theta)$ （样本平均对数似然）
- $\ell(\theta) = \mathbb{E}[\log p(X; \theta)]$ （期望对数似然）

**第二步：大数定律**:

由强大数定律，对每个固定的 $\theta$：

$$
\ell_n(\theta) \xrightarrow{a.s.} \ell(\theta)
$$

**第三步：识别最大值点**:

**引理**: $\ell(\theta)$ 在 $\theta = \theta_0$ 处唯一最大。

**证明引理**: 使用Kullback-Leibler散度。对于 $\theta \neq \theta_0$：

$$
\ell(\theta) - \ell(\theta_0) = \mathbb{E}[\log p(X; \theta)] - \mathbb{E}[\log p(X; \theta_0)]
$$

$$
= \mathbb{E}\left[\log \frac{p(X; \theta)}{p(X; \theta_0)}\right] = -\text{KL}(p_{\theta_0} \| p_\theta) < 0
$$

（KL散度非负，且仅当 $\theta = \theta_0$ 时为零）

**第四步：一致收敛**:

在紧集上，$\ell_n(\theta)$ 一致收敛到 $\ell(\theta)$（需要更强的条件，如可控性）。

**第五步：结论**:

由于 $\hat{\theta}_n$ 最大化 $\ell_n(\theta)$，而 $\ell_n(\theta)$ 一致收敛到在 $\theta_0$ 处唯一最大的 $\ell(\theta)$，因此：

$$
\hat{\theta}_n \xrightarrow{P} \theta_0
$$

$\square$

---

**证明 (2): 渐近正态性**:

这是最重要也最技术性的部分。

**第一步：Score函数**:

定义**Score函数**：

$$
s(\theta) = \frac{\partial \log p(X; \theta)}{\partial \theta}
$$

**关键性质**:

- $\mathbb{E}[s(\theta_0)] = 0$ （在真实参数下，Score的期望为零）
- $\text{Var}(s(\theta_0)) = I(\theta_0)$ （Fisher信息）

**证明第一个性质**:

$$
\mathbb{E}[s(\theta_0)] = \mathbb{E}\left[\frac{\partial \log p(X; \theta_0)}{\partial \theta}\right] = \int \frac{\partial p(x; \theta_0)}{\partial \theta} dx
$$

$$
= \frac{\partial}{\partial \theta} \int p(x; \theta_0) dx = \frac{\partial}{\partial \theta} 1 = 0
$$

**第二步：MLE的一阶条件**:

MLE $\hat{\theta}_n$ 满足：

$$
\frac{\partial \ell_n(\theta)}{\partial \theta}\bigg|_{\theta = \hat{\theta}_n} = \frac{1}{n} \sum_{i=1}^n s_i(\hat{\theta}_n) = 0
$$

其中 $s_i(\theta) = \frac{\partial \log p(X_i; \theta)}{\partial \theta}$。

**第三步：Taylor展开**:

在 $\theta_0$ 附近对Score函数求和进行Taylor展开：

$$
0 = \frac{1}{n} \sum_{i=1}^n s_i(\hat{\theta}_n) = \frac{1}{n} \sum_{i=1}^n s_i(\theta_0) + \frac{1}{n} \sum_{i=1}^n s_i'(\tilde{\theta}_n)(\hat{\theta}_n - \theta_0)
$$

其中 $\tilde{\theta}_n$ 在 $\theta_0$ 和 $\hat{\theta}_n$ 之间。

重新整理：

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) = -\left[\frac{1}{n} \sum_{i=1}^n s_i'(\tilde{\theta}_n)\right]^{-1} \cdot \frac{1}{\sqrt{n}} \sum_{i=1}^n s_i(\theta_0)
$$

**第四步：分析分子**:

由中心极限定理，$\{s_i(\theta_0)\}$ 是独立同分布的，$\mathbb{E}[s_i(\theta_0)] = 0$，$\text{Var}(s_i(\theta_0)) = I(\theta_0)$，因此：

$$
\frac{1}{\sqrt{n}} \sum_{i=1}^n s_i(\theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0))
$$

**第五步：分析分母**:

$$
s_i'(\theta) = \frac{\partial^2 \log p(X_i; \theta)}{\partial \theta^2}
$$

由大数定律：

$$
\frac{1}{n} \sum_{i=1}^n s_i'(\tilde{\theta}_n) \xrightarrow{P} \mathbb{E}[s'(\theta_0)] = -I(\theta_0)
$$

（这里使用了 $\tilde{\theta}_n \xrightarrow{P} \theta_0$ 和连续性）

**第六步：Slutsky定理**:

结合第四步和第五步，应用Slutsky定理：

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) = \frac{1}{\sqrt{n}} \sum_{i=1}^n s_i(\theta_0) \cdot \left[-\frac{1}{n} \sum_{i=1}^n s_i'(\tilde{\theta}_n)\right]^{-1}
$$

$$
\xrightarrow{d} \mathcal{N}(0, I(\theta_0)) \cdot \frac{1}{I(\theta_0)} = \mathcal{N}(0, I(\theta_0)^{-1})
$$

因此：

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0)^{-1})
$$

$\square$

---

**证明 (3): 渐近有效性**:

**Cramér-Rao下界**:

对于任何无偏估计量 $\hat{\theta}$：

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n I(\theta)}
$$

**证明Cramér-Rao下界**:

设 $\hat{\theta}$ 是 $\theta$ 的无偏估计量，即 $\mathbb{E}[\hat{\theta}] = \theta$。

对 $\theta$ 求导：

$$
\frac{\partial}{\partial \theta} \mathbb{E}[\hat{\theta}] = 1
$$

$$
\int \hat{\theta}(x) \frac{\partial p(x; \theta)}{\partial \theta} dx = 1
$$

$$
\int \hat{\theta}(x) p(x; \theta) \frac{\partial \log p(x; \theta)}{\partial \theta} dx = 1
$$

$$
\mathbb{E}[\hat{\theta} \cdot s(\theta)] = 1
$$

由Cauchy-Schwarz不等式：

$$
1 = \mathbb{E}[\hat{\theta} \cdot s(\theta)]^2 \leq \mathbb{E}[\hat{\theta}^2] \cdot \mathbb{E}[s(\theta)^2] = \text{Var}(\hat{\theta}) \cdot I(\theta)
$$

（使用了 $\mathbb{E}[\hat{\theta}] = \theta$, $\mathbb{E}[s(\theta)] = 0$）

因此：

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

对于 $n$ 个样本：

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n I(\theta)}
$$

$\square$

**MLE达到下界**:

从渐近正态性：

$$
\text{Var}(\hat{\theta}_n) \sim \frac{I(\theta_0)^{-1}}{n} = \frac{1}{n I(\theta_0)}
$$

因此MLE渐近地达到Cramér-Rao下界，是渐近有效的。 $\square$

---

**证明的关键要点**：

1. **相合性**: 基于大数定律和KL散度的唯一性
2. **渐近正态性**: 基于CLT和Taylor展开，这是最核心的结果
3. **渐近有效性**: MLE的方差达到理论下界

**几何直观**：

- **Score函数** $s(\theta)$ 是对数似然的梯度，指向似然增加的方向
- **Fisher信息** $I(\theta)$ 度量了Score函数的变异性，反映了数据提供的信息量
- **MLE** 通过设置Score为零找到似然的最大值点

**实际意义**：

MLE的渐近性质保证了：

1. 大样本下，MLE收敛到真实参数（相合性）
2. MLE的分布趋于正态（可以构造置信区间）
3. MLE是最优的（渐近有效性）

这些性质使得MLE成为统计推断中最重要的方法之一。

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
| ---- |---| ---- |
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

**Wilks定理的证明**:

**定理陈述（完整版）**:

设 $X_1, \ldots, X_n$ 是来自密度函数 $f(x; \theta)$ 的独立同分布样本，其中 $\theta \in \Theta \subseteq \mathbb{R}^p$。

考虑假设检验：

- $H_0: \theta \in \Theta_0$（$\Theta_0$ 是 $q$ 维子空间，$q < p$）
- $H_1: \theta \in \Theta$

似然比统计量：

$$
\Lambda_n = \frac{\sup_{\theta \in \Theta_0} L_n(\theta)}{\sup_{\theta \in \Theta} L_n(\theta)}
$$

**定理**: 在正则条件下，当 $H_0$ 为真时：

$$
-2 \log \Lambda_n \xrightarrow{d} \chi^2_{p-q}
$$

其中 $p - q$ 是自由度（参数空间的维数差）。

---

**正则条件**:

1. **参数空间**: $\Theta$ 是 $\mathbb{R}^p$ 的开集，$\Theta_0$ 是 $q$ 维光滑子流形
2. **可识别性**: 不同的 $\theta$ 对应不同的分布
3. **Fisher信息**: Fisher信息矩阵 $I(\theta)$ 在 $\Theta$ 上连续且正定
4. **正则性**: 对数似然函数满足通常的正则性条件（可交换求导与积分顺序等）
5. **真参数**: 真参数 $\theta_0 \in \Theta_0$（$H_0$ 为真）

---

**证明**:

**第一步：MLE的渐近正态性**:

在正则条件下，无约束MLE $\hat{\theta}_n$ 满足：

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})
$$

其中 $I(\theta_0)$ 是Fisher信息矩阵。

**第二步：约束MLE**:

令 $\tilde{\theta}_n$ 是约束MLE（在 $\Theta_0$ 上的MLE）。

由于 $\Theta_0$ 是 $q$ 维子流形，可以用局部参数化：

$$
\theta = h(\psi), \quad \psi \in \mathbb{R}^q
$$

其中 $h: \mathbb{R}^q \to \Theta_0$ 是光滑嵌入。

约束MLE $\tilde{\theta}_n$ 也满足渐近正态性（在 $\Theta_0$ 内）。

**第三步：对数似然比的Taylor展开**:

对数似然比：

$$
\log \Lambda_n = \ell_n(\tilde{\theta}_n) - \ell_n(\hat{\theta}_n)
$$

其中 $\ell_n(\theta) = \log L_n(\theta) = \sum_{i=1}^n \log f(X_i; \theta)$。

在 $\theta_0$ 附近对 $\ell_n$ 进行二阶Taylor展开：

$$
\ell_n(\theta) = \ell_n(\theta_0) + \nabla \ell_n(\theta_0)^T (\theta - \theta_0) + \frac{1}{2} (\theta - \theta_0)^T H_n(\theta^*) (\theta - \theta_0)
$$

其中 $H_n(\theta) = \nabla^2 \ell_n(\theta)$ 是Hessian矩阵，$\theta^*$ 在 $\theta$ 和 $\theta_0$ 之间。

**第四步：无约束MLE的展开**:

由于 $\hat{\theta}_n$ 是无约束MLE，有 $\nabla \ell_n(\hat{\theta}_n) = 0$。

在 $\hat{\theta}_n$ 附近展开：

$$
\ell_n(\theta_0) = \ell_n(\hat{\theta}_n) + \frac{1}{2} (\theta_0 - \hat{\theta}_n)^T H_n(\theta_1^*) (\theta_0 - \hat{\theta}_n)
$$

**第五步：约束MLE的展开**:

类似地，在 $\tilde{\theta}_n$ 附近展开：

$$
\ell_n(\theta_0) = \ell_n(\tilde{\theta}_n) + \nabla \ell_n(\tilde{\theta}_n)^T (\theta_0 - \tilde{\theta}_n) + \frac{1}{2} (\theta_0 - \tilde{\theta}_n)^T H_n(\theta_2^*) (\theta_0 - \tilde{\theta}_n)
$$

但由于 $\tilde{\theta}_n$ 是约束MLE，梯度 $\nabla \ell_n(\tilde{\theta}_n)$ 在约束方向上为零，在垂直方向上非零。

**第六步：关键观察**:

由于 $\hat{\theta}_n$ 是全局最优，而 $\tilde{\theta}_n$ 是约束最优：

$$
\ell_n(\hat{\theta}_n) \geq \ell_n(\tilde{\theta}_n)
$$

因此：

$$
-2 \log \Lambda_n = 2[\ell_n(\hat{\theta}_n) - \ell_n(\tilde{\theta}_n)] \geq 0
$$

**第七步：渐近展开**:

使用 $H_n(\theta) \xrightarrow{P} -n I(\theta_0)$（由大数定律）：

$$
\ell_n(\hat{\theta}_n) - \ell_n(\theta_0) \approx -\frac{n}{2} (\hat{\theta}_n - \theta_0)^T I(\theta_0) (\hat{\theta}_n - \theta_0)
$$

$$
\ell_n(\tilde{\theta}_n) - \ell_n(\theta_0) \approx -\frac{n}{2} (\tilde{\theta}_n - \theta_0)^T I(\theta_0) (\tilde{\theta}_n - \theta_0)
$$

**第八步：投影解释**:

令 $Z_n = \sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$。

约束 $\theta \in \Theta_0$ 相当于线性约束（在一阶近似下）：

$$
A^T (\theta - \theta_0) = 0
$$

其中 $A$ 是 $p \times (p-q)$ 矩阵，列向量张成 $\Theta_0$ 的正交补空间。

$\tilde{\theta}_n$ 是 $\hat{\theta}_n$ 在 $\Theta_0$ 上的投影（在Fisher度量下）。

**第九步：二次型**:

$$
-2 \log \Lambda_n \approx n [(\hat{\theta}_n - \theta_0)^T I(\theta_0) (\hat{\theta}_n - \theta_0) - (\tilde{\theta}_n - \theta_0)^T I(\theta_0) (\tilde{\theta}_n - \theta_0)]
$$

令 $W = I(\theta_0)^{1/2}$，$Y_n = W \sqrt{n}(\hat{\theta}_n - \theta_0)$，则 $Y_n \xrightarrow{d} N(0, I_p)$。

令 $P$ 是到约束子空间的正交投影矩阵（在标准内积下），则：

$$
-2 \log \Lambda_n \approx Y_n^T Y_n - (P Y_n)^T (P Y_n) = Y_n^T (I - P) Y_n
$$

**第十步：投影矩阵的性质**:

$I - P$ 是到 $\Theta_0$ 的正交补空间的投影矩阵，秩为 $p - q$。

因此 $I - P$ 有 $p - q$ 个特征值为 1，其余为 0。

**第十一步：卡方分布**:

对于标准正态向量 $Y \sim N(0, I_p)$ 和秩为 $r$ 的投影矩阵 $P$：

$$
Y^T P Y \sim \chi^2_r
$$

因此：

$$
-2 \log \Lambda_n \xrightarrow{d} \chi^2_{p-q}
$$

$\square$

---

**关键要点**:

1. **自由度**: $p - q$ 是参数空间维数差，表示约束的"紧度"
2. **投影解释**: 似然比检验等价于检验MLE到约束空间的"距离"
3. **Fisher度量**: 使用Fisher信息矩阵定义的度量
4. **正则条件**: 保证MLE的渐近正态性和Taylor展开的有效性

---

**Wilks定理的应用**:

**例1：单参数检验**:

检验 $H_0: \theta = \theta_0$ vs $H_1: \theta \neq \theta_0$（$p = 1, q = 0$）：

$$
-2 \log \Lambda_n = 2[\ell_n(\hat{\theta}_n) - \ell_n(\theta_0)] \xrightarrow{d} \chi^2_1
$$

**例2：嵌套模型比较**:

模型1（完整）: $p$ 个参数
模型2（简化）: $q$ 个参数（$q < p$）

$$
-2 \log \Lambda_n = 2[\ell_n(\text{完整}) - \ell_n(\text{简化})] \xrightarrow{d} \chi^2_{p-q}
$$

**例3：正态分布方差检验**:

$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$，检验 $H_0: \sigma^2 = \sigma_0^2$ vs $H_1: \sigma^2 \neq \sigma_0^2$：

对数似然：

$$
\ell_n(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log \sigma^2 - \frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2
$$

无约束MLE: $\hat{\mu} = \bar{X}$, $\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2$

约束MLE（$\sigma^2 = \sigma_0^2$）: $\tilde{\mu} = \bar{X}$, $\tilde{\sigma}^2 = \sigma_0^2$

似然比统计量：

$$
-2 \log \Lambda_n = n \log \frac{\sigma_0^2}{\hat{\sigma}^2} + n \frac{\hat{\sigma}^2}{\sigma_0^2} - n \xrightarrow{d} \chi^2_1
$$

**例4：线性回归系数检验**:

完整模型: $Y = X\beta + \epsilon$（$p$ 个系数）
简化模型: $Y = X_0\beta_0 + \epsilon$（$q$ 个系数）

$$
-2 \log \Lambda_n = n \log \frac{\text{RSS}_0}{\text{RSS}} \xrightarrow{d} \chi^2_{p-q}
$$

其中 RSS 是残差平方和。

---

**Wilks定理的局限性**:

1. **正则条件**: 需要满足严格的正则性条件
2. **边界问题**: 当真参数在参数空间边界上时，结论可能不成立
3. **小样本**: 渐近结果，小样本时可能不准确
4. **模型错误指定**: 假设模型正确，否则结论无效

---

**与其他检验的关系**:

**1. Wald检验**:

$$
W_n = n(\hat{\theta}_n - \theta_0)^T I(\hat{\theta}_n) (\hat{\theta}_n - \theta_0) \xrightarrow{d} \chi^2_{p-q}
$$

**2. Score检验（Rao检验）**:

$$
S_n = \frac{1}{n} \nabla \ell_n(\theta_0)^T I(\theta_0)^{-1} \nabla \ell_n(\theta_0) \xrightarrow{d} \chi^2_{p-q}
$$

**3. 三者的关系**:

在 $H_0$ 下，三个统计量渐近等价：

$$
W_n \sim S_n \sim -2 \log \Lambda_n \xrightarrow{d} \chi^2_{p-q}
$$

但在有限样本中可能有差异：

- Wald: 只需要无约束MLE
- Score: 只需要约束MLE
- LRT: 需要两个MLE，但通常功效最好

---

**数值例子**:

设 $X_1, \ldots, X_{100} \sim N(\mu, 1)$，检验 $H_0: \mu = 0$ vs $H_1: \mu \neq 0$。

观测到 $\bar{X} = 0.3$。

对数似然比：

$$
-2 \log \Lambda = 100 \times (\bar{X} - 0)^2 = 100 \times 0.09 = 9
$$

临界值（$\alpha = 0.05$）: $\chi^2_1(0.95) = 3.841$

由于 $9 > 3.841$，拒绝 $H_0$。

p值: $P(\chi^2_1 > 9) \approx 0.0027$

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

**ELBO的完整推导**:

**目标**: 推导证据下界（ELBO）并说明为什么最大化ELBO等价于最小化KL散度。

**第一步：从KL散度开始**:

我们想要最小化变分分布 $q(\theta)$ 与后验 $p(\theta | X)$ 之间的KL散度：

$$
\text{KL}(q \| p) = \int q(\theta) \log \frac{q(\theta)}{p(\theta | X)} d\theta
$$

展开：

$$
\text{KL}(q \| p) = \int q(\theta) \log q(\theta) d\theta - \int q(\theta) \log p(\theta | X) d\theta
$$

$$
= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\theta | X)]
$$

**第二步：使用Bayes定理**:

由Bayes定理：

$$
p(\theta | X) = \frac{p(X, \theta)}{p(X)} = \frac{p(X | \theta) p(\theta)}{p(X)}
$$

因此：

$$
\log p(\theta | X) = \log p(X, \theta) - \log p(X)
$$

**第三步：代入KL散度**:

$$
\text{KL}(q \| p) = \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(X, \theta)] + \mathbb{E}_q[\log p(X)]
$$

由于 $\log p(X)$ 不依赖于 $\theta$，$\mathbb{E}_q[\log p(X)] = \log p(X)$：

$$
\text{KL}(q \| p) = \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(X, \theta)] + \log p(X)
$$

**第四步：重新整理**:

$$
\log p(X) = \mathbb{E}_q[\log p(X, \theta)] - \mathbb{E}_q[\log q(\theta)] + \text{KL}(q \| p)
$$

定义 **ELBO**（证据下界）：

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(X, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

因此：

$$
\log p(X) = \text{ELBO}(q) + \text{KL}(q \| p)
$$

$\square$

**第五步：ELBO是下界**:

由于 $\text{KL}(q \| p) \geq 0$（KL散度的非负性），我们有：

$$
\log p(X) \geq \text{ELBO}(q)
$$

这就是为什么称为"证据下界"（Evidence Lower Bound）。

**第六步：优化等价性**:

从 $\log p(X) = \text{ELBO}(q) + \text{KL}(q \| p)$，我们看到：

$$
\max_q \text{ELBO}(q) \Leftrightarrow \min_q \text{KL}(q \| p)
$$

因为 $\log p(X)$ 不依赖于 $q$（它是固定的"证据"）。

---

**ELBO的两种等价形式**:

**形式1**（期望形式）:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(X, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

**形式2**（重建 + 正则化）:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(X | \theta)] - \text{KL}(q(\theta) \| p(\theta))
$$

**推导形式2**:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(X, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

$$
= \mathbb{E}_q[\log p(X | \theta) + \log p(\theta)] - \mathbb{E}_q[\log q(\theta)]
$$

$$
= \mathbb{E}_q[\log p(X | \theta)] + \mathbb{E}_q[\log p(\theta)] - \mathbb{E}_q[\log q(\theta)]
$$

$$
= \mathbb{E}_q[\log p(X | \theta)] - \text{KL}(q(\theta) \| p(\theta))
$$

**解释**:

- 第一项：**重建项**（reconstruction term），衡量生成数据的能力
- 第二项：**正则化项**（regularization term），使 $q$ 接近先验 $p$

---

**在VAE中的应用**:

在变分自编码器（VAE）中：

- $\theta$ 变为潜在变量 $z$
- $q(z | x)$ 是编码器（encoder）
- $p(x | z)$ 是解码器（decoder）

ELBO变为：

$$
\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)] - \text{KL}(q_\phi(z|x) \| p(z))
$$

**优化**:

- 最大化重建项：解码器学习重建输入
- 最小化KL项：编码器学习接近先验（通常是标准正态分布）

---

**ELBO梯度的计算**:

**问题**: 如何计算 $\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)]$？

**方法1：REINFORCE（Score Function Estimator）**:

$$
\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{q_\phi(z)}[f(z) \nabla_\phi \log q_\phi(z)]
$$

**方法2：重参数化技巧（Reparameterization Trick）**:

若 $z = g(\phi, \epsilon)$，其中 $\epsilon \sim p(\epsilon)$：

$$
\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g(\phi, \epsilon))]
$$

**例子**（正态分布）:

若 $q_\phi(z) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$，重参数化为：

$$
z = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

则：

$$
\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[\nabla_\phi f(\mu_\phi + \sigma_\phi \cdot \epsilon)]
$$

这个梯度可以用蒙特卡洛估计，且方差较低。

---

**ELBO与EM算法的关系**:

EM算法可以看作是变分推断的特殊情况：

- **E步**: 固定参数，计算后验 $q(\theta) = p(\theta | X, \hat{\theta}^{(t)})$
- **M步**: 固定后验，最大化ELBO关于参数

在EM中，E步得到精确后验，而变分推断中 $q$ 是近似的。

---

**数值示例**:

考虑简单的高斯混合模型：

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)
$$

**真实后验**（难以计算）:

$$
p(z | x) = \frac{\pi_z \mathcal{N}(x | \mu_z, \sigma_z^2)}{\sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)}
$$

**变分近似**:

$$
q(z) = \text{Categorical}(\phi_1, \ldots, \phi_K)
$$

最大化ELBO得到 $\phi$ 的最优值，近似真实后验。

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
| ---- |------|
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
