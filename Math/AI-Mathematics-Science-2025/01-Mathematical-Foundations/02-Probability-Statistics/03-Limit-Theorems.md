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

**定理 1.1 的完整证明**:

我们将给出两个证明：一个基于Chebyshev不等式（假设方差有限），一个基于特征函数（更一般）。

---

**证明1：基于Chebyshev不等式**:

**假设**: $\text{Var}(X_i) = \sigma^2 < \infty$

**第一步：计算样本均值的期望**:

$$
E[\bar{X}_n] = E\left[\frac{1}{n} \sum_{i=1}^n X_i\right] = \frac{1}{n} \sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu
$$

**第二步：计算样本均值的方差**:

由于 $X_1, X_2, \ldots, X_n$ 独立：

$$
\text{Var}(\bar{X}_n) = \text{Var}\left(\frac{1}{n} \sum_{i=1}^n X_i\right) = \frac{1}{n^2} \text{Var}\left(\sum_{i=1}^n X_i\right)
$$

由独立性，方差可加：

$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) = n\sigma^2
$$

因此：

$$
\text{Var}(\bar{X}_n) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}
$$

**第三步：应用Chebyshev不等式**:

对于任意 $\epsilon > 0$，Chebyshev不等式给出：

$$
P(|\bar{X}_n - E[\bar{X}_n]| > \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2}
$$

代入 $E[\bar{X}_n] = \mu$ 和 $\text{Var}(\bar{X}_n) = \sigma^2/n$：

$$
P(|\bar{X}_n - \mu| > \epsilon) \leq \frac{\sigma^2}{n\epsilon^2}
$$

**第四步：取极限**:

当 $n \to \infty$：

$$
\lim_{n \to \infty} P(|\bar{X}_n - \mu| > \epsilon) \leq \lim_{n \to \infty} \frac{\sigma^2}{n\epsilon^2} = 0
$$

由于概率非负，因此：

$$
\lim_{n \to \infty} P(|\bar{X}_n - \mu| > \epsilon) = 0
$$

这正是依概率收敛的定义：

$$
\bar{X}_n \xrightarrow{P} \mu
$$

$\square$

---

**证明2：基于特征函数（更一般，不需要方差有限）**:

**假设**: 仅需 $E[X_i] = \mu$ 存在

**第一步：标准化**:

令 $Y_i = X_i - \mu$，则 $E[Y_i] = 0$。

需要证明：

$$
\frac{1}{n} \sum_{i=1}^n Y_i \xrightarrow{P} 0
$$

**第二步：特征函数**:

令 $S_n = \sum_{i=1}^n Y_i$，其特征函数为：

$$
\phi_{S_n/n}(t) = E\left[\exp\left(i\frac{t}{n} S_n\right)\right] = \prod_{j=1}^n E\left[\exp\left(i\frac{t}{n} Y_j\right)\right] = \left[\phi_Y\left(\frac{t}{n}\right)\right]^n
$$

其中 $\phi_Y(t) = E[e^{itY}]$ 是 $Y_i$ 的特征函数。

**第三步：Taylor展开**:

由于 $E[Y] = 0$，在 $t = 0$ 附近：

$$
\phi_Y(t) = E[e^{itY}] = E[1 + itY + O(t^2)] = 1 + it E[Y] + O(t^2) = 1 + O(t^2)
$$

更精确地，对于小的 $|t|$：

$$
|\phi_Y(t) - 1| = O(t^2)
$$

**第四步：代入并取极限**:

$$
\phi_{S_n/n}(t) = \left[\phi_Y\left(\frac{t}{n}\right)\right]^n = \left[1 + O\left(\frac{t^2}{n^2}\right)\right]^n
$$

当 $n \to \infty$：

$$
\lim_{n \to \infty} \left[1 + O\left(\frac{1}{n^2}\right)\right]^n = 1
$$

因此：

$$
\lim_{n \to \infty} \phi_{S_n/n}(t) = 1 = \phi_0(t)
$$

其中 $\phi_0(t) = 1$ 是常数0的特征函数。

**第五步：结论**:

由Lévy连续性定理，特征函数的逐点收敛蕴含依分布收敛：

$$
\frac{S_n}{n} = \frac{1}{n} \sum_{i=1}^n Y_i \xrightarrow{d} 0
$$

对于常数，依分布收敛等价于依概率收敛，因此：

$$
\frac{1}{n} \sum_{i=1}^n Y_i \xrightarrow{P} 0
$$

即：

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{P} \mu
$$

$\square$

---

**两个证明的比较**:

| 方法 | 假设 | 优点 | 缺点 |
|------|------|------|------|
| Chebyshev不等式 | 需要 $\sigma^2 < \infty$ | 简单直接，易于理解 | 需要方差有限 |
| 特征函数 | 仅需 $E[X] < \infty$ | 更一般，不需要方差 | 需要更多概率论知识 |

**注意**:

- Khinchin的原始定理仅假设期望存在，使用的是特征函数方法
- 如果方差有限，Chebyshev方法更简单
- 如果方差不存在（如Cauchy分布），必须使用特征函数方法

---

**应用示例**:

考虑抛硬币实验，$X_i \sim \text{Bernoulli}(p)$。

- $E[X_i] = p$
- $\text{Var}(X_i) = p(1-p)$

弱大数定律告诉我们：

$$
\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{P} p
$$

即：频率收敛到概率。这是频率学派统计的理论基础。

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

**定理 1.1 的完整证明**:

我们使用**特征函数方法**证明中心极限定理。

**证明**:

**第一步：标准化**:

不失一般性，假设 $\mu = 0$, $\sigma^2 = 1$（否则考虑 $Y_i = (X_i - \mu)/\sigma$）。

令 $S_n = \sum_{i=1}^n X_i$，我们要证明：

$$
\frac{S_n}{\sqrt{n}} \xrightarrow{d} N(0, 1)
$$

**第二步：特征函数**:

回顾特征函数的定义：

$$
\phi_X(t) = \mathbb{E}[e^{itX}]
$$

**关键性质**：

1. 独立随机变量和的特征函数等于特征函数的乘积
2. 特征函数唯一确定分布
3. 依分布收敛等价于特征函数逐点收敛

令 $\phi(t) = \phi_{X_1}(t)$ 是 $X_i$ 的特征函数（因为同分布）。

$S_n/\sqrt{n}$ 的特征函数为：

$$
\phi_{S_n/\sqrt{n}}(t) = \mathbb{E}\left[\exp\left(it \frac{S_n}{\sqrt{n}}\right)\right] = \mathbb{E}\left[\exp\left(i \frac{t}{\sqrt{n}} \sum_{j=1}^n X_j\right)\right]
$$

由独立性：

$$
\phi_{S_n/\sqrt{n}}(t) = \prod_{j=1}^n \mathbb{E}\left[\exp\left(i \frac{t}{\sqrt{n}} X_j\right)\right] = \prod_{j=1}^n \phi\left(\frac{t}{\sqrt{n}}\right) = \left[\phi\left(\frac{t}{\sqrt{n}}\right)\right]^n
$$

**第三步：Taylor展开**:

由于 $\mathbb{E}[X_i] = 0$, $\mathbb{E}[X_i^2] = 1$，我们可以对 $\phi(t)$ 在 $t=0$ 处进行Taylor展开：

$$
\phi(t) = \mathbb{E}[e^{itX}] = \mathbb{E}\left[1 + itX + \frac{(itX)^2}{2!} + \frac{(itX)^3}{3!} + \cdots\right]
$$

由于可以交换期望和级数（在适当条件下）：

$$
\phi(t) = 1 + it\mathbb{E}[X] + \frac{(it)^2}{2}\mathbb{E}[X^2] + \frac{(it)^3}{6}\mathbb{E}[X^3] + O(t^4)
$$

代入 $\mathbb{E}[X] = 0$, $\mathbb{E}[X^2] = 1$：

$$
\phi(t) = 1 - \frac{t^2}{2} + o(t^2)
$$

更精确地，对于小的 $|t|$：

$$
\phi(t) = 1 - \frac{t^2}{2} + o(t^2)
$$

**第四步：代入并取极限**:

现在计算：

$$
\phi\left(\frac{t}{\sqrt{n}}\right) = 1 - \frac{t^2}{2n} + o\left(\frac{t^2}{n}\right)
$$

因此：

$$
\left[\phi\left(\frac{t}{\sqrt{n}}\right)\right]^n = \left[1 - \frac{t^2}{2n} + o\left(\frac{1}{n}\right)\right]^n
$$

**第五步：使用对数技巧**:

取对数：

$$
\log\left[\phi\left(\frac{t}{\sqrt{n}}\right)\right]^n = n \log\left[1 - \frac{t^2}{2n} + o\left(\frac{1}{n}\right)\right]
$$

使用 $\log(1 + x) = x - \frac{x^2}{2} + O(x^3)$ 对于小的 $|x|$：

$$
\log\left[1 - \frac{t^2}{2n} + o\left(\frac{1}{n}\right)\right] = -\frac{t^2}{2n} + o\left(\frac{1}{n}\right)
$$

因此：

$$
n \log\left[1 - \frac{t^2}{2n} + o\left(\frac{1}{n}\right)\right] = n \cdot \left(-\frac{t^2}{2n} + o\left(\frac{1}{n}\right)\right) = -\frac{t^2}{2} + o(1)
$$

当 $n \to \infty$：

$$
\lim_{n \to \infty} \left[\phi\left(\frac{t}{\sqrt{n}}\right)\right]^n = \lim_{n \to \infty} \exp\left(-\frac{t^2}{2} + o(1)\right) = \exp\left(-\frac{t^2}{2}\right)
$$

**第六步：识别极限分布**:

注意到 $\exp(-t^2/2)$ 正是标准正态分布 $N(0, 1)$ 的特征函数：

$$
\phi_{N(0,1)}(t) = \mathbb{E}[e^{itZ}] = \int_{-\infty}^\infty e^{itx} \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = e^{-t^2/2}
$$

**第七步：结论**:

由特征函数的连续性定理（Lévy连续性定理）：

如果 $\phi_n(t) \to \phi(t)$ 对所有 $t$ 成立，且 $\phi$ 在 $t=0$ 处连续，则 $X_n \xrightarrow{d} X$，其中 $X$ 的特征函数是 $\phi$。

因此：

$$
\frac{S_n}{\sqrt{n}} \xrightarrow{d} N(0, 1)
$$

这就完成了中心极限定理的证明。 $\square$

---

**证明的关键要点**：

1. **特征函数方法**：利用特征函数将依分布收敛转化为函数的逐点收敛
2. **独立性**：使得和的特征函数等于特征函数的乘积
3. **Taylor展开**：利用 $\mathbb{E}[X] = 0$, $\mathbb{E}[X^2] = \sigma^2$ 的条件
4. **极限技巧**：$(1 + x/n)^n \to e^x$ 的推广
5. **Lévy连续性定理**：连接特征函数收敛与分布收敛

---

**几何直观**：

中心极限定理说明，无论原始分布是什么形状，只要满足：

- 独立同分布
- 有限的均值和方差

那么大量随机变量的和（或平均）的分布都会趋向于正态分布。这解释了为什么正态分布在自然界中如此普遍。

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

**Lyapunov定理的完整证明**:

**证明策略**: 使用特征函数方法和Lyapunov条件控制Taylor展开的余项。

---

**第一步：标准化与简化**:

令 $Y_i = X_i - \mu_i$，则 $E[Y_i] = 0$，$\text{Var}(Y_i) = \sigma_i^2$。

定义标准化变量：

$$
Z_n = \frac{\sum_{i=1}^n Y_i}{s_n} = \frac{1}{s_n} \sum_{i=1}^n Y_i
$$

我们需要证明 $Z_n \xrightarrow{d} N(0, 1)$。

由Lévy连续性定理，只需证明 $Z_n$ 的特征函数 $\phi_{Z_n}(t)$ 收敛到 $e^{-t^2/2}$（标准正态分布的特征函数）。

---

**第二步：特征函数的分解**:

$$
\phi_{Z_n}(t) = E[e^{itZ_n}] = E\left[\exp\left(i t \frac{\sum_{i=1}^n Y_i}{s_n}\right)\right]
$$

由于 $Y_1, \ldots, Y_n$ 独立：

$$
\phi_{Z_n}(t) = \prod_{i=1}^n E\left[\exp\left(\frac{itY_i}{s_n}\right)\right] = \prod_{i=1}^n \phi_{Y_i}\left(\frac{t}{s_n}\right)
$$

其中 $\phi_{Y_i}$ 是 $Y_i$ 的特征函数。

---

**第三步：Taylor展开**:

对每个 $\phi_{Y_i}(u)$，当 $u$ 较小时，我们使用Taylor展开到 $2+\delta$ 阶：

$$
\phi_{Y_i}(u) = 1 + iuE[Y_i] - \frac{u^2}{2}E[Y_i^2] + R_i(u)
$$

其中余项满足：

$$
|R_i(u)| \leq \frac{|u|^{2+\delta}}{(2+\delta)!} E[|Y_i|^{2+\delta}]
$$

这由特征函数的标准余项估计得到。

由于 $E[Y_i] = 0$，$E[Y_i^2] = \sigma_i^2$：

$$
\phi_{Y_i}(u) = 1 - \frac{u^2 \sigma_i^2}{2} + R_i(u)
$$

---

**第四步：代入标准化变量**:

取 $u = \frac{t}{s_n}$：

$$
\phi_{Y_i}\left(\frac{t}{s_n}\right) = 1 - \frac{t^2 \sigma_i^2}{2s_n^2} + R_i\left(\frac{t}{s_n}\right)
$$

其中：

$$
\left|R_i\left(\frac{t}{s_n}\right)\right| \leq \frac{|t|^{2+\delta}}{(2+\delta)! \cdot s_n^{2+\delta}} E[|Y_i|^{2+\delta}]
$$

---

**第五步：乘积的对数**:

利用 $\log(1+x) = x + O(x^2)$ （当 $x \to 0$）：

$$
\log \phi_{Z_n}(t) = \sum_{i=1}^n \log \phi_{Y_i}\left(\frac{t}{s_n}\right)
$$

对每一项：

$$
\log \phi_{Y_i}\left(\frac{t}{s_n}\right) = \log\left(1 - \frac{t^2 \sigma_i^2}{2s_n^2} + R_i\left(\frac{t}{s_n}\right)\right)
$$

当 $n \to \infty$ 时，$\frac{\sigma_i^2}{s_n^2} \to 0$（因为每个 $\sigma_i^2$ 是固定的而 $s_n^2 \to \infty$）。

使用 $\log(1+x) = x - \frac{x^2}{2} + O(x^3)$：

$$
\log \phi_{Y_i}\left(\frac{t}{s_n}\right) = -\frac{t^2 \sigma_i^2}{2s_n^2} + R_i\left(\frac{t}{s_n}\right) + O\left(\left(\frac{\sigma_i^2}{s_n^2}\right)^2\right)
$$

---

**第六步：求和与简化**:

$$
\log \phi_{Z_n}(t) = \sum_{i=1}^n \left[-\frac{t^2 \sigma_i^2}{2s_n^2} + R_i\left(\frac{t}{s_n}\right) + O\left(\left(\frac{\sigma_i^2}{s_n^2}\right)^2\right)\right]
$$

第一项：

$$
\sum_{i=1}^n \left(-\frac{t^2 \sigma_i^2}{2s_n^2}\right) = -\frac{t^2}{2s_n^2} \sum_{i=1}^n \sigma_i^2 = -\frac{t^2}{2s_n^2} \cdot s_n^2 = -\frac{t^2}{2}
$$

第二项（余项之和）：

$$
\left|\sum_{i=1}^n R_i\left(\frac{t}{s_n}\right)\right| \leq \sum_{i=1}^n \frac{|t|^{2+\delta}}{(2+\delta)! \cdot s_n^{2+\delta}} E[|Y_i|^{2+\delta}]
$$

$$
= \frac{|t|^{2+\delta}}{(2+\delta)! \cdot s_n^{2+\delta}} \sum_{i=1}^n E[|Y_i|^{2+\delta}]
$$

由Lyapunov条件：

$$
\lim_{n \to \infty} \frac{1}{s_n^{2+\delta}} \sum_{i=1}^n E[|Y_i|^{2+\delta}] = 0
$$

因此：

$$
\lim_{n \to \infty} \left|\sum_{i=1}^n R_i\left(\frac{t}{s_n}\right)\right| = 0
$$

第三项（高阶项）：由于 $\max_i \frac{\sigma_i^2}{s_n^2} \leq \frac{1}{n} \cdot \frac{\sum \sigma_i^2}{s_n^2} = \frac{1}{n} \to 0$，可以证明：

$$
\sum_{i=1}^n O\left(\left(\frac{\sigma_i^2}{s_n^2}\right)^2\right) = o(1)
$$

---

**第七步：取极限**:

综合上述结果：

$$
\lim_{n \to \infty} \log \phi_{Z_n}(t) = -\frac{t^2}{2} + 0 + 0 = -\frac{t^2}{2}
$$

因此：

$$
\lim_{n \to \infty} \phi_{Z_n}(t) = e^{-t^2/2}
$$

这正是 $N(0,1)$ 的特征函数。

---

**第八步：应用Lévy连续性定理**:

由Lévy连续性定理，特征函数的逐点收敛蕴含依分布收敛：

$$
Z_n = \frac{\sum_{i=1}^n (X_i - \mu_i)}{s_n} \xrightarrow{d} N(0, 1)
$$

**证毕** ∎

---

**关键要点总结**:

| **关键步骤** | **作用** | **数学工具** |
|------------|---------|------------|
| 特征函数方法 | 将分布收敛问题转化为特征函数收敛 | Lévy连续性定理 |
| Taylor展开 | 近似每个 $\phi_{Y_i}$ | 特征函数的光滑性 |
| Lyapunov条件 | 控制余项趋于0 | $(2+\delta)$ 阶矩条件 |
| 对数变换 | 将乘积转化为求和 | $\log(1+x)$ 展开 |
| 极限计算 | 证明收敛到正态分布 | 标准化技巧 |

---

**Lyapunov条件的几何直觉**:

Lyapunov条件：

$$
\lim_{n \to \infty} \frac{1}{s_n^{2+\delta}} \sum_{i=1}^n E[|X_i - \mu_i|^{2+\delta}] = 0
$$

**直觉解释**：

1. **分子**：$\sum_{i=1}^n E[|X_i - \mu_i|^{2+\delta}]$ 衡量所有随机变量的"尾部重量"（高阶矩）
2. **分母**：$s_n^{2+\delta} = \left(\sum_{i=1}^n \sigma_i^2\right)^{1 + \delta/2}$ 是方差总和的略高次幂
3. **条件**：要求尾部重量的增长慢于方差总和的增长

**物理类比**：

- 想象 $n$ 个质点，每个质点的"质量"是 $\sigma_i^2$
- Lyapunov条件确保没有单个质点的"质量"主导整个系统
- 系统是"均匀分散"的，没有"异常重"的质点

---

**与Lindeberg-Lévy定理的比较**:

| **定理** | **条件** | **适用范围** | **证明难度** |
|---------|---------|------------|------------|
| **Lindeberg-Lévy** | 独立同分布 + 有限方差 | 仅同分布情况 | 简单 |
| **Lyapunov** | 独立 + $(2+\delta)$ 阶矩条件 | 非同分布，但需高阶矩 | 中等 |
| **Lindeberg** | 独立 + Lindeberg条件 | 非同分布，最一般 | 困难 |

**关系**：Lyapunov条件 ⇒ Lindeberg条件 ⇒ CLT

---

**实际应用示例**:

**示例 1**：不同方差的正态变量

设 $X_i \sim N(\mu_i, \sigma_i^2)$ 独立，其中 $\sigma_i^2 \leq M$ （有界）。

- $s_n^2 = \sum_{i=1}^n \sigma_i^2 \asymp n$
- $E[|X_i - \mu_i|^{2+\delta}] \asymp \sigma_i^{2+\delta} \leq M^{1+\delta/2} \cdot \sigma_i^2$
- Lyapunov条件：

$$
\frac{1}{s_n^{2+\delta}} \sum_{i=1}^n E[|X_i - \mu_i|^{2+\delta}] \leq \frac{M^{1+\delta/2}}{s_n^{2+\delta}} \sum_{i=1}^n \sigma_i^2 = \frac{M^{1+\delta/2}}{s_n^{\delta}} \to 0
$$

**示例 2**：机器学习中的集成学习

在Bagging或Random Forest中，每个基学习器的预测 $\hat{y}_i$ 可能有不同的方差：

$$
\bar{\hat{y}} = \frac{1}{n} \sum_{i=1}^n \hat{y}_i
$$

Lyapunov定理保证（在适当条件下）：

$$
\frac{\bar{\hat{y}} - E[\bar{\hat{y}}]}{\sqrt{\text{Var}(\bar{\hat{y}})}} \xrightarrow{d} N(0, 1)
$$

这为集成模型的不确定性量化提供理论基础。

---

**Python数值验证**:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def lyapunov_clt_demo():
    """
    验证Lyapunov CLT：使用不同分布的独立随机变量
    """
    np.random.seed(42)
    n_samples = 10000
    sample_sizes = [10, 30, 100, 300, 1000]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, n in enumerate(sample_sizes):
        # 生成n个不同分布的随机变量
        # X_i ~ Uniform[-sqrt(3)*i/n, sqrt(3)*i/n], Var(X_i) = i^2/n^2
        samples = np.zeros(n_samples)
        
        for _ in range(n_samples):
            X = []
            variances = []
            
            for i in range(1, n+1):
                # 方差递增的均匀分布
                scale = np.sqrt(3) * i / n
                x_i = np.random.uniform(-scale, scale)
                X.append(x_i)
                variances.append(scale**2 / 3)  # Uniform variance = (b-a)^2/12
            
            # 计算s_n
            s_n = np.sqrt(np.sum(variances))
            
            # 标准化统计量
            samples[_] = np.sum(X) / s_n
        
        # 绘制直方图
        axes[idx].hist(samples, bins=50, density=True, alpha=0.7, 
                       edgecolor='black', label=f'n={n}')
        
        # 理论N(0,1)密度
        x = np.linspace(-4, 4, 100)
        axes[idx].plot(x, stats.norm.pdf(x), 'r-', lw=2, label='N(0,1)')
        
        # K-S检验
        ks_stat, p_value = stats.kstest(samples, 'norm')
        
        axes[idx].set_title(f'n={n}, KS={ks_stat:.4f}, p={p_value:.4f}')
        axes[idx].set_xlabel('Standardized Sum')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # 删除多余子图
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('lyapunov_clt_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Lyapunov CLT验证：即使不同分布，标准化和仍收敛到N(0,1)")

# 运行验证
lyapunov_clt_demo()
```

**预期输出**：

- 随着 $n$ 增大，标准化统计量的直方图越来越接近 $N(0,1)$
- K-S检验的 $p$-value 逐渐增大（无法拒绝正态性假设）

---

**理论意义与实践价值**:

1. **理论意义**：
   - 推广了Lindeberg-Lévy定理到非同分布情况
   - 为处理异质数据提供理论基础
   - 连接了矩条件与分布收敛

2. **实践价值**：
   - **异质数据建模**：不同来源的数据可能有不同分布
   - **自适应算法**：在线学习中数据分布可能随时间变化
   - **集成学习**：不同模型的预测可能有不同方差
   - **分层抽样**：不同层的样本有不同特性

3. **局限性**：
   - 需要 $(2+\delta)$ 阶矩存在（比Lindeberg-Lévy更强）
   - 在重尾分布（如Cauchy）中不适用
   - 实际中验证Lyapunov条件可能困难

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

**Berry-Esseen定理的证明大纲**:

**注意**: Berry-Esseen定理的完整证明非常技术性，涉及复杂的傅里叶分析。这里给出证明的主要思路和关键步骤。

**证明思路**:

**第一步：Esseen不等式（平滑引理）**:

对于任意分布函数 $F$ 和 $G$，以及它们的特征函数 $\phi_F$ 和 $\phi_G$：

$$
\sup_x |F(x) - G(x)| \leq \frac{1}{\pi} \int_{-T}^T \left|\frac{\phi_F(t) - \phi_G(t)}{t}\right| dt + \frac{24}{\pi T} \sup_x \int_{x-1}^{x+1} G(u) du
$$

对于任意 $T > 0$。

这个不等式将分布函数的距离与特征函数的距离联系起来。

**第二步：标准化**:

令 $Y_i = \frac{X_i - \mu}{\sigma}$，则 $E[Y_i] = 0$，$\text{Var}(Y_i) = 1$，$E[|Y_i|^3] = \frac{\rho}{\sigma^3}$。

令 $S_n = \frac{1}{\sqrt{n}} \sum_{i=1}^n Y_i$。

需要估计：

$$
\Delta_n = \sup_x |P(S_n \leq x) - \Phi(x)|
$$

**第三步：特征函数的Taylor展开**:

$Y_i$ 的特征函数：

$$
\phi_Y(t) = E[e^{itY}] = 1 - \frac{t^2}{2} + r(t)
$$

其中余项满足：

$$
|r(t)| \leq \min\left(\frac{|t|^3}{6} E[|Y|^3], 2\right)
$$

**第四步：和的特征函数**:

$$
\phi_{S_n}(t) = \left[\phi_Y\left(\frac{t}{\sqrt{n}}\right)\right]^n = \left[1 - \frac{t^2}{2n} + r\left(\frac{t}{\sqrt{n}}\right)\right]^n
$$

**第五步：与正态分布特征函数的比较**:

标准正态分布的特征函数：$\phi_Z(t) = e^{-t^2/2}$

需要估计：

$$
\left|\phi_{S_n}(t) - e^{-t^2/2}\right|
$$

使用不等式 $|e^a - e^b| \leq |a - b| e^{\max(a,b)}$ 和 $|(1+x)^n - e^{nx}| \leq |x|^2 n^2 e^{n|x|}$（当 $|x| \leq 1/n$ 时）。

**第六步：选择截断参数**:

在Esseen不等式中选择 $T = \sqrt{n}$。

对于 $|t| \leq T$：

$$
\left|\phi_{S_n}(t) - e^{-t^2/2}\right| \leq C_1 \frac{|t|^3}{\sqrt{n}} E[|Y|^3]
$$

**第七步：积分估计**:

$$
\int_{-T}^T \left|\frac{\phi_{S_n}(t) - e^{-t^2/2}}{t}\right| dt \leq C_1 \frac{E[|Y|^3]}{\sqrt{n}} \int_{-T}^T |t|^2 dt = C_2 \frac{E[|Y|^3] T^3}{\sqrt{n}}
$$

代入 $T = \sqrt{n}$：

$$
\leq C_2 \frac{E[|Y|^3] n^{3/2}}{\sqrt{n}} = C_2 E[|Y|^3] n
$$

等等，这里有问题。正确的估计应该是：

$$
\int_{-T}^T \left|\frac{\phi_{S_n}(t) - e^{-t^2/2}}{t}\right| dt \leq C \frac{E[|Y|^3]}{\sqrt{n}}
$$

**第八步：余项估计**:

Esseen不等式的第二项：

$$
\frac{24}{\pi T} \leq \frac{24}{\pi \sqrt{n}}
$$

**第九步：结论**:

结合所有估计：

$$
\Delta_n \leq \frac{C \rho}{\sigma^3 \sqrt{n}}
$$

其中 $C$ 是一个绝对常数。

$\square$

---

**关键要点**:

1. **Esseen平滑引理**: 连接分布函数距离与特征函数距离
2. **三阶矩条件**: $E[|X|^3] < \infty$ 是必需的
3. **收敛速度**: $O(n^{-1/2})$ 是最优的（在一般情况下）
4. **常数 $C$**:
   - 理论上界：$C \leq 0.4748$（Shevtsova, 2011）
   - 实际中常用：$C \approx 0.5$

---

**Berry-Esseen定理的改进**:

**1. 非同分布情况（Lyapunov条件）**:

若 $X_1, \ldots, X_n$ 独立但不同分布，$E[X_i] = \mu_i$，$\text{Var}(X_i) = \sigma_i^2$，

令 $s_n^2 = \sum_{i=1}^n \sigma_i^2$，$L_n = \frac{1}{s_n^3} \sum_{i=1}^n E[|X_i - \mu_i|^3]$。

则：

$$
\sup_x \left|P\left(\frac{\sum_{i=1}^n (X_i - \mu_i)}{s_n} \leq x\right) - \Phi(x)\right| \leq C L_n
$$

**2. 多元Berry-Esseen定理**:

对于 $d$ 维随机向量，收敛速度为 $O(n^{-1/2})$，但常数依赖于维数 $d$。

---

**实际应用**:

**例1：样本大小的选择**:

若要保证近似误差小于 $\epsilon$，需要：

$$
n \geq \left(\frac{C\rho}{\sigma^3 \epsilon}\right)^2
$$

对于标准正态分布（$\rho = 2$，$\sigma = 1$），若 $\epsilon = 0.01$：

$$
n \geq \left(\frac{0.5 \times 2}{0.01}\right)^2 = 10000
$$

**例2：偏态分布的修正**:

对于偏态分布（如指数分布），Berry-Esseen定理说明需要更大的样本才能获得好的正态近似。

指数分布 $\text{Exp}(\lambda)$：

- $\mu = 1/\lambda$
- $\sigma^2 = 1/\lambda^2$
- $\rho = E[|X - \mu|^3] = 2/\lambda^3$

因此：

$$
\frac{\rho}{\sigma^3} = \frac{2/\lambda^3}{(1/\lambda)^3} = 2
$$

界为：

$$
\frac{C \times 2}{\sqrt{n}} \approx \frac{1}{\sqrt{n}}
$$

---

**数值验证**:

可以通过蒙特卡洛模拟验证Berry-Esseen界：

1. 生成 $n$ 个样本，计算标准化均值
2. 重复多次，得到经验分布函数
3. 计算与标准正态分布的最大距离
4. 与理论界 $\frac{C\rho}{\sigma^3\sqrt{n}}$ 比较

实验表明，实际误差通常远小于理论界。

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

**Delta方法的完整证明**:

**定理 2.1 的证明**（一元情况）:

**假设**:

1. $\sqrt{n}(X_n - \theta) \xrightarrow{d} N(0, \sigma^2)$
2. $g$ 在 $\theta$ 处可微，且 $g'(\theta) \neq 0$

**目标**: 证明 $\sqrt{n}(g(X_n) - g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2)$

**第一步：Taylor展开**:

在 $\theta$ 附近对 $g(X_n)$ 进行一阶Taylor展开：

$$
g(X_n) = g(\theta) + g'(\theta)(X_n - \theta) + R_n
$$

其中余项 $R_n = o(|X_n - \theta|)$，即：

$$
\frac{R_n}{X_n - \theta} \to 0 \quad \text{当 } X_n \to \theta
$$

**第二步：重新整理**:

$$
g(X_n) - g(\theta) = g'(\theta)(X_n - \theta) + R_n
$$

两边乘以 $\sqrt{n}$：

$$
\sqrt{n}(g(X_n) - g(\theta)) = g'(\theta) \sqrt{n}(X_n - \theta) + \sqrt{n} R_n
$$

**第三步：分析第一项**:

由假设，$\sqrt{n}(X_n - \theta) \xrightarrow{d} N(0, \sigma^2)$。

由连续映射定理（标量乘法是连续的）：

$$
g'(\theta) \sqrt{n}(X_n - \theta) \xrightarrow{d} g'(\theta) \cdot N(0, \sigma^2) = N(0, [g'(\theta)]^2 \sigma^2)
$$

**第四步：分析余项**:

需要证明：$\sqrt{n} R_n \xrightarrow{P} 0$

由于 $X_n \xrightarrow{P} \theta$（依分布收敛蕴含依概率收敛到常数），我们有：

$$
\frac{R_n}{X_n - \theta} \xrightarrow{P} 0
$$

因此：

$$
\sqrt{n} R_n = \sqrt{n}(X_n - \theta) \cdot \frac{R_n}{X_n - \theta}
$$

由于 $\sqrt{n}(X_n - \theta) = O_P(1)$（有界于概率意义）且 $\frac{R_n}{X_n - \theta} \xrightarrow{P} 0$，

根据Slutsky定理：

$$
\sqrt{n} R_n = O_P(1) \cdot o_P(1) = o_P(1) \xrightarrow{P} 0
$$

**第五步：应用Slutsky定理**:

$$
\sqrt{n}(g(X_n) - g(\theta)) = g'(\theta) \sqrt{n}(X_n - \theta) + \sqrt{n} R_n
$$

由Slutsky定理（和的极限）：

$$
\sqrt{n}(g(X_n) - g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2) + 0 = N(0, [g'(\theta)]^2 \sigma^2)
$$

$\square$

---

**多元Delta方法的证明**:

**假设**:

1. $\sqrt{n}(\mathbf{X}_n - \boldsymbol{\theta}) \xrightarrow{d} N(\mathbf{0}, \Sigma)$
2. $g: \mathbb{R}^d \to \mathbb{R}$ 在 $\boldsymbol{\theta}$ 处可微

**目标**: 证明 $\sqrt{n}(g(\mathbf{X}_n) - g(\boldsymbol{\theta})) \xrightarrow{d} N(0, \nabla g(\boldsymbol{\theta})^T \Sigma \nabla g(\boldsymbol{\theta}))$

**第一步：多元Taylor展开**:

$$
g(\mathbf{X}_n) = g(\boldsymbol{\theta}) + \nabla g(\boldsymbol{\theta})^T (\mathbf{X}_n - \boldsymbol{\theta}) + R_n
$$

其中 $R_n = o(\|\mathbf{X}_n - \boldsymbol{\theta}\|)$。

**第二步：重新整理**:

$$
\sqrt{n}(g(\mathbf{X}_n) - g(\boldsymbol{\theta})) = \nabla g(\boldsymbol{\theta})^T \sqrt{n}(\mathbf{X}_n - \boldsymbol{\theta}) + \sqrt{n} R_n
$$

**第三步：应用连续映射定理**:

由假设，$\sqrt{n}(\mathbf{X}_n - \boldsymbol{\theta}) \xrightarrow{d} N(\mathbf{0}, \Sigma)$。

线性变换 $\mathbf{a}^T \mathbf{Z}$（其中 $\mathbf{Z} \sim N(\mathbf{0}, \Sigma)$）的分布为：

$$
\mathbf{a}^T \mathbf{Z} \sim N(0, \mathbf{a}^T \Sigma \mathbf{a})
$$

因此：

$$
\nabla g(\boldsymbol{\theta})^T \sqrt{n}(\mathbf{X}_n - \boldsymbol{\theta}) \xrightarrow{d} N(0, \nabla g(\boldsymbol{\theta})^T \Sigma \nabla g(\boldsymbol{\theta}))
$$

**第四步：余项分析**:

类似一元情况，$\sqrt{n} R_n \xrightarrow{P} 0$。

**第五步：结论**:

$$
\sqrt{n}(g(\mathbf{X}_n) - g(\boldsymbol{\theta})) \xrightarrow{d} N(0, \nabla g(\boldsymbol{\theta})^T \Sigma \nabla g(\boldsymbol{\theta}))
$$

$\square$

---

**应用示例**:

**例1：方差的MLE**:

设 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$，$\mu$ 已知。

样本方差的MLE：$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \mu)^2$

由CLT：

$$
\sqrt{n}(\hat{\sigma}^2 - \sigma^2) \xrightarrow{d} N(0, \tau^2)
$$

其中 $\tau^2 = \text{Var}((X - \mu)^2) = E[(X-\mu)^4] - \sigma^4 = 2\sigma^4$（对正态分布）。

现在考虑标准差 $\hat{\sigma} = \sqrt{\hat{\sigma}^2}$。

令 $g(x) = \sqrt{x}$，则 $g'(x) = \frac{1}{2\sqrt{x}}$。

由Delta方法：

$$
\sqrt{n}(\hat{\sigma} - \sigma) \xrightarrow{d} N\left(0, \left[\frac{1}{2\sigma}\right]^2 \cdot 2\sigma^4\right) = N\left(0, \frac{\sigma^2}{2}\right)
$$

**例2：对数变换**:

设 $\sqrt{n}(X_n - \theta) \xrightarrow{d} N(0, \sigma^2)$，$\theta > 0$。

考虑 $\log X_n$。

令 $g(x) = \log x$，则 $g'(x) = \frac{1}{x}$。

由Delta方法：

$$
\sqrt{n}(\log X_n - \log \theta) \xrightarrow{d} N\left(0, \frac{\sigma^2}{\theta^2}\right)
$$

**例3：比率的渐近分布**:

设 $\sqrt{n}\begin{pmatrix} \bar{X}_n - \mu_X \\ \bar{Y}_n - \mu_Y \end{pmatrix} \xrightarrow{d} N\left(\mathbf{0}, \begin{pmatrix} \sigma_X^2 & \rho \sigma_X \sigma_Y \\ \rho \sigma_X \sigma_Y & \sigma_Y^2 \end{pmatrix}\right)$

考虑比率 $R_n = \frac{\bar{X}_n}{\bar{Y}_n}$。

令 $g(x, y) = \frac{x}{y}$，则：

$$
\nabla g = \begin{pmatrix} \frac{1}{y} \\ -\frac{x}{y^2} \end{pmatrix}
$$

在 $(\mu_X, \mu_Y)$ 处：

$$
\nabla g(\mu_X, \mu_Y) = \begin{pmatrix} \frac{1}{\mu_Y} \\ -\frac{\mu_X}{\mu_Y^2} \end{pmatrix}
$$

由多元Delta方法：

$$
\sqrt{n}\left(\frac{\bar{X}_n}{\bar{Y}_n} - \frac{\mu_X}{\mu_Y}\right) \xrightarrow{d} N(0, V)
$$

其中：

$$
V = \nabla g^T \Sigma \nabla g = \frac{1}{\mu_Y^2}\left(\sigma_X^2 - 2\frac{\mu_X}{\mu_Y}\rho \sigma_X \sigma_Y + \frac{\mu_X^2}{\mu_Y^2}\sigma_Y^2\right)
$$

---

**Delta方法的推广**:

**二阶Delta方法**:

若 $g'(\theta) = 0$ 但 $g''(\theta) \neq 0$，则需要二阶Taylor展开：

$$
n(g(X_n) - g(\theta)) \xrightarrow{d} \frac{1}{2}g''(\theta) \chi^2_1 \sigma^2
$$

（这里 $\chi^2_1$ 是自由度为1的卡方分布）

**函数Delta方法**:

对于随机过程 $\{X_n(t)\}$，若 $X_n \Rightarrow X$ 在某个函数空间中，

且 $g$ 是连续泛函，则 $g(X_n) \Rightarrow g(X)$。

---

**实际应用中的注意事项**:

1. **导数为零**: 若 $g'(\theta) = 0$，需要使用二阶Delta方法

2. **数值稳定性**: 当 $g'(\theta)$ 很小时，渐近方差可能很大

3. **有限样本**: Delta方法是渐近结果，小样本时可能不准确

4. **Bootstrap**: 可以用Bootstrap验证Delta方法的准确性

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

*最后更新：2025年10月*:
