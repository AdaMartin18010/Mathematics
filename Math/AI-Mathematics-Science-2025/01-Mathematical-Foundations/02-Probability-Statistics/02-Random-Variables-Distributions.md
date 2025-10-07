# 随机变量与分布 (Random Variables and Distributions)

> **The Foundation of Probabilistic Machine Learning**
>
> 概率机器学习的基础

---

## 目录

- [随机变量与分布 (Random Variables and Distributions)](#随机变量与分布-random-variables-and-distributions)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 随机变量](#-随机变量)
    - [1. 随机变量定义](#1-随机变量定义)
    - [2. 分布函数](#2-分布函数)
    - [3. 概率密度函数](#3-概率密度函数)
  - [📊 常见分布](#-常见分布)
    - [1. 离散分布](#1-离散分布)
    - [2. 连续分布](#2-连续分布)
  - [🔬 期望与方差](#-期望与方差)
    - [1. 期望](#1-期望)
    - [2. 方差](#2-方差)
    - [3. 协方差与相关系数](#3-协方差与相关系数)
  - [💡 多元随机变量](#-多元随机变量)
    - [1. 联合分布](#1-联合分布)
    - [2. 边缘分布](#2-边缘分布)
    - [3. 条件分布](#3-条件分布)
    - [4. 独立性](#4-独立性)
  - [🎨 变换与矩母函数](#-变换与矩母函数)
    - [1. 随机变量的变换](#1-随机变量的变换)
    - [2. 矩母函数](#2-矩母函数)
    - [3. 特征函数](#3-特征函数)
  - [🔧 在机器学习中的应用](#-在机器学习中的应用)
    - [1. 贝叶斯推断](#1-贝叶斯推断)
    - [2. 最大似然估计](#2-最大似然估计)
    - [3. 变分推断](#3-变分推断)
    - [4. 采样方法](#4-采样方法)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：分布计算](#练习1分布计算)
    - [练习2：期望与方差](#练习2期望与方差)
    - [练习3：变换](#练习3变换)
    - [练习4：最大似然估计](#练习4最大似然估计)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**随机变量**是将随机事件映射到实数的函数，是概率论的核心概念。

**为什么随机变量重要**:

```text
机器学习中的随机性:
├─ 数据本身是随机的
├─ 模型参数是随机的 (贝叶斯观点)
├─ 训练过程是随机的 (SGD)
└─ 预测是概率性的

核心应用:
├─ 贝叶斯推断
├─ 最大似然估计
├─ 变分推断
└─ 生成模型 (VAE, GAN)
```

---

## 🎯 随机变量

### 1. 随机变量定义

**定义 1.1 (随机变量)**:

设 $(\Omega, \mathcal{F}, P)$ 是概率空间，随机变量 $X$ 是从 $\Omega$ 到 $\mathbb{R}$ 的**可测函数**：

$$
X: \Omega \to \mathbb{R}
$$

使得对于任意 $a \in \mathbb{R}$，集合 $\{X \leq a\} = \{\omega \in \Omega : X(\omega) \leq a\} \in \mathcal{F}$。

**直觉**：随机变量将随机事件的结果映射到数值。

**示例**:

- 掷骰子：$X(\omega) = \omega$ (点数)
- 抛硬币：$X(\text{正}) = 1, X(\text{反}) = 0$

---

### 2. 分布函数

**定义 2.1 (累积分布函数, CDF)**:

随机变量 $X$ 的累积分布函数 $F_X: \mathbb{R} \to [0, 1]$ 定义为：

$$
F_X(x) = P(X \leq x)
$$

**性质**:

1. **单调性**: $F_X$ 是非递减的
2. **右连续性**: $\lim_{x \to a^+} F_X(x) = F_X(a)$
3. **极限性**: $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$

---

**性质的完整证明**:

**证明1：单调性**:

需要证明：若 $x_1 \leq x_2$，则 $F_X(x_1) \leq F_X(x_2)$

**证明**:

设 $x_1 \leq x_2$。注意到：

$$
\{X \leq x_1\} \subseteq \{X \leq x_2\}
$$

因为如果 $X(\omega) \leq x_1$，则 $X(\omega) \leq x_1 \leq x_2$，所以 $X(\omega) \leq x_2$。

由概率的单调性（如果 $A \subseteq B$，则 $P(A) \leq P(B)$）：

$$
F_X(x_1) = P(X \leq x_1) \leq P(X \leq x_2) = F_X(x_2)
$$

$\square$

---

**证明2：右连续性**:

需要证明：$\lim_{x \to a^+} F_X(x) = F_X(a)$

**证明**:

考虑递减序列 $x_n \downarrow a$（即 $x_1 > x_2 > \cdots > a$ 且 $\lim_{n \to \infty} x_n = a$）。

定义事件序列：

$$
A_n = \{X \leq x_n\}
$$

**关键观察**:

1. $A_1 \supseteq A_2 \supseteq A_3 \supseteq \cdots$ （递减序列）

2. $\bigcap_{n=1}^\infty A_n = \{X \leq a\}$

**证明第2点**:

- 若 $\omega \in \bigcap_{n=1}^\infty A_n$，则对所有 $n$，$X(\omega) \leq x_n$。
  
  取极限：$X(\omega) \leq \lim_{n \to \infty} x_n = a$，所以 $\omega \in \{X \leq a\}$。

- 反之，若 $\omega \in \{X \leq a\}$，则 $X(\omega) \leq a < x_n$ 对所有 $n$（因为 $x_n > a$）。
  
  所以 $\omega \in A_n$ 对所有 $n$，即 $\omega \in \bigcap_{n=1}^\infty A_n$。

**应用测度的连续性**:

对于递减事件序列，概率测度的连续性给出：

$$
\lim_{n \to \infty} P(A_n) = P\left(\bigcap_{n=1}^\infty A_n\right)
$$

即：

$$
\lim_{n \to \infty} F_X(x_n) = \lim_{n \to \infty} P(X \leq x_n) = P(X \leq a) = F_X(a)
$$

由于这对任意递减序列 $x_n \downarrow a$ 成立，我们有：

$$
\lim_{x \to a^+} F_X(x) = F_X(a)
$$

$\square$

**注意**: CDF一般不是左连续的。左极限为：

$$
\lim_{x \to a^-} F_X(x) = P(X < a) = F_X(a) - P(X = a)
$$

如果 $P(X = a) > 0$（即 $a$ 是原子点），则CDF在 $a$ 处有跳跃。

---

**证明3：极限性**:

需要证明：
1. $\lim_{x \to -\infty} F_X(x) = 0$
2. $\lim_{x \to \infty} F_X(x) = 1$

**证明 (1)**:

考虑递减序列 $x_n \to -\infty$（例如 $x_n = -n$）。

定义事件序列：

$$
A_n = \{X \leq x_n\}
$$

则 $A_1 \supseteq A_2 \supseteq A_3 \supseteq \cdots$（递减）。

**关键观察**: $\bigcap_{n=1}^\infty A_n = \emptyset$

**证明**:

假设存在 $\omega \in \bigcap_{n=1}^\infty A_n$，则对所有 $n$，$X(\omega) \leq x_n$。

但 $x_n \to -\infty$，这意味着 $X(\omega) \leq -n$ 对所有 $n$，即 $X(\omega) = -\infty$。

这与 $X$ 是实值随机变量矛盾（$X: \Omega \to \mathbb{R}$）。

因此 $\bigcap_{n=1}^\infty A_n = \emptyset$。

**应用测度的连续性**:

$$
\lim_{n \to \infty} P(A_n) = P\left(\bigcap_{n=1}^\infty A_n\right) = P(\emptyset) = 0
$$

即：

$$
\lim_{x \to -\infty} F_X(x) = 0
$$

$\square$

**证明 (2)**:

考虑递增序列 $x_n \to \infty$（例如 $x_n = n$）。

定义事件序列：

$$
B_n = \{X \leq x_n\}
$$

则 $B_1 \subseteq B_2 \subseteq B_3 \subseteq \cdots$（递增）。

**关键观察**: $\bigcup_{n=1}^\infty B_n = \Omega$

**证明**:

对于任意 $\omega \in \Omega$，$X(\omega)$ 是有限实数。

选择 $n$ 足够大使得 $x_n > X(\omega)$，则 $\omega \in B_n$。

因此 $\omega \in \bigcup_{n=1}^\infty B_n$，所以 $\Omega \subseteq \bigcup_{n=1}^\infty B_n$。

反向包含显然，因此 $\bigcup_{n=1}^\infty B_n = \Omega$。

**应用测度的连续性**:

$$
\lim_{n \to \infty} P(B_n) = P\left(\bigcup_{n=1}^\infty B_n\right) = P(\Omega) = 1
$$

即：

$$
\lim_{x \to \infty} F_X(x) = 1
$$

$\square$

---

**性质的几何意义**:

1. **单调性**: CDF是阶梯函数或连续增函数，永不下降
2. **右连续性**: CDF从右侧逼近每个点的值，跳跃发生在左侧
3. **极限性**: CDF从0开始，最终达到1，反映了概率的归一化

**应用示例**:

考虑离散随机变量 $X \sim \text{Bernoulli}(p)$：

$$
F_X(x) = \begin{cases}
0 & x < 0 \\
1-p & 0 \leq x < 1 \\
1 & x \geq 1
\end{cases}
$$

验证性质：
- 单调性: $0 \leq 1-p \leq 1$ ✓
- 右连续性: 在 $x=0$ 和 $x=1$ 处右连续 ✓
- 极限性: $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$ ✓
- 跳跃: 在 $x=0$ 处跳跃 $1-p$，在 $x=1$ 处跳跃 $p$

---

### 3. 概率密度函数

**定义 3.1 (概率密度函数, PDF)**:

如果存在非负函数 $f_X: \mathbb{R} \to [0, \infty)$ 使得：

$$
F_X(x) = \int_{-\infty}^x f_X(t) \, dt
$$

则称 $X$ 为**连续随机变量**，$f_X$ 为其**概率密度函数**。

**性质**:

1. $f_X(x) \geq 0$
2. $\int_{-\infty}^\infty f_X(x) \, dx = 1$
3. $P(a \leq X \leq b) = \int_a^b f_X(x) \, dx$

**离散情况**:

对于离散随机变量，使用**概率质量函数 (PMF)**：

$$
p_X(x) = P(X = x)
$$

---

## 📊 常见分布

### 1. 离散分布

**伯努利分布** (Bernoulli):

$$
X \sim \text{Bernoulli}(p)
$$

$$
P(X = 1) = p, \quad P(X = 0) = 1 - p
$$

**期望**: $E[X] = p$  
**方差**: $\text{Var}(X) = p(1-p)$

**应用**: 二分类问题

---

**二项分布** (Binomial):

$$
X \sim \text{Binomial}(n, p)
$$

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

**期望**: $E[X] = np$  
**方差**: $\text{Var}(X) = np(1-p)$

**应用**: $n$ 次独立伯努利试验中成功的次数

---

**泊松分布** (Poisson):

$$
X \sim \text{Poisson}(\lambda)
$$

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

**期望**: $E[X] = \lambda$  
**方差**: $\text{Var}(X) = \lambda$

**应用**: 单位时间内事件发生的次数

---

### 2. 连续分布

**均匀分布** (Uniform):

$$
X \sim \text{Uniform}(a, b)
$$

$$
f_X(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$

**期望**: $E[X] = \frac{a+b}{2}$  
**方差**: $\text{Var}(X) = \frac{(b-a)^2}{12}$

---

**正态分布** (Gaussian):

$$
X \sim \mathcal{N}(\mu, \sigma^2)
$$

$$
f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**期望**: $E[X] = \mu$  
**方差**: $\text{Var}(X) = \sigma^2$

**性质**:

- **对称性**: 关于 $\mu$ 对称
- **线性组合**: 正态分布的线性组合仍是正态分布
- **中心极限定理**: 独立同分布随机变量的和趋向于正态分布

**应用**: 最常用的分布，噪声建模

---

**指数分布** (Exponential):

$$
X \sim \text{Exp}(\lambda)
$$

$$
f_X(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

**期望**: $E[X] = \frac{1}{\lambda}$  
**方差**: $\text{Var}(X) = \frac{1}{\lambda^2}$

**性质**: **无记忆性** $P(X > s + t | X > s) = P(X > t)$

**应用**: 等待时间建模

---

## 🔬 期望与方差

### 1. 期望

**定义 1.1 (期望)**:

**离散情况**:

$$
E[X] = \sum_x x \cdot P(X = x)
$$

**连续情况**:

$$
E[X] = \int_{-\infty}^\infty x \cdot f_X(x) \, dx
$$

**性质**:

1. **线性性**: $E[aX + bY] = aE[X] + bE[Y]$
2. **非负性**: 如果 $X \geq 0$，则 $E[X] \geq 0$
3. **单调性**: 如果 $X \leq Y$，则 $E[X] \leq E[Y]$

**函数的期望**:

$$
E[g(X)] = \int_{-\infty}^\infty g(x) \cdot f_X(x) \, dx
$$

---

### 2. 方差

**定义 2.1 (方差)**:

$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$

**标准差**: $\sigma_X = \sqrt{\text{Var}(X)}$

**性质**:

1. $\text{Var}(aX + b) = a^2 \text{Var}(X)$
2. $\text{Var}(X) \geq 0$
3. 如果 $X, Y$ 独立，则 $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

---

### 3. 协方差与相关系数

**定义 3.1 (协方差)**:

$$
\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
$$

**性质**:

1. $\text{Cov}(X, X) = \text{Var}(X)$
2. $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
3. $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$

**相关系数**:

$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

**性质**: $-1 \leq \rho_{XY} \leq 1$

- $\rho_{XY} = 1$: 完全正相关
- $\rho_{XY} = -1$: 完全负相关
- $\rho_{XY} = 0$: 不相关（但不一定独立）

---

## 💡 多元随机变量

### 1. 联合分布

**定义 1.1 (联合分布)**:

对于随机向量 $(X, Y)$，其联合分布函数：

$$
F_{X,Y}(x, y) = P(X \leq x, Y \leq y)
$$

**联合密度函数**:

$$
f_{X,Y}(x, y) = \frac{\partial^2 F_{X,Y}}{\partial x \partial y}
$$

**性质**:

$$
\int_{-\infty}^\infty \int_{-\infty}^\infty f_{X,Y}(x, y) \, dx \, dy = 1
$$

---

### 2. 边缘分布

**定义 2.1 (边缘分布)**:

从联合分布得到单个变量的分布：

$$
f_X(x) = \int_{-\infty}^\infty f_{X,Y}(x, y) \, dy
$$

$$
f_Y(y) = \int_{-\infty}^\infty f_{X,Y}(x, y) \, dx
$$

---

### 3. 条件分布

**定义 3.1 (条件分布)**:

给定 $Y = y$ 时，$X$ 的条件密度：

$$
f_{X|Y}(x|y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}
$$

**条件期望**:

$$
E[X | Y = y] = \int_{-\infty}^\infty x \cdot f_{X|Y}(x|y) \, dx
$$

**全期望公式**:

$$
E[X] = E[E[X | Y]]
$$

---

### 4. 独立性

**定义 4.1 (独立性)**:

随机变量 $X$ 和 $Y$ 独立，如果：

$$
f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y)
$$

**等价条件**:

- $P(X \in A, Y \in B) = P(X \in A) \cdot P(Y \in B)$
- $E[XY] = E[X] \cdot E[Y]$
- $\text{Cov}(X, Y) = 0$ (但反之不一定成立)

---

## 🎨 变换与矩母函数

### 1. 随机变量的变换

**定理 1.1 (单调变换)**:

设 $Y = g(X)$，其中 $g$ 是严格单调函数，则：

$$
f_Y(y) = f_X(g^{-1}(y)) \left| \frac{d g^{-1}}{dy}(y) \right|
$$

**示例**: 如果 $X \sim \mathcal{N}(0, 1)$，$Y = X^2$，则 $Y \sim \chi^2_1$。

---

### 2. 矩母函数

**定义 2.1 (矩母函数, MGF)**:

$$
M_X(t) = E[e^{tX}]
$$

**性质**:

1. **唯一性**: MGF唯一确定分布
2. **矩的计算**: $E[X^n] = M_X^{(n)}(0)$
3. **独立和**: 如果 $X, Y$ 独立，则 $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$

**示例** (正态分布):

$$
X \sim \mathcal{N}(\mu, \sigma^2) \Rightarrow M_X(t) = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)
$$

---

**MGF性质的完整证明**:

**性质1：唯一性定理**

**定理**: 若两个随机变量 $X$ 和 $Y$ 的MGF在0的某个邻域内存在且相等，则 $X$ 和 $Y$ 有相同的分布。

**证明思路**（完整证明需要复分析）:

MGF与特征函数的关系：$M_X(t) = \phi_X(-it)$（当MGF存在时）。

特征函数唯一确定分布（Lévy唯一性定理）。

因此，若 $M_X(t) = M_Y(t)$ 在0的邻域内，则 $\phi_X(s) = \phi_Y(s)$ 对纯虚数 $s = -it$ 成立。

由解析延拓，这意味着 $\phi_X = \phi_Y$ 处处成立。

因此 $X$ 和 $Y$ 有相同的分布。 $\square$

**注意**: 这个定理的完整证明需要复分析和测度论的深入知识。在实践中，我们通常直接引用这个结果。

---

**性质2：矩的计算**

**定理**: 若 $M_X(t)$ 在0的邻域内存在，则：

$$
E[X^n] = M_X^{(n)}(0) = \frac{d^n M_X}{dt^n}\bigg|_{t=0}
$$

**证明**:

**第一步：Taylor展开**

假设可以交换期望和求导（在MGF存在的条件下通常成立）：

$$
M_X(t) = E[e^{tX}] = E\left[\sum_{n=0}^\infty \frac{(tX)^n}{n!}\right]
$$

交换期望和求和（由单调收敛定理或控制收敛定理）：

$$
M_X(t) = \sum_{n=0}^\infty \frac{E[X^n]}{n!} t^n
$$

这是 $M_X(t)$ 在 $t=0$ 处的Taylor展开。

**第二步：求导**

对 $M_X(t)$ 求 $n$ 阶导数：

$$
M_X^{(n)}(t) = \frac{d^n}{dt^n} E[e^{tX}]
$$

交换期望和求导（在适当条件下）：

$$
M_X^{(n)}(t) = E\left[\frac{d^n}{dt^n} e^{tX}\right] = E[X^n e^{tX}]
$$

**第三步：在 $t=0$ 处求值**

$$
M_X^{(n)}(0) = E[X^n e^{0 \cdot X}] = E[X^n]
$$

$\square$

**详细示例**:

对于 $X \sim \mathcal{N}(0, 1)$，$M_X(t) = e^{t^2/2}$。

**一阶矩**:

$$
M_X'(t) = e^{t^2/2} \cdot t
$$

$$
E[X] = M_X'(0) = 0
$$

**二阶矩**:

$$
M_X''(t) = e^{t^2/2} \cdot (1 + t^2)
$$

$$
E[X^2] = M_X''(0) = 1
$$

**三阶矩**:

$$
M_X'''(t) = e^{t^2/2} \cdot (3t + t^3)
$$

$$
E[X^3] = M_X'''(0) = 0
$$

**四阶矩**:

$$
M_X^{(4)}(t) = e^{t^2/2} \cdot (3 + 6t^2 + t^4)
$$

$$
E[X^4] = M_X^{(4)}(0) = 3
$$

这验证了标准正态分布的矩：$E[X] = 0$, $E[X^2] = 1$, $E[X^3] = 0$, $E[X^4] = 3$。

---

**性质3：独立和的MGF**

**定理**: 若 $X$ 和 $Y$ 独立，则：

$$
M_{X+Y}(t) = M_X(t) \cdot M_Y(t)
$$

**证明**:

$$
M_{X+Y}(t) = E[e^{t(X+Y)}] = E[e^{tX} \cdot e^{tY}]
$$

由于 $X$ 和 $Y$ 独立，$e^{tX}$ 和 $e^{tY}$ 也独立（独立随机变量的函数仍独立）。

因此，由期望的独立性性质：

$$
M_{X+Y}(t) = E[e^{tX}] \cdot E[e^{tY}] = M_X(t) \cdot M_Y(t)
$$

$\square$

**应用示例**:

**例1**: 正态分布的可加性

若 $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$，$Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ 独立，则：

$$
M_X(t) = \exp\left(\mu_1 t + \frac{\sigma_1^2 t^2}{2}\right)
$$

$$
M_Y(t) = \exp\left(\mu_2 t + \frac{\sigma_2^2 t^2}{2}\right)
$$

$$
M_{X+Y}(t) = M_X(t) \cdot M_Y(t) = \exp\left((\mu_1 + \mu_2) t + \frac{(\sigma_1^2 + \sigma_2^2) t^2}{2}\right)
$$

这是 $\mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$ 的MGF。

因此 $X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$。

**例2**: 泊松分布的可加性

若 $X \sim \text{Poisson}(\lambda_1)$，$Y \sim \text{Poisson}(\lambda_2)$ 独立，则：

$$
M_X(t) = \exp(\lambda_1(e^t - 1))
$$

$$
M_Y(t) = \exp(\lambda_2(e^t - 1))
$$

$$
M_{X+Y}(t) = \exp((\lambda_1 + \lambda_2)(e^t - 1))
$$

因此 $X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$。

---

**MGF的存在性**:

**注意**: MGF不总是存在。

**例子**: Cauchy分布

Cauchy分布的MGF不存在（积分发散），但特征函数存在：

$$
\phi_X(t) = e^{-|t|}
$$

这就是为什么在理论工作中，特征函数比MGF更常用。

---

**MGF与中心极限定理**:

MGF在证明中心极限定理时起关键作用（虽然通常使用特征函数）。

**思路**: 
1. 计算标准化和的MGF
2. 证明它收敛到标准正态分布的MGF
3. 由唯一性定理，得到分布收敛

---

### 3. 特征函数

**定义 3.1 (特征函数)**:

$$
\phi_X(t) = E[e^{itX}] = \int_{-\infty}^\infty e^{itx} f_X(x) \, dx
$$

**优势**: 总是存在（MGF可能不存在）

**性质**: 与MGF类似，但使用复数

---

## 🔧 在机器学习中的应用

### 1. 贝叶斯推断

**贝叶斯公式**:

$$
p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}
$$

其中：

- $p(\theta)$: 先验分布
- $p(\mathcal{D} | \theta)$: 似然函数
- $p(\theta | \mathcal{D})$: 后验分布

**应用**:

- 贝叶斯线性回归
- 高斯过程
- 贝叶斯神经网络

---

### 2. 最大似然估计

**定义**:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta p(\mathcal{D} | \theta)
$$

**对数似然**:

$$
\ell(\theta) = \log p(\mathcal{D} | \theta) = \sum_{i=1}^n \log p(x_i | \theta)
$$

**示例** (正态分布):

给定数据 $\{x_1, \ldots, x_n\}$，假设 $x_i \sim \mathcal{N}(\mu, \sigma^2)$：

$$
\hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n x_i
$$

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2
$$

---

### 3. 变分推断

**问题**: 计算后验分布 $p(\theta | \mathcal{D})$ 通常是困难的。

**变分推断**: 用简单分布 $q(\theta)$ 近似 $p(\theta | \mathcal{D})$。

**KL散度**:

$$
\text{KL}(q \| p) = \int q(\theta) \log \frac{q(\theta)}{p(\theta | \mathcal{D})} \, d\theta
$$

**ELBO** (Evidence Lower Bound):

$$
\mathcal{L}(q) = E_q[\log p(\mathcal{D}, \theta)] - E_q[\log q(\theta)]
$$

最大化ELBO等价于最小化KL散度。

---

### 4. 采样方法

**蒙特卡洛方法**:

$$
E[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i), \quad x_i \sim p(x)
$$

**马尔可夫链蒙特卡洛 (MCMC)**:

- Metropolis-Hastings算法
- Gibbs采样
- Hamiltonian Monte Carlo (HMC)

**应用**: 贝叶斯推断、生成模型

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import comb

# 1. 常见分布可视化
def plot_distributions():
    """可视化常见概率分布"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 伯努利分布
    ax = axes[0, 0]
    p = 0.7
    x = [0, 1]
    pmf = [1-p, p]
    ax.bar(x, pmf, width=0.3)
    ax.set_title(f'Bernoulli(p={p})')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X=x)')
    
    # 二项分布
    ax = axes[0, 1]
    n, p = 10, 0.5
    x = np.arange(0, n+1)
    pmf = stats.binom.pmf(x, n, p)
    ax.bar(x, pmf)
    ax.set_title(f'Binomial(n={n}, p={p})')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X=x)')
    
    # 泊松分布
    ax = axes[0, 2]
    lambda_ = 3
    x = np.arange(0, 15)
    pmf = stats.poisson.pmf(x, lambda_)
    ax.bar(x, pmf)
    ax.set_title(f'Poisson(λ={lambda_})')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X=x)')
    
    # 均匀分布
    ax = axes[1, 0]
    a, b = 0, 1
    x = np.linspace(-0.5, 1.5, 1000)
    pdf = stats.uniform.pdf(x, a, b-a)
    ax.plot(x, pdf)
    ax.fill_between(x, pdf, alpha=0.3)
    ax.set_title(f'Uniform(a={a}, b={b})')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    
    # 正态分布
    ax = axes[1, 1]
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 1000)
    pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf)
    ax.fill_between(x, pdf, alpha=0.3)
    ax.set_title(f'Normal(μ={mu}, σ²={sigma**2})')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    
    # 指数分布
    ax = axes[1, 2]
    lambda_ = 1
    x = np.linspace(0, 5, 1000)
    pdf = stats.expon.pdf(x, scale=1/lambda_)
    ax.plot(x, pdf)
    ax.fill_between(x, pdf, alpha=0.3)
    ax.set_title(f'Exponential(λ={lambda_})')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    
    plt.tight_layout()
    # plt.show()


# 2. 中心极限定理演示
def central_limit_theorem_demo():
    """演示中心极限定理"""
    np.random.seed(42)
    
    # 原始分布 (均匀分布)
    n_samples = 1000
    sample_sizes = [1, 5, 30, 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, n in enumerate(sample_sizes):
        # 生成样本均值
        sample_means = []
        for _ in range(n_samples):
            sample = np.random.uniform(0, 1, n)
            sample_means.append(np.mean(sample))
        
        # 绘制直方图
        ax = axes[idx]
        ax.hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        # 理论正态分布
        mu = 0.5  # 均匀分布的期望
        sigma = 1/np.sqrt(12*n)  # 样本均值的标准差
        x = np.linspace(0, 1, 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Theoretical Normal')
        
        ax.set_title(f'Sample Size n={n}')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle('Central Limit Theorem: Sample Means of Uniform(0,1)', fontsize=14)
    plt.tight_layout()
    # plt.show()


# 3. 最大似然估计
def maximum_likelihood_estimation():
    """最大似然估计示例"""
    np.random.seed(42)
    
    # 生成数据
    true_mu = 2.0
    true_sigma = 1.5
    n = 100
    data = np.random.normal(true_mu, true_sigma, n)
    
    # MLE估计
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)  # 使用n而不是n-1
    
    print("=== 最大似然估计 ===")
    print(f"真实参数: μ={true_mu}, σ={true_sigma}")
    print(f"MLE估计: μ̂={mu_mle:.4f}, σ̂={sigma_mle:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    # 数据直方图
    plt.hist(data, bins=20, density=True, alpha=0.7, edgecolor='black', label='Data')
    
    # 真实分布
    x = np.linspace(data.min(), data.max(), 100)
    pdf_true = stats.norm.pdf(x, true_mu, true_sigma)
    plt.plot(x, pdf_true, 'r-', linewidth=2, label=f'True: N({true_mu}, {true_sigma**2})')
    
    # MLE拟合的分布
    pdf_mle = stats.norm.pdf(x, mu_mle, sigma_mle)
    plt.plot(x, pdf_mle, 'b--', linewidth=2, label=f'MLE: N({mu_mle:.2f}, {sigma_mle**2:.2f})')
    
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Maximum Likelihood Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()


# 4. 贝叶斯推断示例
def bayesian_inference_demo():
    """贝叶斯推断示例 (硬币投掷)"""
    # 先验: Beta(α, β)
    alpha_prior = 2
    beta_prior = 2
    
    # 数据: 10次投掷，7次正面
    n_heads = 7
    n_tails = 3
    
    # 后验: Beta(α + n_heads, β + n_tails)
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + n_tails
    
    # 可视化
    p = np.linspace(0, 1, 100)
    
    prior = stats.beta.pdf(p, alpha_prior, beta_prior)
    likelihood = stats.binom.pmf(n_heads, n_heads + n_tails, p)
    posterior = stats.beta.pdf(p, alpha_post, beta_post)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, prior, 'b-', label=f'Prior: Beta({alpha_prior}, {beta_prior})', linewidth=2)
    plt.plot(p, likelihood / likelihood.max() * prior.max(), 'g--', 
             label=f'Likelihood (scaled)', linewidth=2)
    plt.plot(p, posterior, 'r-', label=f'Posterior: Beta({alpha_post}, {beta_post})', linewidth=2)
    
    plt.xlabel('p (probability of heads)')
    plt.ylabel('Density')
    plt.title('Bayesian Inference: Coin Flip')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    print("\n=== 贝叶斯推断 ===")
    print(f"先验均值: {alpha_prior/(alpha_prior+beta_prior):.4f}")
    print(f"后验均值: {alpha_post/(alpha_post+beta_post):.4f}")
    print(f"MLE估计: {n_heads/(n_heads+n_tails):.4f}")


# 5. 蒙特卡洛积分
def monte_carlo_integration():
    """蒙特卡洛积分示例"""
    np.random.seed(42)
    
    # 计算 E[X^2] where X ~ N(0, 1)
    # 理论值: 1
    
    sample_sizes = [10, 100, 1000, 10000]
    estimates = []
    
    for n in sample_sizes:
        samples = np.random.normal(0, 1, n)
        estimate = np.mean(samples**2)
        estimates.append(estimate)
        print(f"n={n:5d}: E[X²] ≈ {estimate:.6f}")
    
    print(f"\n理论值: E[X²] = 1.000000")
    
    # 可视化收敛
    plt.figure(figsize=(10, 6))
    plt.semilogx(sample_sizes, estimates, 'bo-', markersize=8, label='MC Estimate')
    plt.axhline(y=1, color='r', linestyle='--', linewidth=2, label='True Value')
    plt.xlabel('Sample Size')
    plt.ylabel('Estimate of E[X²]')
    plt.title('Monte Carlo Integration Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()


if __name__ == "__main__":
    print("=== 随机变量与分布示例 ===\n")
    
    print("1. 常见分布可视化")
    plot_distributions()
    
    print("\n2. 中心极限定理")
    central_limit_theorem_demo()
    
    print("\n3. 最大似然估计")
    maximum_likelihood_estimation()
    
    print("\n4. 贝叶斯推断")
    bayesian_inference_demo()
    
    print("\n5. 蒙特卡洛积分")
    monte_carlo_integration()
```

---

## 📚 练习题

### 练习1：分布计算

设 $X \sim \mathcal{N}(0, 1)$，计算：

1. $P(X \leq 1)$
2. $P(-1 \leq X \leq 1)$
3. $P(X^2 \leq 1)$

### 练习2：期望与方差

设 $X \sim \text{Uniform}(0, 1)$，计算：

1. $E[X^2]$
2. $\text{Var}(X)$
3. $E[e^X]$

### 练习3：变换

设 $X \sim \text{Exp}(\lambda)$，$Y = \sqrt{X}$，求 $Y$ 的密度函数。

### 练习4：最大似然估计

给定数据 $\{x_1, \ldots, x_n\}$ 来自 $\text{Exp}(\lambda)$，求 $\lambda$ 的MLE估计。

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **MIT** | 6.041 - Probabilistic Systems Analysis |
| **MIT** | 18.650 - Statistics for Applications |
| **Stanford** | CS109 - Probability for Computer Scientists |
| **Stanford** | STATS214 - Machine Learning Theory |
| **UC Berkeley** | STAT134 - Concepts of Probability |
| **CMU** | 36-705 - Intermediate Statistics |

---

## 📖 参考文献

1. **Casella & Berger (2002)**. *Statistical Inference*. Duxbury Press.

2. **Wasserman, L. (2004)**. *All of Statistics*. Springer.

3. **Bishop, C. (2006)**. *Pattern Recognition and Machine Learning*. Springer.

4. **Murphy, K. (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press.

5. **Goodfellow et al. (2016)**. *Deep Learning*. MIT Press. (Chapter 3: Probability)

---

*最后更新：2025年10月*-
