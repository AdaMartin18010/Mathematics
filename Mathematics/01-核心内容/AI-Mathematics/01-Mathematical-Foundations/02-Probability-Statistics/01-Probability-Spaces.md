# 概率空间与测度论基础 (Probability Spaces and Measure Theory)

> **AI概率模型的数学基础**

---

## 📋 目录

- [概率空间与测度论基础 (Probability Spaces and Measure Theory)](#概率空间与测度论基础-probability-spaces-and-measure-theory)
  - [📋 目录](#-目录)
  - [🎯 为什么需要测度论?](#-为什么需要测度论)
  - [📐 测度论基础](#-测度论基础)
    - [1. σ-代数](#1-σ-代数)
    - [2. 测度](#2-测度)
    - [3. 可测函数](#3-可测函数)
  - [🎲 概率空间](#-概率空间)
    - [1. Kolmogorov公理](#1-kolmogorov公理)
    - [2. 随机变量](#2-随机变量)
    - [3. 期望与积分](#3-期望与积分)
  - [🔍 重要定理](#-重要定理)
    - [1. 单调收敛定理](#1-单调收敛定理)
    - [2. Fatou引理](#2-fatou引理)
    - [3. 控制收敛定理](#3-控制收敛定理)
  - [🤖 在AI中的应用](#-在ai中的应用)
    - [1. 连续型随机变量](#1-连续型随机变量)
    - [2. 条件期望](#2-条件期望)
    - [3. 随机过程](#3-随机过程)
  - [💻 Python实现](#-python实现)
  - [📚 相关资源](#-相关资源)
  - [💡 练习题](#-练习题)

---

## 🎯 为什么需要测度论?

**朴素概率**的局限:

- 只能处理有限或可数样本空间
- 无法严格定义连续型随机变量
- 难以处理无穷维空间(如函数空间)

**测度论概率**的优势:

- ✅ 统一离散和连续情形
- ✅ 严格的数学基础
- ✅ 支持高维和无穷维
- ✅ 现代机器学习的语言

**AI应用**:

- 深度生成模型(VAE, Diffusion Models)
- 高斯过程
- 随机优化
- 强化学习的数学基础

---

## 📐 测度论基础

### 1. σ-代数

**定义** (σ-Algebra):

设 $\Omega$ 是样本空间, $\mathcal{F}$ 是 $\Omega$ 的子集族。$\mathcal{F}$ 是**σ-代数**,如果:

1. $\Omega \in \mathcal{F}$
2. 若 $A \in \mathcal{F}$, 则 $A^c \in \mathcal{F}$ (对补运算封闭)
3. 若 $A_1, A_2, \ldots \in \mathcal{F}$, 则 $\bigcup_{i=1}^\infty A_i \in \mathcal{F}$ (对可数并封闭)

**示例**:

1. **平凡σ-代数**: $\mathcal{F} = \{\emptyset, \Omega\}$
2. **离散σ-代数**: $\mathcal{F} = 2^\Omega$ (所有子集)
3. **Borel σ-代数**: $\mathcal{B}(\mathbb{R})$ (由所有开集生成)

---

**Borel σ-代数** (重要!):

定义: $\mathbb{R}$ 上的**最小**σ-代数包含所有开区间:

$$
\mathcal{B}(\mathbb{R}) = \sigma(\{(a,b) : a < b\})
$$

包含:

- 所有开集、闭集
- 所有单点集
- 可数并、可数交

---

**Borel σ-代数的详细构造**:

**定义**: Borel σ-代数 $\mathcal{B}(\mathbb{R})$ 是包含所有开区间的**最小** σ-代数。

**生成过程**:

**第一步：从开区间开始**:

令 $\mathcal{I} = \{(a,b) : a, b \in \mathbb{R}, a < b\}$ 是所有开区间的集合。

$\mathcal{B}(\mathbb{R})$ 是包含 $\mathcal{I}$ 的最小 σ-代数，记为：

$$
\mathcal{B}(\mathbb{R}) = \sigma(\mathcal{I})
$$

**第二步：σ-代数的生成**:

$\sigma(\mathcal{I})$ 的构造过程：

1. 从 $\mathcal{I}$ 开始
2. 添加所有可数并：$\bigcup_{n=1}^\infty A_n$（其中 $A_n \in$ 当前集合）
3. 添加所有补集：$A^c$（其中 $A \in$ 当前集合）
4. 重复步骤2-3直到不再增加新元素

这个过程可能需要**超限归纳**（transfinite induction）才能完成。

**第三步：Borel集包含的集合类型**:

**命题**: Borel σ-代数包含以下集合：

1. **所有开集**
2. **所有闭集**
3. **所有单点集**
4. **所有可数集**
5. **所有半开区间** $[a,b)$, $(a,b]$, $[a,b]$

**证明**:

**(1) 所有开集**:

$\mathbb{R}$ 中的任何开集 $U$ 都可以表示为可数个开区间的并：

$$
U = \bigcup_{n=1}^\infty (a_n, b_n)
$$

（这是因为 $\mathbb{R}$ 有可数基，例如有理端点的开区间）

由于 $(a_n, b_n) \in \mathcal{B}(\mathbb{R})$，且 σ-代数对可数并封闭，因此 $U \in \mathcal{B}(\mathbb{R})$。

**(2) 所有闭集**:

闭集是开集的补集。若 $F$ 是闭集，则 $F^c$ 是开集。

由 (1)，$F^c \in \mathcal{B}(\mathbb{R})$。

由 σ-代数对补集封闭，$F = (F^c)^c \in \mathcal{B}(\mathbb{R})$。

**(3) 所有单点集**:

对于 $x \in \mathbb{R}$，单点集 $\{x\}$ 是闭集（因为 $\{x\}^c = (-\infty, x) \cup (x, \infty)$ 是开集）。

由 (2)，$\{x\} \in \mathcal{B}(\mathbb{R})$。

**(4) 所有可数集**:

若 $A = \{x_1, x_2, x_3, \ldots\}$ 是可数集，则：

$$
A = \bigcup_{n=1}^\infty \{x_n\}
$$

由 (3) 和 σ-代数对可数并封闭，$A \in \mathcal{B}(\mathbb{R})$。

**(5) 所有半开区间**:

对于 $[a, b)$：

$$
[a, b) = \bigcap_{n=1}^\infty \left(a - \frac{1}{n}, b\right)
$$

每个 $(a - 1/n, b) \in \mathcal{B}(\mathbb{R})$（开区间），因此 $[a, b) \in \mathcal{B}(\mathbb{R})$（可数交）。

类似地可以证明 $(a, b]$ 和 $[a, b]$ 也在 $\mathcal{B}(\mathbb{R})$ 中。

$\square$

---

**具体例子**:

**例1**: 有理数集 $\mathbb{Q}$

$\mathbb{Q}$ 是可数集，因此 $\mathbb{Q} \in \mathcal{B}(\mathbb{R})$。

**例2**: 无理数集 $\mathbb{R} \setminus \mathbb{Q}$

由于 $\mathbb{Q} \in \mathcal{B}(\mathbb{R})$ 且 σ-代数对补集封闭，$\mathbb{R} \setminus \mathbb{Q} \in \mathcal{B}(\mathbb{R})$。

**例3**: Cantor集

Cantor集是通过迭代去除中间三分之一得到的闭集，因此 $C \in \mathcal{B}(\mathbb{R})$。

**例4**: 不在Borel σ-代数中的集合

存在 $\mathbb{R}$ 的子集不在 $\mathcal{B}(\mathbb{R})$ 中（需要选择公理）。这些集合称为**非Borel集**。

例如，在某些模型中，存在 Lebesgue 可测但非 Borel 可测的集合。

---

**Borel σ-代数的基数**:

**定理**: $|\mathcal{B}(\mathbb{R})| = \mathfrak{c} = 2^{\aleph_0}$（连续统的基数）

这意味着 Borel 集的个数与 $\mathbb{R}$ 的个数相同，但小于 $\mathbb{R}$ 的所有子集的个数 $2^{\mathfrak{c}}$。

---

**为什么使用Borel σ-代数？**

1. **自然性**: 包含所有"常见"的集合（开集、闭集、可数集等）
2. **可测性**: Borel 可测函数是连续函数的自然推广
3. **概率论**: 随机变量通常取值在 Borel 集上
4. **分析学**: 与拓扑结构兼容

---

**推广到其他空间**:

对于任何拓扑空间 $(X, \tau)$，Borel σ-代数定义为：

$$
\mathcal{B}(X) = \sigma(\tau)
$$

即包含所有开集的最小 σ-代数。

**例子**:

- $\mathcal{B}(\mathbb{R}^n)$: $n$ 维欧氏空间的 Borel σ-代数
- $\mathcal{B}([0,1])$: 单位区间的 Borel σ-代数

---

### 2. 测度

**定义** (Measure):

函数 $\mu: \mathcal{F} \to [0, \infty]$ 是**测度**,如果:

1. $\mu(\emptyset) = 0$
2. **可数可加性**: 对不相交的 $A_1, A_2, \ldots \in \mathcal{F}$:

$$
\mu\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty \mu(A_i)
$$

**示例**:

1. **计数测度**: $\mu(A) = |A|$ (元素个数)
2. **Lebesgue测度**: $\lambda([a,b]) = b - a$ (长度)
3. **Dirac测度**: $\delta_x(A) = \mathbb{1}_A(x)$

---

**概率测度**:

测度 $\mathbb{P}$ 是**概率测度**,如果:

$$
\mathbb{P}(\Omega) = 1
$$

---

### 3. 可测函数

**定义** (Measurable Function):

函数 $f: (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B})$ 是**可测的**,如果:

$$
\forall B \in \mathcal{B}, \quad f^{-1}(B) \in \mathcal{F}
$$

即: 开集的原像仍是可测集。

**等价条件**:

$$
\{f \leq a\} \in \mathcal{F}, \quad \forall a \in \mathbb{R}
$$

---

**性质**:

1. 连续函数是可测的
2. 可测函数的和、积、极限仍可测
3. 随机变量就是可测函数!

---

**可测函数性质的完整证明**:

**性质1：连续函数是Borel可测的**:

**定理**: 若 $f: \mathbb{R} \to \mathbb{R}$ 连续，则 $f$ 是 Borel 可测的。

**证明**:

需要证明：对任意 $a \in \mathbb{R}$，$\{x : f(x) \leq a\} \in \mathcal{B}(\mathbb{R})$。

令 $A = \{x : f(x) \leq a\} = f^{-1}((-\infty, a])$。

由于 $f$ 连续，$(-\infty, a]$ 的原像是闭集（连续函数的原像保持闭集）。

因此 $A$ 是闭集，而所有闭集都在 Borel σ-代数中。

所以 $A \in \mathcal{B}(\mathbb{R})$，即 $f$ 是 Borel 可测的。 $\square$

---

**性质2：可测函数的和是可测的**:

**定理**: 若 $f, g: (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$ 可测，则 $f + g$ 可测。

**证明**:

需要证明：对任意 $a \in \mathbb{R}$，$\{f + g \leq a\} \in \mathcal{F}$。

**关键观察**:

$$
\{f + g \leq a\} = \{f \leq a - g\}
$$

但这个表达式不能直接使用，因为右边不是标准形式。

**正确方法**：使用有理数的稠密性。

$$
\{f + g \leq a\} = \bigcup_{r \in \mathbb{Q}} \left(\{f \leq r\} \cap \{g \leq a - r\}\right)
$$

**证明这个等式**:

**"⊆"**: 若 $\omega \in \{f + g \leq a\}$，则 $f(\omega) + g(\omega) \leq a$。

选择有理数 $r$ 使得 $f(\omega) < r < a - g(\omega)$（这样的 $r$ 存在，因为有理数稠密）。

则 $f(\omega) \leq r$ 且 $g(\omega) \leq a - r$，所以 $\omega \in \{f \leq r\} \cap \{g \leq a - r\}$。

**"⊇"**: 若 $\omega \in \{f \leq r\} \cap \{g \leq a - r\}$ 对某个 $r \in \mathbb{Q}$，

则 $f(\omega) \leq r$ 且 $g(\omega) \leq a - r$，

因此 $f(\omega) + g(\omega) \leq r + (a - r) = a$，即 $\omega \in \{f + g \leq a\}$。

**结论**:

由于 $f, g$ 可测，$\{f \leq r\}, \{g \leq a - r\} \in \mathcal{F}$。

σ-代数对交封闭，所以 $\{f \leq r\} \cap \{g \leq a - r\} \in \mathcal{F}$。

σ-代数对可数并封闭（$\mathbb{Q}$ 可数），所以：

$$
\{f + g \leq a\} = \bigcup_{r \in \mathbb{Q}} \left(\{f \leq r\} \cap \{g \leq a - r\}\right) \in \mathcal{F}
$$

因此 $f + g$ 可测。 $\square$

---

**性质3：可测函数的积是可测的**:

**定理**: 若 $f, g: (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$ 可测，则 $fg$ 可测。

**证明**:

**第一步：证明 $f^2$ 可测**

需要证明：对任意 $a \in \mathbb{R}$，$\{f^2 \leq a\} \in \mathcal{F}$。

**情况1**: $a < 0$

$\{f^2 \leq a\} = \emptyset \in \mathcal{F}$（因为 $f^2 \geq 0$）

**情况2**: $a \geq 0$

$$
\{f^2 \leq a\} = \{-\sqrt{a} \leq f \leq \sqrt{a}\} = \{f \leq \sqrt{a}\} \cap \{f \geq -\sqrt{a}\}
$$

$$
= \{f \leq \sqrt{a}\} \cap \{-f \leq \sqrt{a}\}
$$

由于 $f$ 可测，$\{f \leq \sqrt{a}\} \in \mathcal{F}$。

由于 $-f$ 可测（下面会证明），$\{-f \leq \sqrt{a}\} \in \mathcal{F}$。

因此 $\{f^2 \leq a\} \in \mathcal{F}$。

**第二步：证明 $-f$ 可测**

$$
\{-f \leq a\} = \{f \geq -a\} = \{f \leq -a\}^c \in \mathcal{F}
$$

（σ-代数对补集封闭）

**第三步：证明 $fg$ 可测**

使用恒等式：

$$
fg = \frac{(f+g)^2 - (f-g)^2}{4}
$$

由性质2，$f + g$ 和 $f - g$ 可测。

由第一步，$(f+g)^2$ 和 $(f-g)^2$ 可测。

由性质2，$(f+g)^2 - (f-g)^2$ 可测。

由标量乘法（$cf$ 可测当 $f$ 可测），$\frac{1}{4}[(f+g)^2 - (f-g)^2]$ 可测。

因此 $fg$ 可测。 $\square$

---

**性质4：可测函数的极限是可测的**:

**定理**: 若 $f_n: (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$ 可测（$n = 1, 2, 3, \ldots$），则：

1. $\sup_n f_n$ 可测
2. $\inf_n f_n$ 可测
3. $\limsup_n f_n$ 可测
4. $\liminf_n f_n$ 可测
5. 若 $\lim_n f_n$ 存在，则它可测

**证明**:

**(1) $\sup_n f_n$ 可测**:

令 $g = \sup_n f_n$。需要证明：对任意 $a \in \mathbb{R}$，$\{g > a\} \in \mathcal{F}$。

**关键观察**:

$$
\{g > a\} = \{\sup_n f_n > a\} = \bigcup_{n=1}^\infty \{f_n > a\}
$$

**证明等式**:

- $\omega \in \{g > a\}$ ⟺ $\sup_n f_n(\omega) > a$ ⟺ 存在 $n$ 使得 $f_n(\omega) > a$ ⟺ $\omega \in \bigcup_{n=1}^\infty \{f_n > a\}$

由于 $f_n$ 可测，$\{f_n > a\} = \{f_n \leq a\}^c \in \mathcal{F}$。

σ-代数对可数并封闭，所以 $\{g > a\} \in \mathcal{F}$。

因此 $g$ 可测。 $\square$

**(2) $\inf_n f_n$ 可测**:

使用 $\inf_n f_n = -\sup_n (-f_n)$。

由第二步，$-f_n$ 可测。

由 (1)，$\sup_n (-f_n)$ 可测。

因此 $\inf_n f_n$ 可测。 $\square$

**(3) $\limsup_n f_n$ 可测**:

回顾定义：

$$
\limsup_n f_n = \inf_m \sup_{n \geq m} f_n
$$

令 $g_m = \sup_{n \geq m} f_n$。

由 (1)，$g_m$ 可测（对每个 $m$）。

由 (2)，$\inf_m g_m$ 可测。

因此 $\limsup_n f_n$ 可测。 $\square$

**(4) $\liminf_n f_n$ 可测**:

类似地，使用：

$$
\liminf_n f_n = \sup_m \inf_{n \geq m} f_n
$$

**(5) 若 $\lim_n f_n$ 存在，则它可测**:

若 $\lim_n f_n$ 存在，则：

$$
\lim_n f_n = \limsup_n f_n = \liminf_n f_n
$$

由 (3) 和 (4)，$\lim_n f_n$ 可测。 $\square$

---

**应用示例**:

**例1**: 若 $f, g$ 可测，则 $\max(f, g)$ 和 $\min(f, g)$ 可测。

证明：$\max(f, g) = \sup\{f, g\}$，$\min(f, g) = \inf\{f, g\}$。

**例2**: 若 $f$ 可测，则 $|f|$ 可测。

证明：$|f| = \max(f, -f)$。

**例3**: 若 $f$ 可测，则 $f^+ = \max(f, 0)$ 和 $f^- = \max(-f, 0)$ 可测。

这是 $f = f^+ - f^-$ 分解的基础。

---

## 🎲 概率空间

### 1. Kolmogorov公理

**定义** (Probability Space):

概率空间是三元组 $(\Omega, \mathcal{F}, \mathbb{P})$:

- $\Omega$: 样本空间
- $\mathcal{F}$: σ-代数 (事件空间)
- $\mathbb{P}$: 概率测度

**公理**:

1. $\mathbb{P}(A) \geq 0$ (非负性)
2. $\mathbb{P}(\Omega) = 1$ (归一化)
3. 对不相交的 $A_1, A_2, \ldots$:

$$
\mathbb{P}\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty \mathbb{P}(A_i)
$$

---

**推论**:

$$
\begin{align}
\mathbb{P}(A^c) &= 1 - \mathbb{P}(A) \\
\mathbb{P}(A \cup B) &= \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B) \\
A \subseteq B &\Rightarrow \mathbb{P}(A) \leq \mathbb{P}(B)
\end{align}
$$

---

### 2. 随机变量

**定义** (Random Variable):

随机变量是可测函数:

$$
X: (\Omega, \mathcal{F}, \mathbb{P}) \to (\mathbb{R}, \mathcal{B})
$$

**直观理解**: 将随机实验的结果映射到实数。

---

**分布函数** (CDF):

$$
F_X(x) = \mathbb{P}(X \leq x) = \mathbb{P}(\{\omega : X(\omega) \leq x\})
$$

**性质**:

1. 单调递增
2. 右连续
3. $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$

---

**概率密度函数** (PDF):

如果存在 $f_X$ 使得:

$$
F_X(x) = \int_{-\infty}^x f_X(t) \, dt
$$

则称 $X$ 是**连续型**随机变量, $f_X$ 是其密度函数。

---

### 3. 期望与积分

**Lebesgue积分**:

对非负可测函数 $f$:

$$
\int_\Omega f \, d\mu = \sup \left\{ \int_\Omega s \, d\mu : 0 \leq s \leq f, \, s \text{ simple} \right\}
$$

其中**简单函数**:

$$
s = \sum_{i=1}^n a_i \mathbb{1}_{A_i}
$$

---

**期望**:

随机变量 $X$ 的期望:

$$
\mathbb{E}[X] = \int_\Omega X \, d\mathbb{P}
$$

**离散情形**:

$$
\mathbb{E}[X] = \sum_{x} x \cdot \mathbb{P}(X = x)
$$

**连续情形**:

$$
\mathbb{E}[X] = \int_{-\infty}^\infty x \, f_X(x) \, dx
$$

---

**性质**:

1. **线性性**: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
2. **单调性**: $X \leq Y \Rightarrow \mathbb{E}[X] \leq \mathbb{E}[Y]$
3. **独立性**: $X, Y$ 独立 $\Rightarrow \mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$

---

**性质的完整证明**:

**证明1：线性性**:

需要证明：$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$

**第一步：标量乘法**:

先证明 $\mathbb{E}[aX] = a\mathbb{E}[X]$。

对于简单函数 $X = \sum_{i=1}^n x_i \mathbb{1}_{A_i}$：

$$
\mathbb{E}[aX] = \mathbb{E}\left[a \sum_{i=1}^n x_i \mathbb{1}_{A_i}\right] = \mathbb{E}\left[\sum_{i=1}^n (ax_i) \mathbb{1}_{A_i}\right]
$$

$$
= \sum_{i=1}^n (ax_i) P(A_i) = a \sum_{i=1}^n x_i P(A_i) = a\mathbb{E}[X]
$$

对于一般的非负可测函数，用简单函数逼近并取极限（使用单调收敛定理）。

对于一般可测函数，分解为 $X = X^+ - X^-$，分别应用上述结果。

**第二步：加法**:

先证明 $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$。

对于简单函数 $X = \sum_{i=1}^m x_i \mathbb{1}_{A_i}$，$Y = \sum_{j=1}^n y_j \mathbb{1}_{B_j}$：

$$
X + Y = \sum_{i=1}^m \sum_{j=1}^n (x_i + y_j) \mathbb{1}_{A_i \cap B_j}
$$

因此：

$$
\mathbb{E}[X + Y] = \sum_{i=1}^m \sum_{j=1}^n (x_i + y_j) P(A_i \cap B_j)
$$

$$
= \sum_{i=1}^m \sum_{j=1}^n x_i P(A_i \cap B_j) + \sum_{i=1}^m \sum_{j=1}^n y_j P(A_i \cap B_j)
$$

$$
= \sum_{i=1}^m x_i \sum_{j=1}^n P(A_i \cap B_j) + \sum_{j=1}^n y_j \sum_{i=1}^m P(A_i \cap B_j)
$$

$$
= \sum_{i=1}^m x_i P(A_i) + \sum_{j=1}^n y_j P(B_j) = \mathbb{E}[X] + \mathbb{E}[Y]
$$

（这里使用了 $\sum_{j=1}^n P(A_i \cap B_j) = P(A_i)$，因为 $\{B_j\}$ 构成分割）

对于一般可测函数，同样用逼近和极限论证。

**第三步：结合**:

$$
\mathbb{E}[aX + bY] = \mathbb{E}[aX] + \mathbb{E}[bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]
$$

$\square$

---

**证明2：单调性**:

需要证明：若 $X \leq Y$ a.e.，则 $\mathbb{E}[X] \leq \mathbb{E}[Y]$

**证明**:

令 $Z = Y - X \geq 0$ a.e.

则：

$$
\mathbb{E}[Y] = \mathbb{E}[X + Z] = \mathbb{E}[X] + \mathbb{E}[Z]
$$

由于 $Z \geq 0$ a.e.，我们有 $\mathbb{E}[Z] \geq 0$（这是期望定义的直接结果）。

因此：

$$
\mathbb{E}[Y] = \mathbb{E}[X] + \mathbb{E}[Z] \geq \mathbb{E}[X]
$$

$\square$

**注意**: 这个证明依赖于非负函数期望的非负性，这是期望定义的基本性质。

---

**证明3：独立性**:

需要证明：若 $X, Y$ 独立，则 $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$

**第一步：指示函数的情况**:

设 $X = \mathbb{1}_A$，$Y = \mathbb{1}_B$，其中 $A, B$ 独立。

则：

$$
\mathbb{E}[XY] = \mathbb{E}[\mathbb{1}_A \mathbb{1}_B] = \mathbb{E}[\mathbb{1}_{A \cap B}] = P(A \cap B)
$$

由独立性：

$$
P(A \cap B) = P(A)P(B) = \mathbb{E}[\mathbb{1}_A]\mathbb{E}[\mathbb{1}_B] = \mathbb{E}[X]\mathbb{E}[Y]
$$

**第二步：简单函数的情况**:

设 $X = \sum_{i=1}^m x_i \mathbb{1}_{A_i}$，$Y = \sum_{j=1}^n y_j \mathbb{1}_{B_j}$，其中 $\{A_i\}$ 和 $\{B_j\}$ 分别是分割。

由 $X, Y$ 独立，事件 $A_i$ 和 $B_j$ 独立（对所有 $i, j$）。

则：

$$
XY = \sum_{i=1}^m \sum_{j=1}^n x_i y_j \mathbb{1}_{A_i} \mathbb{1}_{B_j} = \sum_{i=1}^m \sum_{j=1}^n x_i y_j \mathbb{1}_{A_i \cap B_j}
$$

因此：

$$
\mathbb{E}[XY] = \sum_{i=1}^m \sum_{j=1}^n x_i y_j P(A_i \cap B_j)
$$

由独立性 $P(A_i \cap B_j) = P(A_i)P(B_j)$：

$$
\mathbb{E}[XY] = \sum_{i=1}^m \sum_{j=1}^n x_i y_j P(A_i)P(B_j)
$$

$$
= \left(\sum_{i=1}^m x_i P(A_i)\right) \left(\sum_{j=1}^n y_j P(B_j)\right) = \mathbb{E}[X]\mathbb{E}[Y]
$$

**第三步：非负可测函数的情况**:

设 $X, Y \geq 0$ 可测且独立。

存在简单函数序列 $X_n \uparrow X$，$Y_n \uparrow Y$。

由第二步：

$$
\mathbb{E}[X_n Y_n] = \mathbb{E}[X_n]\mathbb{E}[Y_n]
$$

由于 $X_n Y_n \uparrow XY$（单调递增），应用单调收敛定理：

$$
\mathbb{E}[XY] = \lim_{n \to \infty} \mathbb{E}[X_n Y_n] = \lim_{n \to \infty} \mathbb{E}[X_n]\mathbb{E}[Y_n]
$$

再次应用单调收敛定理到 $X_n$ 和 $Y_n$：

$$
= \left(\lim_{n \to \infty} \mathbb{E}[X_n]\right) \left(\lim_{n \to \infty} \mathbb{E}[Y_n]\right) = \mathbb{E}[X]\mathbb{E}[Y]
$$

**第四步：一般可测函数**:

对于一般的可积随机变量 $X, Y$，分解为：

$$
X = X^+ - X^-, \quad Y = Y^+ - Y^-
$$

其中 $X^+, X^-, Y^+, Y^-$ 都是非负的。

由独立性的保持，$X^+, X^-, Y^+, Y^-$ 之间也独立。

展开：

$$
XY = (X^+ - X^-)(Y^+ - Y^-) = X^+Y^+ - X^+Y^- - X^-Y^+ + X^-Y^-
$$

应用线性性和第三步的结果：

$$
\mathbb{E}[XY] = \mathbb{E}[X^+Y^+] - \mathbb{E}[X^+Y^-] - \mathbb{E}[X^-Y^+] + \mathbb{E}[X^-Y^-]
$$

$$
= \mathbb{E}[X^+]\mathbb{E}[Y^+] - \mathbb{E}[X^+]\mathbb{E}[Y^-] - \mathbb{E}[X^-]\mathbb{E}[Y^+] + \mathbb{E}[X^-]\mathbb{E}[Y^-]
$$

$$
= (\mathbb{E}[X^+] - \mathbb{E}[X^-])(\mathbb{E}[Y^+] - \mathbb{E}[Y^-]) = \mathbb{E}[X]\mathbb{E}[Y]
$$

$\square$

---

**性质的应用**:

1. **线性性**:
   - 计算线性组合的期望
   - 深度学习中的梯度计算
   - 统计推断中的估计量性质

2. **单调性**:
   - 建立不等式
   - 证明收敛性
   - 风险分析

3. **独立性**:
   - 计算独立随机变量和的方差
   - 协方差的性质
   - 多元统计分析

---

## 🔍 重要定理

### 1. 单调收敛定理

**定理** (Monotone Convergence Theorem, MCT):

设 $0 \leq f_1 \leq f_2 \leq \cdots$ 且 $f_n \to f$, 则:

$$
\lim_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu
$$

**意义**: 单调递增序列的积分可以交换极限和积分顺序。

**应用**: 证明Fubini定理、期望的计算

---

**定理的完整证明**:

**证明**:

**第一步：建立不等式的一个方向**:

由于 $f_1 \leq f_2 \leq \cdots$，我们有：

$$
\int f_1 \, d\mu \leq \int f_2 \, d\mu \leq \cdots
$$

因此序列 $\left\{\int f_n \, d\mu\right\}$ 是单调递增的。

令 $A = \lim_{n \to \infty} \int f_n \, d\mu$（可能是 $+\infty$）。

由于 $f_n \leq f$ 对所有 $n$，我们有：

$$
\int f_n \, d\mu \leq \int f \, d\mu
$$

取极限：

$$
A = \lim_{n \to \infty} \int f_n \, d\mu \leq \int f \, d\mu
$$

这给出了不等式的一个方向。

**第二步：证明另一个方向**:

现在需要证明：

$$
\int f \, d\mu \leq \lim_{n \to \infty} \int f_n \, d\mu = A
$$

取任意简单函数 $s$ 使得 $0 \leq s \leq f$。我们将证明：

$$
\int s \, d\mu \leq A
$$

由于 $\int f \, d\mu = \sup \left\{\int s \, d\mu : 0 \leq s \leq f, s \text{ simple}\right\}$，这将完成证明。

**第三步：对简单函数的处理**:

设 $s = \sum_{i=1}^m a_i \mathbb{1}_{A_i}$，其中 $A_i$ 互不相交，$\bigcup_{i=1}^m A_i = \Omega$。

对于 $0 < \alpha < 1$，定义：

$$
E_n = \{x : f_n(x) \geq \alpha s(x)\}
$$

**关键观察**：由于 $f_n \uparrow f$ 且 $s \leq f$，我们有：

$$
E_1 \subseteq E_2 \subseteq \cdots, \quad \bigcup_{n=1}^\infty E_n = \Omega
$$

（因为对任意 $x$，$f_n(x) \to f(x) \geq s(x)$，所以最终 $f_n(x) \geq \alpha s(x)$）

**第四步：估计积分**:

在 $E_n$ 上，$f_n \geq \alpha s$，因此：

$$
\int f_n \, d\mu \geq \int_{E_n} f_n \, d\mu \geq \int_{E_n} \alpha s \, d\mu = \alpha \int_{E_n} s \, d\mu
$$

对于简单函数 $s = \sum_{i=1}^m a_i \mathbb{1}_{A_i}$：

$$
\int_{E_n} s \, d\mu = \sum_{i=1}^m a_i \mu(A_i \cap E_n)
$$

**第五步：使用测度的连续性**:

由于 $E_n \uparrow \Omega$，测度的连续性给出：

$$
\lim_{n \to \infty} \mu(A_i \cap E_n) = \mu(A_i)
$$

因此：

$$
\lim_{n \to \infty} \int_{E_n} s \, d\mu = \sum_{i=1}^m a_i \mu(A_i) = \int s \, d\mu
$$

**第六步：取极限**:

从第四步的不等式：

$$
\int f_n \, d\mu \geq \alpha \int_{E_n} s \, d\mu
$$

取 $n \to \infty$：

$$
A = \lim_{n \to \infty} \int f_n \, d\mu \geq \alpha \int s \, d\mu
$$

由于这对所有 $0 < \alpha < 1$ 成立，令 $\alpha \to 1$：

$$
A \geq \int s \, d\mu
$$

**第七步：结论**:

由于这对所有简单函数 $0 \leq s \leq f$ 成立，取上确界：

$$
A \geq \sup_{0 \leq s \leq f} \int s \, d\mu = \int f \, d\mu
$$

结合第一步的结果 $A \leq \int f \, d\mu$，我们得到：

$$
A = \int f \, d\mu
$$

即：

$$
\lim_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu
$$

这就完成了单调收敛定理的证明。 $\square$

---

**证明的关键要点**：

1. **单调性**：利用 $f_n$ 的单调性建立积分序列的单调性
2. **简单函数逼近**：通过简单函数逼近 $f$
3. **测度的连续性**：$E_n \uparrow \Omega \Rightarrow \mu(E_n) \to \mu(\Omega)$
4. **巧妙的集合构造**：$E_n = \{f_n \geq \alpha s\}$ 使得可以在 $E_n$ 上估计积分

---

### 2. Fatou引理

**定理** (Fatou's Lemma):

设 $f_n \geq 0$, 则:

$$
\int \liminf_{n \to \infty} f_n \, d\mu \leq \liminf_{n \to \infty} \int f_n \, d\mu
$$

**直观**: 积分的下极限不超过下极限的积分。

---

**定理的完整证明**:

**证明**:

**第一步：回顾下极限的定义**:

对于序列 $\{a_n\}$，下极限定义为：

$$
\liminf_{n \to \infty} a_n = \lim_{n \to \infty} \inf_{k \geq n} a_k = \sup_{n} \inf_{k \geq n} a_k
$$

对于函数序列 $\{f_n\}$，逐点下极限为：

$$
g(x) = \liminf_{n \to \infty} f_n(x) = \lim_{n \to \infty} \inf_{k \geq n} f_k(x)
$$

**第二步：定义辅助函数序列**:

对于每个 $n$，定义：

$$
g_n(x) = \inf_{k \geq n} f_k(x)
$$

**关键性质**：

1. $g_n(x) \leq f_k(x)$ 对所有 $k \geq n$
2. $g_1(x) \leq g_2(x) \leq g_3(x) \leq \cdots$ （单调递增）
3. $\lim_{n \to \infty} g_n(x) = \liminf_{n \to \infty} f_n(x) = g(x)$

**第三步：应用单调收敛定理**:

由于 $g_n \geq 0$ 且 $g_n \uparrow g$，根据单调收敛定理：

$$
\int g \, d\mu = \int \liminf_{n \to \infty} f_n \, d\mu = \lim_{n \to \infty} \int g_n \, d\mu
$$

**第四步：建立不等式**:

对于每个 $n$，由于 $g_n(x) = \inf_{k \geq n} f_k(x) \leq f_k(x)$ 对所有 $k \geq n$，我们有：

$$
\int g_n \, d\mu \leq \int f_k \, d\mu, \quad \forall k \geq n
$$

因此：

$$
\int g_n \, d\mu \leq \inf_{k \geq n} \int f_k \, d\mu
$$

**第五步：取极限**:

对上式两边取 $n \to \infty$ 的极限：

$$
\lim_{n \to \infty} \int g_n \, d\mu \leq \lim_{n \to \infty} \inf_{k \geq n} \int f_k \, d\mu = \liminf_{n \to \infty} \int f_n \, d\mu
$$

**第六步：结论**:

结合第三步的结果：

$$
\int \liminf_{n \to \infty} f_n \, d\mu = \lim_{n \to \infty} \int g_n \, d\mu \leq \liminf_{n \to \infty} \int f_n \, d\mu
$$

这就完成了Fatou引理的证明。 $\square$

---

**证明的关键要点**：

1. **辅助函数构造**：$g_n = \inf_{k \geq n} f_k$ 是单调递增的
2. **单调收敛定理**：用于处理 $g_n \uparrow g$
3. **下极限的性质**：$\inf_{k \geq n} a_k$ 单调递增

**重要注意**：

- 不等号一般是严格的（可能取不到等号）
- 非负性条件 $f_n \geq 0$ 是必需的

**反例**（如果去掉非负性）：

令 $f_n = -\mathbb{1}_{[n, \infty)}$ 在 $\mathbb{R}$ 上（Lebesgue测度）。

则 $\liminf f_n = 0$，但 $\int f_n = -\infty$ 对所有 $n$。

---

### 3. 控制收敛定理

**定理** (Dominated Convergence Theorem, DCT):

设 $f_n \to f$ a.e., 且存在可积函数 $g$ 使得:

$$
|f_n| \leq g, \quad \forall n
$$

则:

$$
\lim_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu
$$

**意义**: 在控制函数存在时,可以交换极限和积分。

**应用**:

- 求导与积分交换顺序
- 证明期望的连续性
- 深度学习中的梯度计算

---

**定理的完整证明**:

**证明**:

我们将使用Fatou引理来证明控制收敛定理。

**第一步：建立辅助函数**:

定义：

$$
h_n = g + f_n, \quad k_n = g - f_n
$$

**关键性质**：

由于 $|f_n| \leq g$，我们有：

- $-g \leq f_n \leq g$
- 因此 $h_n = g + f_n \geq 0$
- 且 $k_n = g - f_n \geq 0$

**第二步：计算下极限**:

由于 $f_n \to f$ a.e.，我们有：

$$
\liminf_{n \to \infty} h_n = \liminf_{n \to \infty} (g + f_n) = g + f
$$

$$
\liminf_{n \to \infty} k_n = \liminf_{n \to \infty} (g - f_n) = g - f
$$

（这里使用了：$\liminf (a_n + b_n) = a + \liminf b_n$ 当 $a_n \to a$）

**第三步：应用Fatou引理到 $h_n$**

由于 $h_n \geq 0$，根据Fatou引理：

$$
\int (g + f) \, d\mu = \int \liminf_{n \to \infty} h_n \, d\mu \leq \liminf_{n \to \infty} \int h_n \, d\mu
$$

即：

$$
\int g \, d\mu + \int f \, d\mu \leq \liminf_{n \to \infty} \left(\int g \, d\mu + \int f_n \, d\mu\right)
$$

由于 $\int g \, d\mu < \infty$（$g$ 可积），可以约去：

$$
\int f \, d\mu \leq \liminf_{n \to \infty} \int f_n \, d\mu \quad \quad (*)
$$

**第四步：应用Fatou引理到 $k_n$**

类似地，对 $k_n \geq 0$ 应用Fatou引理：

$$
\int (g - f) \, d\mu = \int \liminf_{n \to \infty} k_n \, d\mu \leq \liminf_{n \to \infty} \int k_n \, d\mu
$$

即：

$$
\int g \, d\mu - \int f \, d\mu \leq \liminf_{n \to \infty} \left(\int g \, d\mu - \int f_n \, d\mu\right)
$$

约去 $\int g \, d\mu$：

$$
-\int f \, d\mu \leq \liminf_{n \to \infty} \left(-\int f_n \, d\mu\right) = -\limsup_{n \to \infty} \int f_n \, d\mu
$$

（使用了 $\liminf(-a_n) = -\limsup(a_n)$）

因此：

$$
\limsup_{n \to \infty} \int f_n \, d\mu \leq \int f \, d\mu \quad \quad (**)
$$

**第五步：结论**:

从 $(*)$ 和 $(**)$：

$$
\liminf_{n \to \infty} \int f_n \, d\mu \leq \int f \, d\mu \leq \limsup_{n \to \infty} \int f_n \, d\mu
$$

但总有 $\liminf \leq \limsup$，因此：

$$
\liminf_{n \to \infty} \int f_n \, d\mu = \limsup_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu
$$

这意味着极限存在且：

$$
\lim_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu
$$

这就完成了控制收敛定理的证明。 $\square$

---

**证明的关键要点**：

1. **控制函数的作用**：$|f_n| \leq g$ 保证了 $g \pm f_n \geq 0$，可以应用Fatou引理
2. **Fatou引理的两次应用**：分别对 $g + f_n$ 和 $g - f_n$ 应用
3. **上下极限的夹逼**：通过 $\liminf$ 和 $\limsup$ 夹逼得到极限存在

**为什么需要控制函数**：

控制函数 $g$ 的可积性是关键。如果没有控制函数，定理不成立。

**反例**：

令 $f_n = n \mathbb{1}_{[0, 1/n]}$ 在 $[0,1]$ 上（Lebesgue测度）。

则 $f_n \to 0$ a.e.，但 $\int f_n = 1$ 对所有 $n$，而 $\int 0 = 0$。

这里没有可积的控制函数。

**深度学习中的应用**：

在计算梯度时，经常需要交换求导和积分（期望）的顺序：

$$
\frac{d}{d\theta} \mathbb{E}[f(X; \theta)] = \mathbb{E}\left[\frac{\partial f}{\partial \theta}(X; \theta)\right]
$$

控制收敛定理保证了这种交换的合法性（当偏导数被可积函数控制时）。

---

## 🤖 在AI中的应用

### 1. 连续型随机变量

**高斯分布**:

$$
f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**多元高斯**:

$$
f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

**应用**: VAE、高斯过程、卡尔曼滤波

---

### 2. 条件期望

**定义**:

给定σ-代数 $\mathcal{G} \subseteq \mathcal{F}$, $\mathbb{E}[X | \mathcal{G}]$ 是$\mathcal{G}$-可测的随机变量,满足:

$$
\int_G \mathbb{E}[X | \mathcal{G}] \, d\mathbb{P} = \int_G X \, d\mathbb{P}, \quad \forall G \in \mathcal{G}
$$

**直观**: 在部分信息下的最佳预测。

**应用**:

- 强化学习(值函数)
- 序列模型(隐马尔可夫)
- 因果推断

---

### 3. 随机过程

**定义**:

随机过程是指标集 $T$ 上的随机变量族:

$$
\{X_t : t \in T\}
$$

**示例**:

- **Brownian运动**: $B_t \sim \mathcal{N}(0, t)$, 连续路径
- **Poisson过程**: 计数过程
- **马尔可夫链**: $\mathbb{P}(X_{n+1} | X_0, \ldots, X_n) = \mathbb{P}(X_{n+1} | X_n)$

**应用**:

- 时间序列建模
- 扩散模型(SDE)
- 强化学习(MDP)

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. 测度论视角: 蒙特卡洛积分
def monte_carlo_integral(f, dist, n_samples=10000):
    """
    使用蒙特卡洛估计期望: E[f(X)]

    Args:
        f: 函数
        dist: 分布 (scipy.stats对象)
        n_samples: 样本数

    Returns:
        期望的估计值
    """
    samples = dist.rvs(size=n_samples)
    return np.mean(f(samples))

# 示例: E[X^2] where X ~ N(0,1)
dist = stats.norm(loc=0, scale=1)
estimate = monte_carlo_integral(lambda x: x**2, dist, n_samples=100000)
true_value = 1.0  # 方差为1
print(f"Estimated E[X^2]: {estimate:.4f} (True: {true_value})")


# 2. 控制收敛定理应用
def dominated_convergence_example():
    """演示控制收敛定理"""
    # 函数序列 f_n(x) = x^n * (1-x)
    # 在 [0,1] 上收敛到 0

    x = np.linspace(0, 1, 1000)

    integrals = []
    for n in [1, 2, 5, 10, 20, 50]:
        f_n = x**n * (1 - x)
        integral = np.trapz(f_n, x)
        integrals.append(integral)

        if n in [1, 10, 50]:
            plt.plot(x, f_n, label=f'n={n}')

    plt.xlabel('x')
    plt.ylabel('f_n(x)')
    plt.title('Dominated Convergence: $f_n(x) = x^n(1-x)$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Integrals: {integrals}")
    print(f"Limit: {integrals[-1]:.6f} → 0")

# dominated_convergence_example()


# 3. 条件期望
def conditional_expectation_demo():
    """条件期望的蒙特卡洛估计"""
    # 联合分布: (X, Y) ~ N(0, Σ)
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]

    samples = np.random.multivariate_normal(mean, cov, size=10000)
    X, Y = samples[:, 0], samples[:, 1]

    # 估计 E[Y | X=x]
    def conditional_expectation(x_value, window=0.1):
        mask = np.abs(X - x_value) < window
        if np.sum(mask) > 0:
            return np.mean(Y[mask])
        return 0

    x_values = np.linspace(-3, 3, 50)
    cond_exp = [conditional_expectation(x) for x in x_values]

    # 理论值: E[Y|X=x] = ρ * x (对于二元高斯)
    theoretical = 0.8 * x_values

    plt.scatter(X, Y, alpha=0.1, s=1, label='Samples')
    plt.plot(x_values, cond_exp, 'r-', linewidth=2, label='Estimated E[Y|X]')
    plt.plot(x_values, theoretical, 'g--', linewidth=2, label='Theoretical')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Conditional Expectation: E[Y|X]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# conditional_expectation_demo()
```

---

## 📚 相关资源

**经典教材**:

1. **Probability: Theory and Examples** - Durrett (2019)
   → 概率论圣经

2. **Real Analysis and Probability** - Dudley (2002)
   → 实分析与概率结合

3. **Probability and Measure** - Billingsley (1995)
   → 测度论概率

**在线资源**:

- MIT OCW 18.175: Theory of Probability
- Stanford STATS310: Theory of Probability

---

## 💡 练习题

**1. σ-代数构造**:

证明: 如果 $\mathcal{F}_1, \mathcal{F}_2$ 是σ-代数,则 $\mathcal{F}_1 \cap \mathcal{F}_2$ 也是σ-代数。

---

**2. 测度的连续性**:

证明: 如果 $A_1 \subseteq A_2 \subseteq \cdots$, 则:

$$
\mu\left(\bigcup_{n=1}^\infty A_n\right) = \lim_{n \to \infty} \mu(A_n)
$$

---

**3. 控制收敛定理应用**:

计算: $\lim_{n \to \infty} \int_0^1 \frac{n x^n}{1+x^2} dx$

---

**📌 下一主题**: [随机变量与概率分布](./02-Random-Variables-and-Distributions.md)

**🔙 返回**: [概率统计](../README.md) | [数学基础](../../README.md)
