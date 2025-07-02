# 04-累积分布函数 (CDF)

## 1. 寻求统一的描述语言

我们已经分别探讨了描述离散型随机变量的**概率质量函数 (PMF)** 和描述连续型随机变量的**概率密度函数 (PDF)**。PMF 处理的是"点"的概率，而 PDF 处理的是"区间"的概率密度。

这两种工具在各自的领域都很有用，但它们是分裂的。是否存在一种单一的、更通用的函数，能够统一描述任何随机变量（无论是离散的、连续的，还是混合的）的概率分布呢？

答案是肯定的。这个强大的工具就是**累积分布函数 (Cumulative Distribution Function, CDF)**，有时也简称为分布函数。

## 2. 累积分布函数的定义

对于任意一个随机变量 $X$（不区分类型），其累积分布函数 $F_X(x)$ 定义为：

$$ F_X(x) = P(X \le x) $$

这个定义的直观含义是：随机变量 $X$ 的取值小于或等于某个给定值 $x$ 的总概率。顾名思义，它是一个"累积"的概率。

## 3. CDF 的核心性质

任何一个合法的 CDF，$F(x)$，都必须满足以下三个核心性质：

1. **非减性 (Non-decreasing)**
    如果 $x_1 < x_2$，那么 $F(x_1) \le F(x_2)$。
    * 这个性质非常直观。因为 $x_2$ 对应的事件 $\{X \le x_2\}$ 包含了 $x_1$ 对应的事件 $\{X \le x_1\}$，所以其概率必然不会更小。

2. **极限行为 (Limiting Behavior)**
    $$ \lim_{x \to -\infty} F(x) = 0 \quad \text{and} \quad \lim_{x \to \infty} F(x) = 1 $$
    * 当 $x$ 趋向负无穷时，变量取值小于等于 $x$ 是一个不可能事件，概率为 0。当 $x$ 趋向正无穷时，变量取值小于等于 $x$ 是一个必然事件，概率为 1。

3. **右连续性 (Right-continuous)**
    对于任何点 $a$，都有 $\lim_{x \to a^+} F(x) = F(a)$。
    * 这意味着当从右侧逼近任何一点时，函数的极限值等于该点的函数值。对于离散型随机变量，这表现为在跳跃点的函数值取跳跃后的值。

## 4. CDF 与 PMF/PDF 的关系

CDF 的美妙之处在于它如何与我们之前学过的概念无缝连接。

### 4.1 对于离散型随机变量

如果 $X$ 是一个离散型随机变量，其 PMF 为 $p(x_i) = P(X=x_i)$。那么它的 CDF 是一个**阶梯函数 (Step Function)**。

$$ F(x) = \sum_{x_i \le x} p(x_i) $$

CDF 在每个 $x_i$ 处发生跳跃，跳跃的高度恰好等于该点的概率 $p(x_i)$。

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Diskrete_Verteilungsfunktion.svg/450px-Diskrete_Verteilungsfunktion.svg.png" alt="CDF of a discrete distribution" width="400"/>
</div>
<div align-center>图1：离散随机变量的CDF是一个阶梯函数。</div>

### 4.2 对于连续型随机变量

如果 $X$ 是一个连续型随机变量，其 PDF 为 $f(x)$。那么它的 CDF 是 PDF 从负无穷到 $x$ 的积分。

$$ F(x) = P(X \le x) = \int_{-\infty}^{x} f(t) \, dt $$

此时，CDF 是一个**连续函数**。根据微积分基本定理，我们可以反过来从 CDF 求得 PDF：

$$ f(x) = \frac{dF(x)}{dx} = F'(x) $$

这意味着，PDF 恰好是 CDF 的导数，它描述了累积概率的"增长率"。

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Normal_Distribution_CDF.svg/450px-Normal_Distribution_CDF.svg.png" alt="CDF of a continuous distribution" width="400"/>
</div>
<div align-center>图2：连续随机变量（正态分布）的CDF是一条平滑的S形曲线。</div>

## 5. 使用 CDF 计算概率

CDF 最直接的应用就是计算变量落在任意区间的概率。对于任何 $a < b$：

$$ P(a < X \le b) = P(X \le b) - P(X \le a) = F(b) - F(a) $$

这个公式对离散和连续情况都适用，进一步凸显了 CDF 的统一性和强大功能。它是概率论中一个承上启下的、至关重要的概念。
