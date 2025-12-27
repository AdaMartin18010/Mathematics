# 归一化流 (Normalizing Flows)

> **Invertible Transformations for Exact Likelihood Modeling**
>
> 可逆变换的精确似然建模

---

## 目录

- [归一化流 (Normalizing Flows)](#归一化流-normalizing-flows)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [1. 流模型基础](#1-流模型基础)
    - [1.1 可逆变换](#11-可逆变换)
    - [1.2 变量变换公式](#12-变量变换公式)
    - [1.3 流模型定义](#13-流模型定义)
  - [2. 耦合层 (Coupling Layers)](#2-耦合层-coupling-layers)
    - [2.1 Real NVP](#21-real-nvp)
    - [2.2 Glow](#22-glow)
    - [2.3 可逆1x1卷积](#23-可逆1x1卷积)
  - [3. 自回归流](#3-自回归流)
    - [3.1 Masked Autoregressive Flow (MAF)](#31-masked-autoregressive-flow-maf)
    - [3.2 Inverse Autoregressive Flow (IAF)](#32-inverse-autoregressive-flow-iaf)
  - [4. 连续归一化流](#4-连续归一化流)
    - [4.1 Neural ODE](#41-neural-ode)
    - [4.2 FFJORD](#42-ffjord)
  - [5. 流模型的优势与局限](#5-流模型的优势与局限)
    - [5.1 优势](#51-优势)
    - [5.2 局限](#52-局限)
  - [6. 应用](#6-应用)
    - [6.1 密度估计](#61-密度估计)
    - [6.2 生成建模](#62-生成建模)
    - [6.3 变分推断](#63-变分推断)
  - [7. 形式化定义 (Lean)](#7-形式化定义-lean)
  - [8. 习题](#8-习题)
  - [9. 参考资料](#9-参考资料)

---

## 📋 核心思想

**归一化流**通过一系列可逆变换将简单分布转换为复杂分布，允许精确的似然计算。

**为什么归一化流重要**:

```text
核心问题:
├─ 如何精确计算数据似然？
├─ 如何从复杂分布中高效采样？
├─ 如何学习可逆变换？
└─ 如何平衡表达能力和计算效率？

理论工具:
├─ 变量变换公式: 密度变换
├─ 耦合层: 可逆变换构建
├─ 自回归流: 条件依赖建模
└─ 连续流: ODE视角

实践应用:
├─ 密度估计: 精确似然
├─ 生成建模: 高质量生成
├─ 变分推断: 后验近似
└─ 数据增强: 合成数据
```

---

## 1. 流模型基础

### 1.1 可逆变换

**定义 1.1** (可逆变换)
变换 \( f: \mathbb{R}^d \to \mathbb{R}^d \) 称为**可逆的**，如果存在 \( f^{-1} \) 使得 \( f^{-1}(f(x)) = x \)。

**要求**: \( f \) 必须是双射（一一对应且满射）。

### 1.2 变量变换公式

**定理 1.1** (变量变换公式)
设 \( z \sim p_z(z) \)，\( x = f(z) \)，其中 \( f \) 是可逆变换，则：
\[
p_x(x) = p_z(f^{-1}(x)) \left|\det \frac{\partial f^{-1}(x)}{\partial x}\right|
\]

**对数形式**:
\[
\log p_x(x) = \log p_z(f^{-1}(x)) + \log \left|\det \frac{\partial f^{-1}(x)}{\partial x}\right|
\]

### 1.3 流模型定义

**定义 1.2** (归一化流)
**归一化流**是一系列可逆变换的复合：
\[
x = f_K \circ f_{K-1} \circ \cdots \circ f_1(z_0)
\]

其中 \( z_0 \sim p_0(z_0) \) 是简单分布（如标准高斯）。

**对数似然**:
\[
\log p(x) = \log p_0(z_0) - \sum_{k=1}^K \log \left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|
\]

---

## 2. 耦合层 (Coupling Layers)

### 2.1 Real NVP

**Real NVP** (Dinh et al., 2017) 使用**仿射耦合层**：

**前向变换**:
\[
\begin{cases}
x_{1:d} = z_{1:d} \\
x_{d+1:D} = z_{d+1:D} \odot \exp(s(z_{1:d})) + t(z_{1:d})
\end{cases}
\]

其中 \( s \) 和 \( t \) 是神经网络（缩放和平移函数）。

**反向变换**:
\[
\begin{cases}
z_{1:d} = x_{1:d} \\
z_{d+1:D} = (x_{d+1:D} - t(x_{1:d})) \odot \exp(-s(x_{1:d}))
\end{cases}
\]

**雅可比行列式**:
\[
\det \frac{\partial x}{\partial z} = \prod_{i=d+1}^D \exp(s_i(z_{1:d}))
\]

**优势**: 计算高效，雅可比行列式易于计算。

### 2.2 Glow

**Glow** (Kingma & Dhariwal, 2018) 在Real NVP基础上添加：

1. **可逆1x1卷积**: 增强表达能力
2. **Actnorm**: 激活归一化
3. **仿射耦合层**: 与Real NVP相同

### 2.3 可逆1x1卷积

**可逆1x1卷积**:
\[
x = W z
\]

其中 \( W \) 是可学习的 \( d \times d \) 矩阵。

**雅可比行列式**: \( \det W \)

**优势**: 允许通道间的信息混合。

---

## 3. 自回归流

### 3.1 Masked Autoregressive Flow (MAF)

**MAF** (Papamakarios et al., 2017) 使用自回归结构：

**变换**:
\[
x_i = z_i \exp(\alpha_i) + \mu_i
\]

其中 \( \mu_i = \mu_i(x_{1:i-1}) \)，\( \alpha_i = \alpha_i(x_{1:i-1}) \) 是自回归函数。

**雅可比行列式**:
\[
\det \frac{\partial x}{\partial z} = \prod_{i=1}^d \exp(\alpha_i)
\]

**特点**: 前向变换快速，反向变换需要顺序计算。

### 3.2 Inverse Autoregressive Flow (IAF)

**IAF** (Kingma et al., 2016) 反转MAF的依赖关系：

**变换**:
\[
x_i = z_i \exp(\alpha_i) + \mu_i
\]

其中 \( \mu_i = \mu_i(z_{1:i-1}) \)，\( \alpha_i = \alpha_i(z_{1:i-1}) \)。

**特点**: 采样快速，似然计算需要顺序计算。

---

## 4. 连续归一化流

### 4.1 Neural ODE

**Neural ODE** (Chen et al., 2018) 将流模型视为常微分方程：

**ODE**:
\[
\frac{dz(t)}{dt} = f_\theta(z(t), t)
\]

**变换**:
\[
x = z(T) = z(0) + \int_0^T f_\theta(z(t), t) dt
\]

**对数似然**:
\[
\log p(x) = \log p_0(z(0)) - \int_0^T \text{tr}\left(\frac{\partial f_\theta}{\partial z(t)}\right) dt
\]

**优势**: 连续深度，内存高效。

### 4.2 FFJORD

**FFJORD** (Grathwohl et al., 2019) 使用无迹变换估计雅可比行列式：

**对数似然**:
\[
\log p(x) = \log p_0(z(0)) - \int_0^T \text{div}(f_\theta)(z(t), t) dt
\]

其中 \( \text{div}(f) = \sum_i \frac{\partial f_i}{\partial z_i} \)。

**优势**: 避免存储完整雅可比矩阵。

---

## 5. 流模型的优势与局限

### 5.1 优势

1. **精确似然**: 可以精确计算 \( \log p(x) \)
2. **高效采样**: 前向采样快速
3. **潜在空间**: 提供有意义的潜在表示
4. **可逆性**: 双向变换

### 5.2 局限

1. **表达能力**: 需要足够深的网络
2. **计算成本**: 雅可比行列式计算可能昂贵
3. **架构设计**: 需要精心设计可逆层
4. **维度限制**: 高维数据可能困难

---

## 6. 应用

### 6.1 密度估计

**应用**: 学习数据分布 \( p(x) \)，用于异常检测、数据压缩。

### 6.2 生成建模

**应用**: 从 \( p_0(z) \) 采样，通过流变换生成 \( x \)。

### 6.3 变分推断

**应用**: 使用流模型作为变分后验 \( q(z \mid x) \)，提高近似质量。

---

## 7. 形式化定义 (Lean)

```lean
-- 可逆变换
structure InvertibleTransform (X : Type) where
  forward : X → X
  inverse : X → X
  inv_proof : ∀ x, inverse (forward x) = x

-- 变量变换公式
theorem change_of_variables (f : ℝⁿ → ℝⁿ) (p_z : ℝⁿ → ℝ) :
  p_x x = p_z (f⁻¹ x) * |det (∂f⁻¹/∂x)|

-- 归一化流
def normalizing_flow (f : Fin K → InvertibleTransform ℝⁿ) (z₀ : ℝⁿ) : ℝⁿ :=
  foldl (· ∘ ·) id f z₀
```

---

## 8. 习题

### 基础习题

1. **变量变换**:
   推导变量变换公式。

2. **Real NVP**:
   实现Real NVP的仿射耦合层。

3. **雅可比行列式**:
   计算Real NVP的雅可比行列式。

### 进阶习题

1. **Neural ODE**:
   实现Neural ODE流模型。

2. **自回归流**:
   比较MAF和IAF的优缺点。

3. **流模型应用**:
   使用流模型进行变分推断。

---

## 9. 参考资料

### 教材

1. **Bishop, C. M.** *Pattern Recognition and Machine Learning*. Springer, 2006. (Chapter 2)

### 课程

1. **Stanford CS236** - Deep Generative Models
2. **MIT 6.S192** - Deep Learning for Art, Aesthetics, and Creativity

### 论文

1. **Dinh, L. et al.** "Density estimation using Real NVP." *ICLR*, 2017.
2. **Kingma, D. P. & Dhariwal, P.** "Glow: Generative Flow with Invertible 1x1 Convolutions." *NeurIPS*, 2018.
3. **Papamakarios, G. et al.** "Masked Autoregressive Flow for Density Estimation." *NeurIPS*, 2017.
4. **Chen, T. Q. et al.** "Neural Ordinary Differential Equations." *NeurIPS*, 2018.
5. **Grathwohl, W. et al.** "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." *ICLR*, 2019.

---

**最后更新**: 2025-12-20
**完成度**: 约75% (核心内容完成，可继续补充更多应用实例和形式化证明)
