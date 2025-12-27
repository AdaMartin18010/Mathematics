# 向量空间与线性映射 (Vector Spaces and Linear Maps)

> **AI基础的基础**：理解神经网络权重、数据表示和变换的数学本质

---

## 目录

- [向量空间与线性映射 (Vector Spaces and Linear Maps)](#向量空间与线性映射-vector-spaces-and-linear-maps)
  - [目录](#目录)
  - [📋 学习目标](#-学习目标)
  - [1. 向量空间 (Vector Space)](#1-向量空间-vector-space)
    - [1.1 定义](#11-定义)
    - [1.2 重要例子](#12-重要例子)
  - [2. 子空间 (Subspace)](#2-子空间-subspace)
    - [2.1 定义](#21-定义)
    - [2.2 重要例子](#22-重要例子)
  - [3. 线性无关与基 (Linear Independence and Basis)](#3-线性无关与基-linear-independence-and-basis)
    - [3.1 线性组合与张成](#31-线性组合与张成)
    - [3.2 线性无关](#32-线性无关)
    - [3.3 基与维数](#33-基与维数)
  - [4. 线性映射 (Linear Maps)](#4-线性映射-linear-maps)
    - [4.1 定义](#41-定义)
    - [4.2 矩阵表示](#42-矩阵表示)
    - [4.3 核与秩](#43-核与秩)
  - [5. AI中的应用实例](#5-ai中的应用实例)
    - [5.1 全连接神经网络层](#51-全连接神经网络层)
    - [5.2 主成分分析 (PCA)](#52-主成分分析-pca)
    - [5.3 Transformer中的线性投影](#53-transformer中的线性投影)
  - [6. 形式化定义 (Lean)](#6-形式化定义-lean)
  - [7. 习题](#7-习题)
    - [基础习题](#基础习题)
    - [进阶习题](#进阶习题)
  - [8. 参考资料](#8-参考资料)
    - [教材](#教材)
    - [课程](#课程)
    - [论文](#论文)

## 📋 学习目标

完成本章后，你将能够：

- ✅ 严格定义向量空间及其子空间
- ✅ 理解线性无关、基与维数的概念
- ✅ 掌握线性映射的性质与表示
- ✅ 应用这些概念于神经网络结构

---

## 1. 向量空间 (Vector Space)

### 1.1 定义

**定义 1.1** (向量空间)  
设 \( V \) 是一个非空集合，\( \mathbb{F} \) 是一个域（通常为 \( \mathbb{R} \) 或 \( \mathbb{C} \)）。若 \( V \) 上定义了两种运算：

- **向量加法**: \( + : V \times V \to V \)
- **标量乘法**: \( \cdot : \mathbb{F} \times V \to V \)

满足以下8条公理，则称 \( V \) 为 \( \mathbb{F} \) 上的**向量空间**：

**加法公理**：

1. **交换律**: \( \mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u} \)
2. **结合律**: \( (\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w}) \)
3. **零元素**: 存在 \( \mathbf{0} \in V \) 使得 \( \mathbf{v} + \mathbf{0} = \mathbf{v} \)
4. **逆元素**: 对每个 \( \mathbf{v} \in V \), 存在 \( -\mathbf{v} \in V \) 使得 \( \mathbf{v} + (-\mathbf{v}) = \mathbf{0} \)

**标量乘法公理**：
5. **结合律**: \( a(b\mathbf{v}) = (ab)\mathbf{v} \)
6. **单位元**: \( 1\mathbf{v} = \mathbf{v} \)
7. **分配律I**: \( a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v} \)
8. **分配律II**: \( (a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v} \)

### 1.2 重要例子

**例 1.1** (\( \mathbb{R}^n \))  
\( n \) 维欧几里得空间，所有 \( n \) 维实向量的集合：
\[
\mathbb{R}^n = \{(x_1, x_2, \ldots, x_n) : x_i \in \mathbb{R}\}
\]

**例 1.2** (函数空间)  
从集合 \( X \) 到 \( \mathbb{R} \) 的所有函数构成向量空间：
\[
\mathcal{F}(X, \mathbb{R}) = \{f : X \to \mathbb{R}\}
\]

**例 1.3** (矩阵空间)  
所有 \( m \times n \) 矩阵构成向量空间：
\[
\mathbb{R}^{m \times n} = \{A : A \text{ 是 } m \times n \text{ 实矩阵}\}
\]

**AI应用**：

- \( \mathbb{R}^n \): 神经网络输入/输出向量
- \( \mathbb{R}^{m \times n} \): 权重矩阵
- 函数空间：神经网络函数类

---

## 2. 子空间 (Subspace)

### 2.1 定义

**定义 2.1** (子空间)  
设 \( V \) 是向量空间，\( U \subseteq V \)。若 \( U \) 满足：

1. \( \mathbf{0} \in U \)
2. 对加法封闭：\( \mathbf{u}, \mathbf{v} \in U \Rightarrow \mathbf{u} + \mathbf{v} \in U \)
3. 对标量乘法封闭：\( a \in \mathbb{F}, \mathbf{u} \in U \Rightarrow a\mathbf{u} \in U \)

则称 \( U \) 是 \( V \) 的**子空间**。

### 2.2 重要例子

**例 2.1** (核空间/零空间)  
对线性映射 \( T: V \to W \)，定义：
\[
\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}
\]

**例 2.2** (值域/像空间)  
\[
\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}
\]

**AI应用**：

- 核空间：神经网络中的零激活子空间
- 值域：特征表示空间

---

## 3. 线性无关与基 (Linear Independence and Basis)

### 3.1 线性组合与张成

**定义 3.1** (线性组合)  
向量 \( \mathbf{v} \) 是向量集合 \( \{\mathbf{v}_1, \ldots, \mathbf{v}_k\} \) 的**线性组合**，若存在标量 \( a_1, \ldots, a_k \) 使得：
\[
\mathbf{v} = a_1\mathbf{v}_1 + a_2\mathbf{v}_2 + \cdots + a_k\mathbf{v}_k
\]

**定义 3.2** (张成/生成子空间)  
向量集合 \( S = \{\mathbf{v}_1, \ldots, \mathbf{v}_k\} \) 的**张成**是其所有线性组合的集合：
\[
\text{span}(S) = \left\{\sum_{i=1}^k a_i\mathbf{v}_i : a_i \in \mathbb{F}\right\}
\]

### 3.2 线性无关

**定义 3.3** (线性无关)  
向量集合 \( \{\mathbf{v}_1, \ldots, \mathbf{v}_k\} \) 是**线性无关**的，若方程：
\[
a_1\mathbf{v}_1 + a_2\mathbf{v}_2 + \cdots + a_k\mathbf{v}_k = \mathbf{0}
\]
仅有平凡解 \( a_1 = a_2 = \cdots = a_k = 0 \)。

否则称为**线性相关**。

**定理 3.1** (线性相关等价条件)  
向量集合 \( \{\mathbf{v}_1, \ldots, \mathbf{v}_k\} \) 线性相关当且仅当其中某个向量可以表示为其他向量的线性组合。

### 3.3 基与维数

**定义 3.4** (基)  
向量空间 \( V \) 的一个**基**是 \( V \) 的一个线性无关的张成集。

**定理 3.2** (基的等价定义)  
集合 \( B = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\} \) 是 \( V \) 的基当且仅当 \( V \) 中每个向量都可以唯一表示为 \( B \) 中向量的线性组合。

**定理 3.3** (维数定理)  
有限维向量空间的任意两个基具有相同的元素个数，称为空间的**维数** \( \dim(V) \)。

**AI应用**：

- 输入特征的线性无关性分析
- 降维（PCA）的数学基础
- 神经网络表示能力的维数分析

---

## 4. 线性映射 (Linear Maps)

### 4.1 定义

**定义 4.1** (线性映射)  
设 \( V, W \) 是向量空间，映射 \( T: V \to W \) 称为**线性映射**（或线性变换），若：

1. **加法保持性**: \( T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) \)
2. **标量乘法保持性**: \( T(a\mathbf{v}) = aT(\mathbf{v}) \)

等价地，可以合并为：
\[
T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})
\]

### 4.2 矩阵表示

**定理 4.1** (线性映射的矩阵表示)  
设 \( T: \mathbb{R}^n \to \mathbb{R}^m \) 是线性映射。则存在唯一的 \( m \times n \) 矩阵 \( A \) 使得：
\[
T(\mathbf{x}) = A\mathbf{x}
\]

矩阵 \( A \) 的第 \( j \) 列是 \( T(\mathbf{e}_j) \)，其中 \( \mathbf{e}_j \) 是标准基向量。

**AI应用**：

- 神经网络的全连接层：\( \mathbf{y} = W\mathbf{x} + \mathbf{b} \)
- 卷积操作的矩阵形式
- 注意力机制中的线性投影

### 4.3 核与秩

**定义 4.2** (核/零空间)  
\[
\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}
\]

**定义 4.3** (值域/像)  
\[
\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}
\]

**定义 4.4** (秩与零化度)  

- **秩**: \( \text{rank}(T) = \dim(\text{Im}(T)) \)
- **零化度**: \( \text{nullity}(T) = \dim(\ker(T)) \)

**定理 4.2** (秩-零化度定理)  
设 \( T: V \to W \) 是线性映射，\( V \) 是有限维的，则：
\[
\dim(V) = \text{rank}(T) + \text{nullity}(T)
\]

---

**秩-零化度定理的完整证明**:

**证明**:

设 $T: V \to W$ 是线性映射，$\dim(V) = n$。

**第一步：选择核的基**:

设 $\dim(\ker(T)) = k$（即零化度为 $k$）。

选择 $\ker(T)$ 的一组基：

$$
\{v_1, v_2, \ldots, v_k\}
$$

**第二步：扩展为V的基**:

由于 $\ker(T)$ 是 $V$ 的子空间，根据基扩张定理，可以将 $\ker(T)$ 的基扩张为 $V$ 的基：

$$
\{v_1, v_2, \ldots, v_k, v_{k+1}, v_{k+2}, \ldots, v_n\}
$$

其中 $n = \dim(V)$。

**第三步：证明像空间的生成集**:

**断言**: $\{T(v_{k+1}), T(v_{k+2}), \ldots, T(v_n)\}$ 生成 $\text{Im}(T)$。

**证明断言**:

对于任意 $w \in \text{Im}(T)$，存在 $v \in V$ 使得 $w = T(v)$。

由于 $\{v_1, \ldots, v_n\}$ 是 $V$ 的基，可以写：

$$
v = \sum_{i=1}^k c_i v_i + \sum_{j=k+1}^n c_j v_j
$$

应用 $T$：

$$
w = T(v) = T\left(\sum_{i=1}^k c_i v_i\right) + T\left(\sum_{j=k+1}^n c_j v_j\right)
$$

由于 $v_1, \ldots, v_k \in \ker(T)$，有 $T(v_i) = 0$ 对 $i = 1, \ldots, k$。

因此：

$$
w = \sum_{j=k+1}^n c_j T(v_j)
$$

这说明 $\{T(v_{k+1}), \ldots, T(v_n)\}$ 生成 $\text{Im}(T)$。

**第四步：证明像空间基的线性独立性**:

**断言**: $\{T(v_{k+1}), T(v_{k+2}), \ldots, T(v_n)\}$ 线性独立。

**证明断言**:

假设存在标量 $c_{k+1}, \ldots, c_n$ 使得：

$$
\sum_{j=k+1}^n c_j T(v_j) = 0
$$

由线性映射的性质：

$$
T\left(\sum_{j=k+1}^n c_j v_j\right) = 0
$$

这意味着：

$$
\sum_{j=k+1}^n c_j v_j \in \ker(T)
$$

由于 $\{v_1, \ldots, v_k\}$ 是 $\ker(T)$ 的基，存在标量 $c_1, \ldots, c_k$ 使得：

$$
\sum_{j=k+1}^n c_j v_j = \sum_{i=1}^k c_i v_i
$$

重新整理：

$$
\sum_{i=1}^k (-c_i) v_i + \sum_{j=k+1}^n c_j v_j = 0
$$

由于 $\{v_1, \ldots, v_n\}$ 是 $V$ 的基，它们线性独立，因此：

$$
c_1 = c_2 = \cdots = c_k = 0 \quad \text{和} \quad c_{k+1} = c_{k+2} = \cdots = c_n = 0
$$

这证明了 $\{T(v_{k+1}), \ldots, T(v_n)\}$ 线性独立。

**第五步：计算维数**:

由第三步和第四步，$\{T(v_{k+1}), \ldots, T(v_n)\}$ 是 $\text{Im}(T)$ 的一组基。

因此：

$$
\text{rank}(T) = \dim(\text{Im}(T)) = n - k
$$

而：

$$
\text{nullity}(T) = \dim(\ker(T)) = k
$$

所以：

$$
\text{rank}(T) + \text{nullity}(T) = (n - k) + k = n = \dim(V)
$$

$\square$

---

**证明的关键要点**:

1. **基扩张**: 从 $\ker(T)$ 的基扩张到 $V$ 的基
2. **像的生成**: 证明扩张部分的像生成整个像空间
3. **线性独立性**: 证明像向量线性独立
4. **维数计算**: 通过计数得到秩-零化度关系

---

**几何直观**:

秩-零化度定理说明：

- **输入空间的维数** = **输出空间的维数** + **被压缩到零的维数**

可以想象：

- $V$ 的维数 $n$ 是"输入维度"
- $\ker(T)$ 的维数 $k$ 是"被压缩的维度"（映射到零）
- $\text{Im}(T)$ 的维数 $n-k$ 是"有效输出维度"

---

**矩阵形式**:

对于矩阵 $A \in \mathbb{R}^{m \times n}$，秩-零化度定理变为：

$$
n = \text{rank}(A) + \dim(\ker(A))
$$

其中：

- $n$ 是列数（输入维度）
- $\text{rank}(A)$ 是列秩（有效输出维度）
- $\dim(\ker(A))$ 是零空间维数（冗余维度）

---

**AI中的应用例子**:

**例1：神经网络权重矩阵分析**:

设权重矩阵 $W \in \mathbb{R}^{100 \times 50}$（输入50维，输出100维）。

如果 $\text{rank}(W) = 30$，则：

$$
\dim(\ker(W)) = 50 - 30 = 20
$$

**解释**:

- 输入空间有20个维度是"冗余"的（被映射到零）
- 只有30个有效输出维度
- 可以通过低秩分解压缩模型

**例2：主成分分析（PCA）**:

数据矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 个样本，$d$ 个特征）。

如果 $\text{rank}(X) = r < d$，则：

$$
\dim(\ker(X^T)) = d - r
$$

**解释**:

- 有 $d - r$ 个特征是线性相关的
- 可以降维到 $r$ 维而不损失信息

**例3：模型压缩**:

设神经网络层 $f(x) = Wx$，$W \in \mathbb{R}^{m \times n}$。

如果 $\text{rank}(W) = r \ll \min(m, n)$，可以分解：

$$
W = UV^T
$$

其中 $U \in \mathbb{R}^{m \times r}$，$V \in \mathbb{R}^{n \times r}$。

**参数减少**:

- 原始: $mn$ 个参数
- 压缩后: $(m + n)r$ 个参数
- 节省: $mn - (m+n)r$ 个参数

---

**AI应用**：

- 权重矩阵的秩：表示能力分析
- 低秩分解：模型压缩
- 零化度：冗余神经元检测

---

## 5. AI中的应用实例

### 5.1 全连接神经网络层

一个全连接层可以表示为：
\[
\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})
\]

其中：

- \( W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \): 权重矩阵（线性映射）
- \( \mathbf{b} \in \mathbb{R}^{d_{\text{out}}} \): 偏置（平移）
- \( \sigma \): 激活函数（非线性）

**线性部分分析**：

- 输入空间：\( \mathbb{R}^{d_{\text{in}}} \)
- 输出空间：\( \mathbb{R}^{d_{\text{out}}} \)
- 表示能力：由 \( \text{rank}(W) \) 决定

### 5.2 主成分分析 (PCA)

PCA寻找数据的主方向，数学上是求协方差矩阵的特征向量：

1. 中心化数据：\( \tilde{X} = X - \bar{X} \)
2. 计算协方差矩阵：\( C = \frac{1}{n}\tilde{X}^T\tilde{X} \)
3. 求特征向量：\( C\mathbf{v}_i = \lambda_i\mathbf{v}_i \)
4. 投影到主子空间：\( \mathbf{z} = V_k^T\mathbf{x} \)

其中 \( V_k \) 的列是前 \( k \) 个主成分（特征向量），构成一个 \( k \) 维子空间的基。

### 5.3 Transformer中的线性投影

Transformer的注意力机制使用三个线性映射：
\[
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
\]

这些都是从输入空间到查询/键/值空间的线性映射。

---

## 6. 形式化定义 (Lean)

```lean
-- 向量空间的公理化定义
class VectorSpace (F V : Type*) [Field F] extends
  AddCommGroup V,
  Module F V

-- 子空间
structure Subspace (F V : Type*) [Field F] [VectorSpace F V] where
  carrier : Set V
  zero_mem : 0 ∈ carrier
  add_mem : ∀ {x y}, x ∈ carrier → y ∈ carrier → x + y ∈ carrier
  smul_mem : ∀ {c x}, x ∈ carrier → c • x ∈ carrier

-- 线性映射
structure LinearMap (F V W : Type*) [Field F] 
  [VectorSpace F V] [VectorSpace F W] where
  toFun : V → W
  map_add : ∀ x y, toFun (x + y) = toFun x + toFun y
  map_smul : ∀ c x, toFun (c • x) = c • toFun x

-- 秩-零化度定理
theorem rank_nullity {F V W : Type*} [Field F]
  [FiniteDimensional F V] [VectorSpace F V] [VectorSpace F W]
  (f : V →ₗ[F] W) :
  FiniteDimensional.finrank F V = 
    FiniteDimensional.finrank F (LinearMap.range f) +
    FiniteDimensional.finrank F (LinearMap.ker f) := by
  sorry
```

---

## 7. 习题

### 基础习题

**练习 1.1**  
证明 \( \mathbb{R}^{2 \times 2} \)（所有2×2实矩阵）构成向量空间。

**练习 1.2**  
判断以下集合是否是 \( \mathbb{R}^3 \) 的子空间：

- \( U_1 = \{(x, y, z) : x + y + z = 0\} \)
- \( U_2 = \{(x, y, z) : x + y + z = 1\} \)
- \( U_3 = \{(x, 2x, 3x) : x \in \mathbb{R}\} \)

**练习 1.3**  
判断向量 \( (1,2,3), (4,5,6), (7,8,9) \) 是否线性无关。

### 进阶习题

**练习 1.4**  
设 \( T: \mathbb{R}^3 \to \mathbb{R}^2 \) 定义为 \( T(x,y,z) = (x+y, y+z) \)。

- 求 \( \ker(T) \) 和 \( \text{Im}(T) \)
- 验证秩-零化度定理

**练习 1.5**  
设神经网络权重矩阵 \( W \in \mathbb{R}^{100 \times 50} \)，\( \text{rank}(W) = 30 \)。

- 输出空间的最大维数是多少？
- 这对模型表示能力意味着什么？

---

## 8. 参考资料

### 教材

- **Axler**: _Linear Algebra Done Right_ (Chapter 1-3)
- **Strang**: _Linear Algebra and Its Applications_ (Chapter 1-4)
- **Hoffman & Kunze**: _Linear Algebra_ (Chapter 1-2)

### 课程

- **MIT 18.06**: Linear Algebra (Lectures 1-10)
- **Stanford CS229**: Linear Algebra Review

### 论文

- _The Matrix Calculus You Need For Deep Learning_ (Parr & Howard, 2018)
- _Neural Networks and Deep Learning_ (Nielsen, Chapter 2)

---

**下一章**: [矩阵分解与谱理论](./02-Matrix-Decompositions.md)

**创建时间**: 2025-10-04  
**最后更新**: 2025-10-04
