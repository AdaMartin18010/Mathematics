# 矩阵分解 (Matrix Decompositions)

> **The Computational Foundation of Machine Learning**
>
> 机器学习的计算基础

---

## 目录

- [矩阵分解 (Matrix Decompositions)](#矩阵分解-matrix-decompositions)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 特征值分解 (Eigendecomposition)](#-特征值分解-eigendecomposition)
    - [1. 特征值与特征向量](#1-特征值与特征向量)
    - [2. 谱定理](#2-谱定理)
    - [3. 对角化](#3-对角化)
  - [📊 奇异值分解 (SVD)](#-奇异值分解-svd)
    - [1. SVD定义](#1-svd定义)
    - [2. 几何解释](#2-几何解释)
    - [3. 截断SVD与低秩近似](#3-截断svd与低秩近似)
    - [4. SVD的性质](#4-svd的性质)
  - [🔬 QR分解](#-qr分解)
    - [1. QR分解定义](#1-qr分解定义)
    - [2. Gram-Schmidt正交化](#2-gram-schmidt正交化)
    - [3. Householder变换](#3-householder变换)
  - [💡 Cholesky分解](#-cholesky分解)
    - [1. Cholesky分解定义](#1-cholesky分解定义)
    - [2. 算法](#2-算法)
  - [🎨 LU分解](#-lu分解)
    - [1. LU分解定义](#1-lu分解定义)
    - [2. 高斯消元法](#2-高斯消元法)
  - [🔧 在深度学习中的应用](#-在深度学习中的应用)
    - [1. 主成分分析 (PCA)](#1-主成分分析-pca)
    - [2. 奇异值分解与降维](#2-奇异值分解与降维)
    - [3. 矩阵求逆与线性系统](#3-矩阵求逆与线性系统)
    - [4. 权重初始化](#4-权重初始化)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：特征值计算](#练习1特征值计算)
    - [练习2：SVD应用](#练习2svd应用)
    - [练习3：PCA实现](#练习3pca实现)
    - [练习4：低秩近似](#练习4低秩近似)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**矩阵分解**是将矩阵表示为更简单矩阵的乘积，是线性代数计算的核心工具。

**为什么矩阵分解重要**:

```text
计算优势:
├─ 简化计算 (如求逆、求解线性系统)
├─ 数值稳定性
└─ 揭示矩阵结构

机器学习应用:
├─ PCA (主成分分析) → SVD
├─ 推荐系统 → 矩阵分解
├─ 降维 → SVD/特征值分解
└─ 优化 → Cholesky分解
```

**主要分解**:

```text
特征值分解 (Eigendecomposition):
    A = QΛQ⁻¹  (方阵, 可对角化)

奇异值分解 (SVD):
    A = UΣVᵀ  (任意矩阵)

QR分解:
    A = QR  (Q正交, R上三角)

Cholesky分解:
    A = LLᵀ  (正定矩阵)

LU分解:
    A = LU  (L下三角, U上三角)
```

---

## 🎯 特征值分解 (Eigendecomposition)

### 1. 特征值与特征向量

**定义 1.1 (特征值与特征向量)**:

设 $A \in \mathbb{R}^{n \times n}$，如果存在标量 $\lambda$ 和非零向量 $v$ 使得：

$$
Av = \lambda v
$$

则 $\lambda$ 称为 $A$ 的**特征值**，$v$ 称为对应的**特征向量**。

**几何意义**：$A$ 作用在 $v$ 上只改变其长度，不改变方向。

**特征多项式**:

$$
\det(A - \lambda I) = 0
$$

**示例**:

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

特征多项式：$\det(A - \lambda I) = (2 - \lambda)^2 - 1 = 0$

特征值：$\lambda_1 = 3, \lambda_2 = 1$

特征向量：$v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, v_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

---

### 2. 谱定理

**定理 2.1 (谱定理)**:

设 $A \in \mathbb{R}^{n \times n}$ 是**对称矩阵**，则：

1. $A$ 的所有特征值都是**实数**
2. 不同特征值对应的特征向量**正交**
3. $A$ 可以**正交对角化**：

$$
A = Q\Lambda Q^T
$$

其中 $Q$ 是正交矩阵（$Q^T Q = I$），$\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$。

**意义**：对称矩阵有完整的正交特征向量基。

---

### 3. 对角化

**定义 3.1 (可对角化)**:

矩阵 $A$ 可对角化，如果存在可逆矩阵 $P$ 和对角矩阵 $D$ 使得：

$$
A = PDP^{-1}
$$

**条件**：$A$ 有 $n$ 个线性无关的特征向量。

**应用**：计算矩阵幂

$$
A^k = PD^kP^{-1}
$$

其中 $D^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$。

---

## 📊 奇异值分解 (SVD)

### 1. SVD定义

**定理 1.1 (奇异值分解)**:

对于任意矩阵 $A \in \mathbb{R}^{m \times n}$，存在分解：

$$
A = U\Sigma V^T
$$

其中：

- $U \in \mathbb{R}^{m \times m}$ 是正交矩阵（左奇异向量）
- $\Sigma \in \mathbb{R}^{m \times n}$ 是对角矩阵，对角元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 称为**奇异值**
- $V \in \mathbb{R}^{n \times n}$ 是正交矩阵（右奇异向量）

**与特征值的关系**:

- $A^T A$ 的特征值是 $\sigma_i^2$
- $A A^T$ 的特征值也是 $\sigma_i^2$
- $V$ 的列是 $A^T A$ 的特征向量
- $U$ 的列是 $A A^T$ 的特征向量

---

### 2. 几何解释

**SVD的几何意义**:

任意线性变换 $A$ 可以分解为：

```text
A = U Σ Vᵀ
    ↓
1. Vᵀ: 旋转 (正交变换)
2. Σ:  缩放 (沿坐标轴)
3. U:  旋转 (正交变换)
```

**示例**:

$$
A = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
$$

已经是对角矩阵，SVD为：

$$
U = I, \quad \Sigma = A, \quad V = I
$$

---

### 3. 截断SVD与低秩近似

**定理 3.1 (Eckart-Young定理)**:

设 $A = U\Sigma V^T$ 是SVD，定义秩为 $k$ 的截断SVD：

$$
A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^k \sigma_i u_i v_i^T
$$

则 $A_k$ 是所有秩不超过 $k$ 的矩阵中，与 $A$ 的Frobenius范数距离最小的矩阵：

$$
A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F
$$

**误差**:

$$
\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}
$$

**应用**：数据压缩、降维、去噪。

---

### 4. SVD的性质

**性质 4.1**:

1. **秩**: $\text{rank}(A) = r$ (非零奇异值的个数)
2. **范数**: $\|A\|_2 = \sigma_1$ (最大奇异值)
3. **Frobenius范数**: $\|A\|_F = \sqrt{\sum_{i=1}^r \sigma_i^2}$
4. **条件数**: $\kappa(A) = \frac{\sigma_1}{\sigma_r}$
5. **伪逆**: $A^+ = V\Sigma^+ U^T$，其中 $\Sigma^+$ 是 $\Sigma$ 的伪逆

---

## 🔬 QR分解

### 1. QR分解定义

**定理 1.1 (QR分解)**:

对于任意矩阵 $A \in \mathbb{R}^{m \times n}$ ($m \geq n$)，存在分解：

$$
A = QR
$$

其中：

- $Q \in \mathbb{R}^{m \times n}$ 是正交矩阵（$Q^T Q = I$）
- $R \in \mathbb{R}^{n \times n}$ 是上三角矩阵

**应用**：

- 求解最小二乘问题
- 计算特征值（QR算法）
- 正交化

---

### 2. Gram-Schmidt正交化

**算法 2.1 (Gram-Schmidt正交化)**:

给定线性无关向量 $a_1, \ldots, a_n$，构造正交向量 $q_1, \ldots, q_n$：

$$
\begin{align}
u_1 &= a_1 \\
u_i &= a_i - \sum_{j=1}^{i-1} \frac{\langle a_i, q_j \rangle}{\langle q_j, q_j \rangle} q_j \\
q_i &= \frac{u_i}{\|u_i\|}
\end{align}
$$

**问题**：数值不稳定（修正Gram-Schmidt更稳定）。

---

### 3. Householder变换

**定义 3.1 (Householder变换)**:

$$
H = I - 2vv^T
$$

其中 $v$ 是单位向量（$\|v\| = 1$）。

**性质**:

- $H$ 是对称正交矩阵（$H = H^T$，$H^2 = I$）
- $H$ 是关于超平面 $\{x : v^T x = 0\}$ 的反射

**应用**：QR分解（Householder QR）

---

## 💡 Cholesky分解

### 1. Cholesky分解定义

**定理 1.1 (Cholesky分解)**:

设 $A \in \mathbb{R}^{n \times n}$ 是**对称正定矩阵**，则存在唯一的下三角矩阵 $L$（对角元素为正）使得：

$$
A = LL^T
$$

**优势**:

- 计算效率高（约为LU分解的一半）
- 数值稳定
- 保证正定性

**应用**：

- 求解线性系统 $Ax = b$
- 高斯过程
- 优化算法

---

### 2. 算法

**算法 2.1 (Cholesky分解算法)**:

$$
L_{ij} = \begin{cases}
\sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2} & \text{if } i = j \\
\frac{1}{L_{jj}} \left( A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk} \right) & \text{if } i > j \\
0 & \text{if } i < j
\end{cases}
$$

**复杂度**: $O(n^3/3)$

---

## 🎨 LU分解

### 1. LU分解定义

**定理 1.1 (LU分解)**:

设 $A \in \mathbb{R}^{n \times n}$，如果 $A$ 的所有顺序主子式非零，则存在分解：

$$
A = LU
$$

其中：

- $L$ 是下三角矩阵（对角元素为1）
- $U$ 是上三角矩阵

**带主元的LU分解**:

$$
PA = LU
$$

其中 $P$ 是置换矩阵。

---

### 2. 高斯消元法

**算法 2.1 (高斯消元法)**:

通过行变换将 $A$ 化为上三角矩阵 $U$，同时记录变换得到 $L$。

**应用**：

- 求解线性系统
- 计算行列式
- 求逆矩阵

---

## 🔧 在深度学习中的应用

### 1. 主成分分析 (PCA)

**问题**：找到数据的主要方向。

**方法**：

1. 中心化数据：$X_c = X - \bar{X}$
2. 计算协方差矩阵：$C = \frac{1}{n} X_c^T X_c$
3. 特征值分解：$C = Q\Lambda Q^T$
4. 主成分：$Q$ 的前 $k$ 列

**等价方法（SVD）**:

1. SVD: $X_c = U\Sigma V^T$
2. 主成分：$V$ 的前 $k$ 列
3. 降维：$Z = X_c V_k$

---

### 2. 奇异值分解与降维

**应用**：

- **图像压缩**：截断SVD
- **推荐系统**：矩阵分解
- **去噪**：保留大奇异值

**示例**（图像压缩）:

```python
A_k = U[:, :k] @ Sigma[:k, :k] @ V[:k, :]
```

压缩率：$\frac{k(m + n)}{mn}$

---

### 3. 矩阵求逆与线性系统

**求解 $Ax = b$**:

- **Cholesky分解**（$A$ 正定）：
  1. $A = LL^T$
  2. 求解 $Ly = b$ (前向替换)
  3. 求解 $L^T x = y$ (后向替换)

- **LU分解**：
  1. $A = LU$
  2. 求解 $Ly = b$
  3. 求解 $Ux = y$

**优势**：避免直接求逆（数值不稳定）。

---

### 4. 权重初始化

**Xavier初始化**（基于特征值分析）:

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
$$

**He初始化**（ReLU网络）:

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
$$

**理论基础**：保持激活值和梯度的方差。

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, qr, cholesky, lu

# 1. 特征值分解
def eigendecomposition_demo():
    """特征值分解示例"""
    # 对称矩阵
    A = np.array([[2, 1], [1, 2]])
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("矩阵 A:")
    print(A)
    print("\n特征值:")
    print(eigenvalues)
    print("\n特征向量:")
    print(eigenvectors)
    
    # 验证: A = QΛQ^T
    Lambda = np.diag(eigenvalues)
    Q = eigenvectors
    A_reconstructed = Q @ Lambda @ Q.T
    
    print("\n重构误差:")
    print(np.linalg.norm(A - A_reconstructed))


# 2. SVD分解
def svd_demo():
    """SVD分解示例"""
    # 创建矩阵
    A = np.array([[3, 2, 2],
                  [2, 3, -2]])
    
    # SVD分解
    U, S, Vt = svd(A)
    
    print("矩阵 A:")
    print(A)
    print(f"\nA的形状: {A.shape}")
    print(f"U的形状: {U.shape}")
    print(f"S的形状: {S.shape}")
    print(f"Vt的形状: {Vt.shape}")
    
    print("\n奇异值:")
    print(S)
    
    # 重构
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    Sigma[:len(S), :len(S)] = np.diag(S)
    A_reconstructed = U @ Sigma @ Vt
    
    print("\n重构误差:")
    print(np.linalg.norm(A - A_reconstructed))


# 3. PCA实现
def pca_demo():
    """PCA降维示例"""
    np.random.seed(42)
    
    # 生成2D数据
    mean = [0, 0]
    cov = [[3, 1.5], [1.5, 1]]
    X = np.random.multivariate_normal(mean, cov, 200)
    
    # PCA (使用SVD)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = svd(X_centered, full_matrices=False)
    
    # 主成分
    principal_components = Vt.T
    
    # 投影到第一主成分
    Z = X_centered @ principal_components[:, 0:1]
    X_reconstructed = Z @ principal_components[:, 0:1].T + X.mean(axis=0)
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 原始数据
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.arrow(0, 0, principal_components[0, 0]*S[0], principal_components[1, 0]*S[0],
              head_width=0.3, head_length=0.3, fc='r', ec='r', label='PC1')
    plt.arrow(0, 0, principal_components[0, 1]*S[1], principal_components[1, 1]*S[1],
              head_width=0.3, head_length=0.3, fc='g', ec='g', label='PC2')
    plt.title('Original Data with Principal Components')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    # 重构数据
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Original')
    plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.5, label='Reconstructed (1D)')
    plt.title('PCA Reconstruction')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    # plt.show()
    
    # 解释方差比例
    explained_variance_ratio = S**2 / np.sum(S**2)
    print("解释方差比例:")
    print(explained_variance_ratio)


# 4. 低秩近似
def low_rank_approximation_demo():
    """低秩近似示例（图像压缩）"""
    # 创建一个简单的"图像"
    np.random.seed(42)
    img = np.random.randn(50, 50)
    
    # SVD
    U, S, Vt = svd(img, full_matrices=False)
    
    # 不同秩的近似
    ranks = [1, 5, 10, 20, 50]
    
    plt.figure(figsize=(15, 3))
    
    for i, k in enumerate(ranks):
        # 截断SVD
        img_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        
        # 计算误差
        error = np.linalg.norm(img - img_k, 'fro') / np.linalg.norm(img, 'fro')
        
        plt.subplot(1, len(ranks), i+1)
        plt.imshow(img_k, cmap='gray')
        plt.title(f'Rank {k}\nError: {error:.3f}')
        plt.axis('off')
    
    plt.tight_layout()
    # plt.show()


# 5. Cholesky分解
def cholesky_demo():
    """Cholesky分解示例"""
    # 创建正定矩阵
    A = np.array([[4, 2, 1],
                  [2, 3, 1],
                  [1, 1, 2]])
    
    print("矩阵 A (正定):")
    print(A)
    
    # Cholesky分解
    L = cholesky(A, lower=True)
    
    print("\nCholesky分解 L:")
    print(L)
    
    # 验证
    A_reconstructed = L @ L.T
    print("\n重构误差:")
    print(np.linalg.norm(A - A_reconstructed))
    
    # 求解线性系统 Ax = b
    b = np.array([1, 2, 3])
    
    # 使用Cholesky分解求解
    y = np.linalg.solve(L, b)  # Ly = b
    x = np.linalg.solve(L.T, y)  # L^T x = y
    
    print("\n求解 Ax = b:")
    print(f"x = {x}")
    print(f"验证 Ax = {A @ x}")


# 6. QR分解
def qr_demo():
    """QR分解示例"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]], dtype=float)
    
    print("矩阵 A:")
    print(A)
    
    # QR分解
    Q, R = qr(A)
    
    print("\nQ (正交矩阵):")
    print(Q)
    print("\nR (上三角矩阵):")
    print(R)
    
    # 验证正交性
    print("\nQ^T Q:")
    print(Q.T @ Q)
    
    # 重构
    A_reconstructed = Q @ R
    print("\n重构误差:")
    print(np.linalg.norm(A - A_reconstructed))


if __name__ == "__main__":
    print("=== 矩阵分解示例 ===\n")
    
    print("1. 特征值分解")
    eigendecomposition_demo()
    
    print("\n" + "="*50 + "\n")
    print("2. SVD分解")
    svd_demo()
    
    print("\n" + "="*50 + "\n")
    print("3. PCA降维")
    pca_demo()
    
    print("\n" + "="*50 + "\n")
    print("4. 低秩近似")
    low_rank_approximation_demo()
    
    print("\n" + "="*50 + "\n")
    print("5. Cholesky分解")
    cholesky_demo()
    
    print("\n" + "="*50 + "\n")
    print("6. QR分解")
    qr_demo()
```

---

## 📚 练习题

### 练习1：特征值计算

计算以下矩阵的特征值和特征向量：

$$
A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}
$$

### 练习2：SVD应用

对矩阵 $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$ 进行SVD分解，并计算秩-1近似。

### 练习3：PCA实现

给定数据矩阵 $X \in \mathbb{R}^{100 \times 5}$，使用PCA将其降至2维。

### 练习4：低秩近似

证明Eckart-Young定理：截断SVD给出最优低秩近似。

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **MIT** | 18.06 - Linear Algebra (Gilbert Strang) |
| **MIT** | 18.065 - Matrix Methods in Data Analysis |
| **Stanford** | CS205L - Continuous Mathematical Methods |
| **UC Berkeley** | Math 110 - Linear Algebra |
| **CMU** | 21-241 - Matrices and Linear Transformations |

---

## 📖 参考文献

1. **Strang, G. (2016)**. *Introduction to Linear Algebra*. Wellesley-Cambridge Press.

2. **Trefethen & Bau (1997)**. *Numerical Linear Algebra*. SIAM.

3. **Golub & Van Loan (2013)**. *Matrix Computations*. Johns Hopkins University Press.

4. **Horn & Johnson (2012)**. *Matrix Analysis*. Cambridge University Press.

5. **Goodfellow et al. (2016)**. *Deep Learning*. MIT Press. (Chapter 2: Linear Algebra)

---

*最后更新：2025年10月*-
