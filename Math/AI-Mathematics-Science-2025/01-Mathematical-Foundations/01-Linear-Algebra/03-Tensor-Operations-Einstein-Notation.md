# 张量运算与Einstein求和约定 (Tensor Operations & Einstein Notation)

> **The Language of Modern Deep Learning**
>
> 现代深度学习的语言

---

## 目录

- [张量运算与Einstein求和约定 (Tensor Operations \& Einstein Notation)](#张量运算与einstein求和约定-tensor-operations--einstein-notation)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 张量基础](#-张量基础)
    - [1. 张量定义](#1-张量定义)
    - [2. 张量的秩与形状](#2-张量的秩与形状)
    - [3. 张量的索引](#3-张量的索引)
  - [📊 Einstein求和约定](#-einstein求和约定)
    - [1. 基本规则](#1-基本规则)
    - [2. 常见运算](#2-常见运算)
    - [3. 优势](#3-优势)
  - [🔬 张量运算](#-张量运算)
    - [1. 基本运算](#1-基本运算)
    - [2. 张量积](#2-张量积)
    - [3. 张量缩并 (Contraction)](#3-张量缩并-contraction)
  - [💡 在深度学习中的应用](#-在深度学习中的应用)
    - [1. 全连接层](#1-全连接层)
    - [2. 卷积层](#2-卷积层)
    - [3. 注意力机制](#3-注意力机制)
    - [4. 批处理](#4-批处理)
  - [🎨 张量分解](#-张量分解)
    - [1. CP分解 (CANDECOMP/PARAFAC)](#1-cp分解-candecompparafac)
    - [2. Tucker分解](#2-tucker分解)
    - [3. 张量网络](#3-张量网络)
  - [🔧 高级张量运算](#-高级张量运算)
    - [1. 张量重塑 (Reshape)](#1-张量重塑-reshape)
    - [2. 转置与置换](#2-转置与置换)
    - [3. 广播 (Broadcasting)](#3-广播-broadcasting)
    - [4. 张量切片](#4-张量切片)
  - [💻 Python实现](#-python实现)
  - [📚 练习题](#-练习题)
    - [练习1：Einstein求和约定](#练习1einstein求和约定)
    - [练习2：张量缩并](#练习2张量缩并)
    - [练习3：卷积运算](#练习3卷积运算)
    - [练习4：注意力机制](#练习4注意力机制)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**张量**是向量和矩阵的高维推广，是现代深度学习的核心数据结构。**Einstein求和约定**提供了一种简洁优雅的张量运算表示方法。

**为什么张量重要**:

```text
深度学习中的张量:
├─ 数据表示: 图像、视频、文本
├─ 模型参数: 权重、偏置
├─ 中间激活: 特征图
└─ 梯度: 反向传播

Einstein约定优势:
├─ 简洁表示
├─ 避免求和符号
├─ 清晰的索引结构
└─ 易于推导
```

---

## 🎯 张量基础

### 1. 张量定义

**定义 1.1 (张量)**:

张量是多维数组，是标量、向量、矩阵的推广。

**数学定义**:

一个 $n$ 阶张量 $\mathcal{T}$ 是一个多线性映射：

$$
\mathcal{T}: V_1^* \times V_2^* \times \cdots \times V_n^* \to \mathbb{R}
$$

其中 $V_i^*$ 是向量空间 $V_i$ 的对偶空间。

**实际理解**:

- 0阶张量：标量 (scalar)
- 1阶张量：向量 (vector)
- 2阶张量：矩阵 (matrix)
- 3阶张量：立方体 (cube)
- n阶张量：n维数组

---

### 2. 张量的秩与形状

**秩 (Rank/Order)**:

张量的维数，即索引的数量。

**形状 (Shape)**:

每个维度的大小。

**示例**:

- 标量: 秩=0, 形状=()
- 向量 $\mathbf{v} \in \mathbb{R}^n$: 秩=1, 形状=(n,)
- 矩阵 $A \in \mathbb{R}^{m \times n}$: 秩=2, 形状=(m, n)
- RGB图像: 秩=3, 形状=(H, W, 3)
- 批量图像: 秩=4, 形状=(B, H, W, C)

---

### 3. 张量的索引

**索引表示**:

- 向量: $v_i$
- 矩阵: $A_{ij}$
- 3阶张量: $T_{ijk}$
- n阶张量: $T_{i_1 i_2 \cdots i_n}$

**索引约定**:

- 上标：逆变索引 (contravariant)
- 下标：协变索引 (covariant)
- 深度学习中通常使用下标

---

## 📊 Einstein求和约定

### 1. 基本规则

**规则 1.1 (Einstein求和约定)**:

当一个索引在表达式中出现两次（一次上标，一次下标，或两次下标），则对该索引求和，且求和符号 $\sum$ 可以省略。

**示例**:

传统表示：

$$
\sum_{i=1}^n a_i b_i
$$

Einstein约定：

$$
a_i b_i
$$

**重复索引**称为**哑指标 (dummy index)**，非重复索引称为**自由指标 (free index)**。

---

### 2. 常见运算

**向量内积**:

传统：$\mathbf{a}^T \mathbf{b} = \sum_{i=1}^n a_i b_i$

Einstein：$a_i b_i$

**矩阵-向量乘法**:

传统：$(\mathbf{A}\mathbf{x})_i = \sum_{j=1}^n A_{ij} x_j$

Einstein：$(Ax)_i = A_{ij} x_j$

**矩阵乘法**:

传统：$(AB)_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$

Einstein：$(AB)_{ij} = A_{ik} B_{kj}$

**矩阵迹**:

传统：$\text{tr}(A) = \sum_{i=1}^n A_{ii}$

Einstein：$\text{tr}(A) = A_{ii}$

**Frobenius内积**:

传统：$\langle A, B \rangle = \sum_{i=1}^m \sum_{j=1}^n A_{ij} B_{ij}$

Einstein：$\langle A, B \rangle = A_{ij} B_{ij}$

---

### 3. 优势

**简洁性**:

- 避免繁琐的求和符号
- 表达式更紧凑

**清晰性**:

- 索引结构一目了然
- 自由指标明确

**易于推导**:

- 链式法则简洁
- 梯度计算直观

---

## 🔬 张量运算

### 1. 基本运算

**逐元素运算**:

- 加法：$C_{ijk} = A_{ijk} + B_{ijk}$
- 乘法：$C_{ijk} = A_{ijk} \cdot B_{ijk}$

**标量乘法**:

$$
C_{ijk} = \alpha A_{ijk}
$$

---

### 2. 张量积

**外积 (Outer Product)**:

向量外积：

$$
C_{ij} = a_i b_j
$$

张量外积：

$$
D_{ijkl} = A_{ij} B_{kl}
$$

**示例**:

$\mathbf{a} \in \mathbb{R}^m$, $\mathbf{b} \in \mathbb{R}^n$，则 $\mathbf{a} \otimes \mathbf{b} \in \mathbb{R}^{m \times n}$。

---

### 3. 张量缩并 (Contraction)

**定义 3.1 (缩并)**:

对张量的两个索引求和，降低张量的秩。

**示例**:

矩阵乘法是缩并：

$$
C_{ij} = A_{ik} B_{kj}
$$

张量缩并：

$$
C_{ij} = T_{ikj} \quad \text{(对 } k \text{ 求和)}
$$

**性质**:

- 缩并降低秩：$(p, q)$ 型张量缩并后变为 $(p-1, q-1)$ 型
- 矩阵迹是完全缩并：$\text{tr}(A) = A_{ii}$

---

## 💡 在深度学习中的应用

### 1. 全连接层

**前向传播**:

$$
y_i = W_{ij} x_j + b_i
$$

**批量处理**:

$$
Y_{bi} = W_{ij} X_{bj} + b_i
$$

其中 $b$ 是批量索引。

**梯度**:

$$
\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} x_j
$$

Einstein约定：

$$
\frac{\partial L}{\partial W_{ij}} = \delta_i x_j
$$

其中 $\delta_i = \frac{\partial L}{\partial y_i}$。

---

### 2. 卷积层

**2D卷积**:

$$
Y_{bchw} = W_{ckhw'} X_{bk(h+h')(w+w')}
$$

其中：

- $b$: 批量索引
- $c$: 输出通道
- $k$: 输入通道
- $h, w$: 空间位置
- $h', w'$: 卷积核位置

**简化表示**:

$$
Y = W * X
$$

---

### 3. 注意力机制

**Scaled Dot-Product Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**Einstein约定**:

$$
A_{ij} = \frac{Q_{ik} K_{jk}}{\sqrt{d_k}}
$$

$$
\text{Output}_{ik} = \text{softmax}(A)_{ij} V_{jk}
$$

**批量多头注意力**:

$$
\text{Output}_{bhik} = \text{softmax}(A)_{bhij} V_{bhjk}
$$

其中：

- $b$: 批量
- $h$: 头数
- $i, j$: 序列位置
- $k$: 特征维度

---

### 4. 批处理

**批量矩阵乘法**:

$$
Y_{bij} = X_{bik} W_{bkj}
$$

**批量归一化**:

$$
\hat{x}_{bi} = \frac{x_{bi} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
$$

其中：

$$
\mu_i = \frac{1}{B} x_{bi}
$$

$$
\sigma_i^2 = \frac{1}{B} (x_{bi} - \mu_i)^2
$$

---

## 🎨 张量分解

### 1. CP分解 (CANDECOMP/PARAFAC)

**定义**:

将3阶张量 $\mathcal{T} \in \mathbb{R}^{I \times J \times K}$ 分解为秩1张量的和：

$$
T_{ijk} = \sum_{r=1}^R a_{ir} b_{jr} c_{kr}
$$

矩阵形式：

$$
\mathcal{T} = \sum_{r=1}^R \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r
$$

其中 $\circ$ 表示外积。

**应用**:

- 模型压缩
- 特征提取
- 推荐系统

---

### 2. Tucker分解

**定义**:

$$
T_{ijk} = G_{pqr} A_{ip} B_{jq} C_{kr}
$$

其中 $G$ 是核心张量，$A, B, C$ 是因子矩阵。

**矩阵形式**:

$$
\mathcal{T} = \mathcal{G} \times_1 A \times_2 B \times_3 C
$$

**与SVD的关系**:

Tucker分解是SVD的高阶推广。

---

### 3. 张量网络

**张量网络表示**:

复杂的张量运算可以用网络图表示：

```text
    i       j
    |       |
  ┌─┴─┐   ┌─┴─┐
  │ A │───│ B │
  └─┬─┘   └─┬─┘
    k       l
```

表示：$C_{ijkl} = A_{ik} B_{jl}$

**应用**:

- 量子计算
- 深度学习模型压缩
- 物理模拟

---

## 🔧 高级张量运算

### 1. 张量重塑 (Reshape)

**定义**:

改变张量的形状，但保持元素总数不变。

**示例**:

$(2, 3, 4) \to (6, 4)$ 或 $(24,)$

**应用**:

- 展平 (Flatten): $(B, H, W, C) \to (B, H \times W \times C)$
- 重塑: $(B, L, D) \to (B, L, H, D/H)$ (多头注意力)

---

### 2. 转置与置换

**转置 (Transpose)**:

交换两个维度：

$$
B_{ji} = A_{ij}
$$

**置换 (Permute)**:

任意重排维度：

$$
B_{ikj} = A_{ijk}
$$

**应用**:

- 矩阵转置: $(m, n) \to (n, m)$
- 通道顺序转换: $(B, H, W, C) \to (B, C, H, W)$

---

### 3. 广播 (Broadcasting)

**定义**:

自动扩展张量的维度以匹配运算。

**规则**:

1. 如果两个张量维数不同，在较小的张量前面补1
2. 如果某维度大小为1，则沿该维度复制

**示例**:

```python
A: (3, 1)
B: (1, 4)
A + B: (3, 4)  # 广播
```

**应用**:

- 偏置加法: $(B, N) + (N,) \to (B, N)$
- 批量归一化

---

### 4. 张量切片

**切片操作**:

提取张量的子集：

$$
B = A[i_1:i_2, j_1:j_2, :]
$$

**应用**:

- 提取批量子集
- 提取特征子集
- 窗口操作

---

## 💻 Python实现

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Einstein求和约定示例
def einstein_examples():
    """Einstein求和约定示例"""
    print("=== Einstein求和约定 ===\n")
    
    # 向量内积
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # 传统方法
    dot_traditional = np.sum(a * b)
    # Einstein约定 (numpy)
    dot_einstein = np.einsum('i,i->', a, b)
    
    print(f"向量内积:")
    print(f"  传统: {dot_traditional}")
    print(f"  Einstein: {dot_einstein}\n")
    
    # 矩阵-向量乘法
    A = np.array([[1, 2], [3, 4]])
    x = np.array([1, 2])
    
    # 传统方法
    y_traditional = A @ x
    # Einstein约定
    y_einstein = np.einsum('ij,j->i', A, x)
    
    print(f"矩阵-向量乘法:")
    print(f"  传统: {y_traditional}")
    print(f"  Einstein: {y_einstein}\n")
    
    # 矩阵乘法
    B = np.array([[5, 6], [7, 8]])
    
    # 传统方法
    C_traditional = A @ B
    # Einstein约定
    C_einstein = np.einsum('ik,kj->ij', A, B)
    
    print(f"矩阵乘法:")
    print(f"  传统:\n{C_traditional}")
    print(f"  Einstein:\n{C_einstein}\n")
    
    # 矩阵迹
    trace_traditional = np.trace(A)
    trace_einstein = np.einsum('ii->', A)
    
    print(f"矩阵迹:")
    print(f"  传统: {trace_traditional}")
    print(f"  Einstein: {trace_einstein}\n")
    
    # 外积
    outer_traditional = np.outer(a, b)
    outer_einstein = np.einsum('i,j->ij', a, b)
    
    print(f"外积:")
    print(f"  传统:\n{outer_traditional}")
    print(f"  Einstein:\n{outer_einstein}\n")


# 2. 张量运算
def tensor_operations():
    """张量基本运算"""
    print("=== 张量运算 ===\n")
    
    # 创建张量
    T = np.random.randn(2, 3, 4)
    print(f"张量形状: {T.shape}")
    print(f"张量秩: {T.ndim}\n")
    
    # 张量缩并
    # 对第二个维度求和
    T_contract = np.einsum('ijk->ik', T)
    print(f"缩并后形状: {T_contract.shape}\n")
    
    # 张量转置
    T_transpose = np.transpose(T, (2, 0, 1))
    print(f"转置后形状: {T_transpose.shape}\n")
    
    # 张量重塑
    T_reshape = T.reshape(6, 4)
    print(f"重塑后形状: {T_reshape.shape}\n")


# 3. 批量矩阵乘法
def batch_matrix_multiply():
    """批量矩阵乘法"""
    print("=== 批量矩阵乘法 ===\n")
    
    # 批量大小为4，矩阵大小为3x2和2x5
    A = np.random.randn(4, 3, 2)
    B = np.random.randn(4, 2, 5)
    
    # 方法1: 循环
    C_loop = np.zeros((4, 3, 5))
    for i in range(4):
        C_loop[i] = A[i] @ B[i]
    
    # 方法2: Einstein约定
    C_einstein = np.einsum('bij,bjk->bik', A, B)
    
    # 方法3: PyTorch bmm
    A_torch = torch.from_numpy(A)
    B_torch = torch.from_numpy(B)
    C_torch = torch.bmm(A_torch, B_torch).numpy()
    
    print(f"结果形状: {C_einstein.shape}")
    print(f"方法一致性: {np.allclose(C_loop, C_einstein, C_torch)}\n")


# 4. 注意力机制
def attention_mechanism():
    """注意力机制实现"""
    print("=== 注意力机制 ===\n")
    
    # 参数
    batch_size = 2
    seq_len = 4
    d_model = 8
    
    # Q, K, V
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    # 计算注意力分数
    # scores = Q @ K^T / sqrt(d_model)
    scores = np.einsum('bik,bjk->bij', Q, K) / np.sqrt(d_model)
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 加权求和
    # output = attention_weights @ V
    output = np.einsum('bij,bjk->bik', attention_weights, V)
    
    print(f"Q形状: {Q.shape}")
    print(f"K形状: {K.shape}")
    print(f"V形状: {V.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"输出形状: {output.shape}\n")


# 5. 张量分解 - CP分解
def cp_decomposition(T, rank):
    """CP分解 (简化版，使用ALS算法)"""
    I, J, K = T.shape
    
    # 初始化因子矩阵
    A = np.random.randn(I, rank)
    B = np.random.randn(J, rank)
    C = np.random.randn(K, rank)
    
    # ALS迭代
    n_iter = 10
    for _ in range(n_iter):
        # 更新A
        V = np.einsum('jr,kr->jkr', B, C)
        V_flat = V.reshape(J*K, rank)
        T_flat = T.reshape(I, J*K)
        A = T_flat @ V_flat @ np.linalg.inv(V_flat.T @ V_flat + 1e-8*np.eye(rank))
        
        # 更新B (类似)
        V = np.einsum('ir,kr->ikr', A, C)
        V_flat = V.reshape(I*K, rank)
        T_flat = T.transpose(1, 0, 2).reshape(J, I*K)
        B = T_flat @ V_flat @ np.linalg.inv(V_flat.T @ V_flat + 1e-8*np.eye(rank))
        
        # 更新C (类似)
        V = np.einsum('ir,jr->ijr', A, B)
        V_flat = V.reshape(I*J, rank)
        T_flat = T.transpose(2, 0, 1).reshape(K, I*J)
        C = T_flat @ V_flat @ np.linalg.inv(V_flat.T @ V_flat + 1e-8*np.eye(rank))
    
    # 重构张量
    T_reconstructed = np.einsum('ir,jr,kr->ijk', A, B, C)
    
    return A, B, C, T_reconstructed


def test_cp_decomposition():
    """测试CP分解"""
    print("=== CP分解 ===\n")
    
    # 创建低秩张量
    rank = 3
    I, J, K = 10, 12, 8
    
    A_true = np.random.randn(I, rank)
    B_true = np.random.randn(J, rank)
    C_true = np.random.randn(K, rank)
    
    T = np.einsum('ir,jr,kr->ijk', A_true, B_true, C_true)
    
    # CP分解
    A, B, C, T_reconstructed = cp_decomposition(T, rank)
    
    # 计算误差
    error = np.linalg.norm(T - T_reconstructed) / np.linalg.norm(T)
    
    print(f"原始张量形状: {T.shape}")
    print(f"分解秩: {rank}")
    print(f"重构误差: {error:.6f}\n")


# 6. 广播示例
def broadcasting_examples():
    """广播示例"""
    print("=== 广播 ===\n")
    
    # 示例1: 向量加到矩阵每一行
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.array([10, 20, 30])
    
    C = A + b  # 广播
    print(f"矩阵 + 向量 (广播):")
    print(f"A形状: {A.shape}")
    print(f"b形状: {b.shape}")
    print(f"C形状: {C.shape}")
    print(f"C:\n{C}\n")
    
    # 示例2: 批量归一化
    X = np.random.randn(32, 10)  # (batch, features)
    
    # 计算均值和标准差
    mean = np.mean(X, axis=0, keepdims=True)  # (1, 10)
    std = np.std(X, axis=0, keepdims=True)    # (1, 10)
    
    # 归一化 (广播)
    X_normalized = (X - mean) / (std + 1e-8)
    
    print(f"批量归一化:")
    print(f"X形状: {X.shape}")
    print(f"mean形状: {mean.shape}")
    print(f"std形状: {std.shape}")
    print(f"X_normalized形状: {X_normalized.shape}\n")


# 7. 可视化3D张量
def visualize_3d_tensor():
    """可视化3D张量"""
    # 创建简单的3D张量
    T = np.zeros((5, 5, 5))
    T[2, 2, 2] = 1  # 中心点
    T[1:4, 1:4, 1:4] = 0.5  # 内部立方体
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 找到非零元素
    x, y, z = np.where(T > 0)
    colors = T[x, y, z]
    
    # 绘制
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='k')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Tensor Visualization')
    
    plt.colorbar(scatter, ax=ax, label='Value')
    # plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("张量运算与Einstein求和约定示例")
    print("=" * 60 + "\n")
    
    einstein_examples()
    tensor_operations()
    batch_matrix_multiply()
    attention_mechanism()
    test_cp_decomposition()
    broadcasting_examples()
    
    print("\n可视化3D张量...")
    visualize_3d_tensor()
    
    print("\n所有示例完成！")
```

---

## 📚 练习题

### 练习1：Einstein求和约定

使用Einstein求和约定表示以下运算：

1. 向量外积：$\mathbf{a} \otimes \mathbf{b}$
2. 矩阵Frobenius范数：$\|A\|_F$
3. 批量矩阵乘法

### 练习2：张量缩并

给定3阶张量 $T \in \mathbb{R}^{3 \times 4 \times 5}$，计算：

1. 对第一个索引缩并
2. 对第二个索引缩并
3. 完全缩并（所有索引）

### 练习3：卷积运算

使用Einstein约定表示2D卷积运算，包括：

1. 单通道卷积
2. 多通道卷积
3. 批量卷积

### 练习4：注意力机制

实现多头注意力机制，使用Einstein约定表示所有运算。

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **MIT** | 18.065 - Matrix Methods in Data Analysis |
| **Stanford** | CS231n - Convolutional Neural Networks |
| **CMU** | 10-708 - Probabilistic Graphical Models |
| **UC Berkeley** | CS189 - Introduction to Machine Learning |

---

## 📖 参考文献

1. **Kolda & Bader (2009)**. *Tensor Decompositions and Applications*. SIAM Review.

2. **Cichocki et al. (2015)**. *Tensor Decompositions for Signal Processing Applications*. IEEE Signal Processing Magazine.

3. **Goodfellow et al. (2016)**. *Deep Learning*. MIT Press. (Chapter 2)

4. **Novikov et al. (2015)**. *Tensorizing Neural Networks*. NeurIPS.

5. **Paszke et al. (2019)**. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.

---

*最后更新：2025年10月*-
