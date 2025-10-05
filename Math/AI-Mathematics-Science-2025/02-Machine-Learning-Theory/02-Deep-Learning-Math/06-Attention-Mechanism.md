# 注意力机制 (Attention Mechanism) 数学原理

> **Attention Mechanism: Mathematics and Theory**
>
> Transformer与现代LLM的核心技术

---

## 目录

- [注意力机制 (Attention Mechanism) 数学原理](#注意力机制-attention-mechanism-数学原理)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 问题动机](#-问题动机)
    - [1. 序列建模的挑战](#1-序列建模的挑战)
    - [2. 注意力的直觉](#2-注意力的直觉)
  - [📊 Scaled Dot-Product Attention](#-scaled-dot-product-attention)
    - [1. 数学定义](#1-数学定义)
    - [2. 缩放因子的作用](#2-缩放因子的作用)
    - [3. Softmax的作用](#3-softmax的作用)
  - [🔬 Multi-Head Attention](#-multi-head-attention)
    - [1. 核心思想](#1-核心思想)
    - [2. 数学形式化](#2-数学形式化)
    - [3. 为什么多头有效](#3-为什么多头有效)
  - [💻 Self-Attention vs Cross-Attention](#-self-attention-vs-cross-attention)
    - [1. Self-Attention](#1-self-attention)
    - [2. Cross-Attention](#2-cross-attention)
    - [3. 应用场景](#3-应用场景)
  - [🎨 Python实现](#-python实现)
  - [📚 理论分析](#-理论分析)
    - [1. 注意力的表示能力](#1-注意力的表示能力)
    - [2. 计算复杂度](#2-计算复杂度)
    - [3. 长序列问题](#3-长序列问题)
  - [🔧 注意力变体](#-注意力变体)
    - [1. Sparse Attention](#1-sparse-attention)
    - [2. Linear Attention](#2-linear-attention)
    - [3. Flash Attention](#3-flash-attention)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**注意力机制**允许模型**动态关注**输入的不同部分。

**核心流程**：

```text
Query (查询) + Key (键) + Value (值)
        ↓
  计算相似度 (Q·K^T)
        ↓
   归一化 (Softmax)
        ↓
  加权求和 (Attention·V)
        ↓
      输出
```

**关键优势**：

- 捕获长距离依赖
- 并行计算（相比RNN）
- 可解释性（注意力权重）

---

## 🎯 问题动机

### 1. 序列建模的挑战

**RNN的问题**：

- **顺序依赖**：必须逐步处理，无法并行
- **长距离依赖**：梯度消失/爆炸
- **固定容量**：隐状态维度限制信息量

**示例**：

```text
"The cat, which was very hungry, ate the fish."
```

要理解"ate"，需要关注"cat"（主语），但中间隔了很多词。

---

### 2. 注意力的直觉

**人类阅读**：

- 不是均匀关注所有词
- 根据任务动态调整注意力
- 可以"跳跃"关注相关信息

**注意力机制**：

- 为每个输出位置计算对所有输入的注意力权重
- 权重反映相关性
- 输出是输入的加权和

---

## 📊 Scaled Dot-Product Attention

### 1. 数学定义

**定义 1.1 (Scaled Dot-Product Attention)**:

给定查询 $Q \in \mathbb{R}^{n \times d_k}$，键 $K \in \mathbb{R}^{m \times d_k}$，值 $V \in \mathbb{R}^{m \times d_v}$：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**符号说明**：

- $n$：查询序列长度
- $m$：键/值序列长度
- $d_k$：键/查询维度
- $d_v$：值维度

**步骤分解**：

1. **计算相似度**：$S = QK^T \in \mathbb{R}^{n \times m}$
2. **缩放**：$S' = S / \sqrt{d_k}$
3. **归一化**：$A = \text{softmax}(S') \in \mathbb{R}^{n \times m}$
4. **加权求和**：$\text{Output} = AV \in \mathbb{R}^{n \times d_v}$

---

### 2. 缩放因子的作用

**为什么除以 $\sqrt{d_k}$？**

**定理 2.1 (缩放必要性, Vaswani et al. 2017)**:

假设 $Q, K$ 的元素独立同分布，均值0，方差1，则：

$$
\mathbb{E}[QK^T] = 0, \quad \text{Var}[QK^T] = d_k
$$

**问题**：当 $d_k$ 很大时，点积方差很大。

**后果**：

- Softmax输入进入饱和区
- 梯度接近0
- 训练困难

**解决方案**：除以 $\sqrt{d_k}$，使方差归一化为1。

$$
\text{Var}\left[\frac{QK^T}{\sqrt{d_k}}\right] = 1
$$

---

### 3. Softmax的作用

**Softmax定义**：

$$
\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{m} \exp(z_j)}
$$

**性质**：

1. **归一化**：$\sum_{j=1}^{m} A_{ij} = 1$（每行和为1）
2. **非负**：$A_{ij} \geq 0$
3. **可微**：梯度存在

**意义**：

- 将相似度转换为概率分布
- 高相似度位置获得更大权重
- 低相似度位置权重接近0

---

## 🔬 Multi-Head Attention

### 1. 核心思想

**单头注意力的局限**：

- 只能学习一种相似度度量
- 难以同时捕获多种关系

**多头注意力**：

- 并行运行多个注意力"头"
- 每个头学习不同的表示子空间
- 拼接所有头的输出

---

### 2. 数学形式化

**定义 2.1 (Multi-Head Attention)**:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

其中每个头：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**参数**：

- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$：查询投影
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$：键投影
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$：值投影
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$：输出投影

**典型设置**：

- $h = 8$（头数）
- $d_k = d_v = d_{\text{model}} / h = 64$（Transformer中 $d_{\text{model}} = 512$）

---

### 3. 为什么多头有效

**定理 3.1 (表示能力)**:

多头注意力可以同时关注不同的表示子空间。

**直觉**：

- **Head 1**：可能关注句法关系（主谓宾）
- **Head 2**：可能关注语义关系（同义词）
- **Head 3**：可能关注位置关系（相邻词）

**实验证据**：

- 不同头学习到不同的注意力模式
- 某些头专注于局部，某些头专注于全局
- 多头优于单头（实验验证）

---

## 💻 Self-Attention vs Cross-Attention

### 1. Self-Attention

**定义**：$Q, K, V$ 来自同一序列。

$$
\text{SelfAttention}(X) = \text{Attention}(XW^Q, XW^K, XW^V)
$$

**应用**：

- Transformer编码器
- BERT
- GPT

**作用**：

- 每个位置关注序列中的所有位置
- 捕获上下文信息
- 学习词之间的关系

---

### 2. Cross-Attention

**定义**：$Q$ 来自一个序列，$K, V$ 来自另一个序列。

$$
\text{CrossAttention}(X, Y) = \text{Attention}(XW^Q, YW^K, YW^V)
$$

**应用**：

- Transformer解码器
- 机器翻译
- 图像描述生成

**作用**：

- 查询序列关注源序列
- 对齐不同模态
- 信息融合

---

### 3. 应用场景

| 类型 | Q来源 | K/V来源 | 应用 |
|------|-------|---------|------|
| **Self-Attention** | 同一序列 | 同一序列 | BERT, GPT |
| **Cross-Attention** | 目标序列 | 源序列 | 翻译, VQA |
| **Masked Self-Attention** | 同一序列 | 同一序列（掩码） | GPT解码 |

---

## 🎨 Python实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch, n_heads, seq_len_q, d_k)
            K: (batch, n_heads, seq_len_k, d_k)
            V: (batch, n_heads, seq_len_v, d_v)
            mask: (batch, 1, seq_len_q, seq_len_k) or None
        
        Returns:
            output: (batch, n_heads, seq_len_q, d_v)
            attention_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        d_k = Q.size(-1)
        
        # 1. 计算相似度: (batch, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 应用掩码 (可选)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 4. 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch, seq_len_q, d_model)
            K: (batch, seq_len_k, d_model)
            V: (batch, seq_len_v, d_model)
            mask: (batch, seq_len_q, seq_len_k) or None
        
        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = Q.size(0)
        residual = Q
        
        # 1. 线性投影并分割成多头
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # 2. 调整mask维度
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len_q, seq_len_k)
        
        # 3. 应用注意力
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # 4. 拼接多头
        # (batch, n_heads, seq_len_q, d_v) -> (batch, seq_len_q, n_heads, d_v) -> (batch, seq_len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 输出投影
        output = self.W_O(output)
        output = self.dropout(output)
        
        # 6. 残差连接 + Layer Norm
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


# 使用示例
if __name__ == "__main__":
    # 参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建模型
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    
    # 输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Self-Attention
    output, attn_weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # 可视化注意力权重
    import matplotlib.pyplot as plt
    
    # 取第一个样本的第一个头
    attn = attn_weights[0, 0].detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Weights (Head 1)')
    plt.tight_layout()
    # plt.show()


# Cross-Attention示例
class CrossAttentionLayer(nn.Module):
    """Cross-Attention层 (用于Transformer解码器)"""
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, decoder_input, encoder_output, mask=None):
        """
        Args:
            decoder_input: (batch, seq_len_dec, d_model) - Query
            encoder_output: (batch, seq_len_enc, d_model) - Key & Value
            mask: (batch, seq_len_dec, seq_len_enc) or None
        
        Returns:
            output: (batch, seq_len_dec, d_model)
        """
        output, attn_weights = self.cross_attn(
            Q=decoder_input,
            K=encoder_output,
            V=encoder_output,
            mask=mask
        )
        return output, attn_weights


# Masked Self-Attention (用于GPT)
def create_causal_mask(seq_len):
    """创建因果掩码 (下三角矩阵)"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # (seq_len, seq_len)


# 示例：GPT风格的Masked Self-Attention
if __name__ == "__main__":
    seq_len = 5
    d_model = 512
    
    mha = MultiHeadAttention(d_model=d_model, n_heads=8)
    x = torch.randn(1, seq_len, d_model)
    
    # 创建因果掩码
    causal_mask = create_causal_mask(seq_len).unsqueeze(0)  # (1, seq_len, seq_len)
    
    # Masked Self-Attention
    output, attn_weights = mha(x, x, x, mask=causal_mask)
    
    print(f"\nCausal Mask:\n{causal_mask[0]}")
    print(f"\nAttention weights (with mask):\n{attn_weights[0, 0].detach()}")
```

---

## 📚 理论分析

### 1. 注意力的表示能力

**定理 1.1 (Universal Approximation)**:

Transformer可以近似任意序列到序列的函数（在适当条件下）。

**证明要点**：

- 注意力可以实现任意的加权平均
- 多层Transformer可以组合复杂操作
- FFN提供非线性变换

**实践意义**：

- 理论上可以学习任意序列模式
- 实际受限于数据和优化

---

### 2. 计算复杂度

**Self-Attention复杂度**：

| 操作 | 复杂度 |
|------|--------|
| **计算 $QK^T$** | $O(n^2 d)$ |
| **Softmax** | $O(n^2)$ |
| **加权求和** | $O(n^2 d)$ |
| **总计** | $O(n^2 d)$ |

其中 $n$ 是序列长度，$d$ 是维度。

**对比RNN**：

- RNN：$O(nd^2)$（顺序计算）
- Attention：$O(n^2d)$（并行计算）

**问题**：序列长度 $n$ 很大时，$n^2$ 项成为瓶颈。

---

### 3. 长序列问题

**挑战**：

- 内存：存储 $n \times n$ 注意力矩阵
- 计算：$O(n^2)$ 复杂度

**示例**：

- $n = 1024$：1M注意力权重
- $n = 4096$：16M注意力权重
- $n = 100k$：10B注意力权重（不可行！）

**解决方案**：见下节"注意力变体"

---

## 🔧 注意力变体

### 1. Sparse Attention

**核心思想**：只计算部分注意力权重。

**Longformer (Beltagy et al., 2020)**：

- **局部注意力**：每个token只关注窗口内的token
- **全局注意力**：特殊token关注所有token
- **复杂度**：$O(n \cdot w)$，其中 $w$ 是窗口大小

**BigBird (Zaheer et al., 2020)**：

- **随机注意力** + **窗口注意力** + **全局注意力**
- **理论保证**：仍是通用近似器

---

### 2. Linear Attention

**核心思想**：通过核技巧降低复杂度。

**Linformer (Wang et al., 2020)**：

- 将 $K, V$ 投影到低维
- 复杂度：$O(nk)$，其中 $k \ll n$

**Performer (Choromanski et al., 2021)**：

- 用随机特征近似Softmax
- 复杂度：$O(nd)$（线性！）

**公式**：

$$
\text{Attention}(Q, K, V) \approx \phi(Q) (\phi(K)^T V)
$$

其中 $\phi$ 是特征映射。

---

### 3. Flash Attention

**核心思想**：优化内存访问模式。

**Flash Attention (Dao et al., 2022)**：

- 不显式存储 $n \times n$ 注意力矩阵
- 分块计算，减少HBM访问
- **加速**：2-4倍
- **内存**：线性而非二次

**意义**：

- 允许更长序列
- 更高效训练
- 成为新标准

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS224N Natural Language Processing |
| **Stanford** | CS25 Transformers United |
| **MIT** | 6.S191 Introduction to Deep Learning |
| **CMU** | 11-747 Neural Networks for NLP |

---

## 📖 参考文献

1. **Vaswani et al. (2017)**. "Attention Is All You Need". *NeurIPS*.

2. **Bahdanau et al. (2015)**. "Neural Machine Translation by Jointly Learning to Align and Translate". *ICLR*.

3. **Beltagy et al. (2020)**. "Longformer: The Long-Document Transformer". *arXiv*.

4. **Zaheer et al. (2020)**. "Big Bird: Transformers for Longer Sequences". *NeurIPS*.

5. **Choromanski et al. (2021)**. "Rethinking Attention with Performers". *ICLR*.

6. **Dao et al. (2022)**. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". *NeurIPS*.

---

*最后更新：2025年10月*-
