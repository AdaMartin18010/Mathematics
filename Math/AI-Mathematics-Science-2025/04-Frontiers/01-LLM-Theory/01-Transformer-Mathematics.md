# Transformer的数学原理 (Mathematics of Transformers)

> **Attention Is All You Need**  
> 大语言模型背后的数学基础

---

## 📋 目录

- [Transformer的数学原理 (Mathematics of Transformers)](#transformer的数学原理-mathematics-of-transformers)
  - [📋 目录](#-目录)
  - [🎯 Transformer架构概览](#-transformer架构概览)
  - [📐 核心机制数学分析](#-核心机制数学分析)
    - [1. Self-Attention机制](#1-self-attention机制)
    - [2. Multi-Head Attention](#2-multi-head-attention)
    - [3. Position Encoding](#3-position-encoding)
  - [🔍 理论分析](#-理论分析)
    - [1. Attention的表达能力](#1-attention的表达能力)
    - [2. Transformer的通用逼近性质](#2-transformer的通用逼近性质)
    - [3. 计算复杂度分析](#3-计算复杂度分析)
  - [🧮 优化与训练动力学](#-优化与训练动力学)
    - [1. Layer Normalization的作用](#1-layer-normalization的作用)
    - [2. 梯度流分析](#2-梯度流分析)
    - [3. Warmup与学习率调度](#3-warmup与学习率调度)
  - [💻 PyTorch实现](#-pytorch实现)
    - [Self-Attention From Scratch](#self-attention-from-scratch)
    - [完整Transformer块](#完整transformer块)
  - [🔬 前沿研究 (2025)](#-前沿研究-2025)
    - [1. Sparse Attention变体](#1-sparse-attention变体)
    - [2. 线性Attention近似](#2-线性attention近似)
    - [3. State Space Models (Mamba)](#3-state-space-models-mamba)
  - [🤖 在LLM中的应用](#-在llm中的应用)
    - [1. GPT系列](#1-gpt系列)
    - [2. In-Context Learning的数学解释](#2-in-context-learning的数学解释)
    - [3. Scaling Laws](#3-scaling-laws)
  - [📚 相关资源](#-相关资源)
    - [开创性论文](#开创性论文)
    - [理论分析论文](#理论分析论文)
    - [2025年最新研究](#2025年最新研究)
  - [🎓 对标课程](#-对标课程)
  - [💡 练习题](#-练习题)
    - [基础题](#基础题)
    - [进阶题](#进阶题)
    - [挑战题](#挑战题)

---

## 🎯 Transformer架构概览

**Transformer** (Vaswani et al., 2017) 是现代大语言模型的基础架构。

**核心组件**:

```text
Input Embedding + Positional Encoding
       ↓
┌──────────────────┐
│  Transformer块   │ × N层
│  ├─ Multi-Head   │
│  │  Self-Attention│
│  ├─ Layer Norm   │
│  ├─ Feed-Forward │
│  └─ Layer Norm   │
└──────────────────┘
       ↓
  Output Layer
```

**数学流程**:

$$
\begin{align}
\text{Attention}(Q,K,V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \\
\text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O \\
\text{FFN}(x) &= \max(0, xW_1 + b_1)W_2 + b_2
\end{align}
$$

---

## 📐 核心机制数学分析

### 1. Self-Attention机制

**定义**:

给定输入序列 $X = [x_1, \ldots, x_n] \in \mathbb{R}^{n \times d}$:

$$
\begin{align}
Q &= XW^Q, \quad K = XW^K, \quad V = XW^V \\
\text{Attention}(Q,K,V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\end{align}
$$

其中:

- $W^Q, W^K \in \mathbb{R}^{d \times d_k}$, $W^V \in \mathbb{R}^{d \times d_v}$
- $\sqrt{d_k}$ 是**缩放因子** (防止softmax饱和)

---

**逐步推导**:

**步骤1**: 计算注意力分数

$$
S = QK^T = (XW^Q)(XW^K)^T \in \mathbb{R}^{n \times n}
$$

矩阵 $S$ 的元素:

$$
S_{ij} = q_i^T k_j = \langle W^Q x_i, W^K x_j \rangle
$$

衡量 token $i$ 和 token $j$ 的相关性。

---

**步骤2**: 缩放

$$
S' = \frac{S}{\sqrt{d_k}}
$$

**为什么要缩放?**

假设 $q_i, k_j$ 的分量独立同分布,均值0,方差1:

$$
\mathbb{E}[S_{ij}] = 0, \quad \text{Var}(S_{ij}) = d_k
$$

缩放后方差变为1,防止softmax梯度消失。

---

**步骤3**: Softmax归一化

$$
A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=1}^n \exp(S'_{ik})}
$$

满足: $\sum_{j=1}^n A_{ij} = 1$ (每行是概率分布)

---

**步骤4**: 加权求和

$$
\text{Output}_i = \sum_{j=1}^n A_{ij} v_j
$$

每个输出是**值向量的加权平均**,权重由注意力分数决定。

---

**矩阵形式**:

$$
\text{Output} = A V = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

---

### 2. Multi-Head Attention

**动机**: 单个注意力头只能捕捉一种相关性模式,多头可以并行学习多种模式。

**定义**:

$$
\begin{align}
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
\text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\end{align}
$$

其中:

- $W_i^Q, W_i^K \in \mathbb{R}^{d \times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d}$ (输出投影)
- 通常: $d_k = d_v = d/h$

**参数量**:

$$
\text{Params} = 4d^2 + 4d \quad \text{(忽略bias)}
$$

- $W^Q, W^K, W^V$: $3 \times hd_k \times d = 3d^2$
- $W^O$: $hd_v \times d = d^2$

---

### 3. Position Encoding

**问题**: Self-attention是**置换不变**的,即:

$$
\text{Attention}(\pi(X)) = \pi(\text{Attention}(X))
$$

需要**位置编码**注入序列信息。

---

**正弦位置编码** (原始Transformer):

$$
\begin{align}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{align}
$$

**性质**:

1. 每个维度是不同频率的正弦波
2. 相对位置信息: $PE_{pos+k}$ 可由 $PE_{pos}$ 线性表示

$$
PE_{pos+k} = A_k PE_{pos} + B_k
$$

---

**可学习位置编码** (GPT):

$$
PE_{pos} \in \mathbb{R}^d \quad \text{(可训练参数)}
$$

**优点**: 更灵活  
**缺点**: 无法泛化到训练时未见过的长度

---

**旋转位置编码 (RoPE, 2021)**:

将查询和键旋转一个角度,角度依赖位置:

$$
q_m' = e^{im\theta} q_m, \quad k_n' = e^{in\theta} k_n
$$

满足:

$$
\langle q_m', k_n' \rangle = \text{Re}(e^{i(m-n)\theta} \langle q_m, k_n \rangle)
$$

**仅依赖相对位置** $m - n$!

---

## 🔍 理论分析

### 1. Attention的表达能力

**定理** (Attention as Dictionary Lookup):

Attention层可以精确实现**软字典查找**:

给定键值对 $(k_1, v_1), \ldots, (k_n, v_n)$, 查询 $q$:

$$
\text{Attention}(q, K, V) \approx v_{i^*} \quad \text{其中} \quad i^* = \arg\max_i \langle q, k_i \rangle
$$

当温度 $T \to 0$ (即 $\frac{1}{\sqrt{d_k}} \to \infty$):

$$
\text{softmax}\left(\frac{qK^T}{T}\right) \to \text{one-hot}(i^*)
$$

---

**定理** (Contextual Representation):

Attention层计算的是**上下文化表示**:

$$
h_i = \sum_{j=1}^n w_{ij} v_j, \quad w_{ij} \propto \exp(\text{similarity}(x_i, x_j))
$$

每个token的表示是**所有相关token的信息聚合**。

---

### 2. Transformer的通用逼近性质

**定理** (Yun et al., 2020):

Transformer可以逼近**任意序列到序列映射** (在适当假设下)。

**证明思路**:

1. Attention层可以实现**任意稀疏连接**
2. FFN层是通用函数逼近器 (ReLU网络)
3. 组合起来可以逼近任意计算图

---

**定理** (Turing Completeness):

具有足够深度和宽度的Transformer是**图灵完备**的。

**构造**:

- 用Attention模拟指针操作
- 用FFN实现算术和逻辑运算
- 可以模拟通用图灵机

---

### 3. 计算复杂度分析

**时间复杂度**:

| 操作 | 复杂度 |
| ---- |--------|
| Self-Attention | $O(n^2 d)$ |
| FFN | $O(nd^2)$ |
| **总计** | $O(n^2 d + nd^2)$ |

**瓶颈**: 序列长度 $n$ 较大时, $O(n^2)$ 成为主要瓶颈。

---

**空间复杂度**:

- 存储注意力矩阵: $O(n^2)$
- 激活值: $O(nLd)$ (L是层数)

**KV Cache** (推理优化):

缓存过去的键值对,避免重复计算:

$$
\text{Cache size} = O(Lnd)
$$

---

## 🧮 优化与训练动力学

### 1. Layer Normalization的作用

**定义**:

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中:

- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ (均值)
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$ (方差)

---

**Pre-LN vs Post-LN**:

```text
Post-LN (原始):                Pre-LN (现代):
x → Attention → LN → FFN → LN   x → LN → Attention → LN → FFN
```

**Pre-LN的优势**:

- 更稳定的梯度流
- 可以不用Warmup
- 更深的网络训练更稳定

---

### 2. 梯度流分析

**残差连接的重要性**:

$$
x_{l+1} = x_l + F_l(x_l)
$$

**反向传播**:

$$
\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \left(I + \frac{\partial F_l}{\partial x_l}\right)
$$

**梯度始终包含恒等项** $I$, 避免梯度消失!

---

**定理** (Gradient Flow in Transformers):

在Pre-LN架构中,梯度可以**无阻碍地**从输出流向输入:

$$
\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} + \sum_{l=0}^{L-1} \text{correction terms}
$$

---

### 3. Warmup与学习率调度

**Transformer学习率调度** (原始论文):

$$
\text{lr}(t) = d^{-0.5} \cdot \min(t^{-0.5}, t \cdot \text{warmup}^{-1.5})
$$

**两个阶段**:

1. **Warmup** ($t < \text{warmup}$): 线性增长
2. **Decay**: 按 $t^{-0.5}$ 衰减

---

**为什么需要Warmup?**

**假说1**: 防止Adam的二阶矩估计不准确  
**假说2**: 初期梯度方差大,小学习率更稳定  
**假说3**: 帮助找到更平坦的最小值

---

## 💻 PyTorch实现

### Self-Attention From Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """从零实现Self-Attention"""
    
    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        
        # Query, Key, Value投影
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        
    def forward(self, X, mask=None):
        """
        Args:
            X: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) 或 None
        
        Returns:
            output: (batch, seq_len, d_v)
            attention_weights: (batch, seq_len, seq_len)
        """
        # 步骤1: 计算Q, K, V
        Q = self.W_q(X)  # (batch, seq_len, d_k)
        K = self.W_k(X)  # (batch, seq_len, d_k)
        V = self.W_v(X)  # (batch, seq_len, d_v)
        
        # 步骤2: 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_k)  # 缩放
        
        # 步骤3: 应用mask (可选)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 步骤4: Softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        
        # 步骤5: 加权求和
        output = torch.matmul(attention_weights, V)  # (batch, seq_len, d_v)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 所有头的Q,K,V投影 (合并成一个大矩阵)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, X, mask=None):
        batch_size, seq_len, d_model = X.shape
        
        # 计算Q,K,V并分割成多个头
        Q = self.W_q(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, d_k)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        # Shape: (batch, num_heads, seq_len, d_k)
        
        # 拼接所有头
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        
        # 输出投影
        output = self.W_o(attention_output)
        
        return output, attention_weights


# 测试
if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    X = torch.randn(batch_size, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = mha(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
```

---

### 完整Transformer块

```python
class TransformerBlock(nn.Module):
    """完整的Transformer块 (Pre-LN架构)"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-LN架构
        
        # Multi-Head Attention子层
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, mask)
        x = x + self.dropout1(attn_output)  # 残差连接
        
        # FFN子层
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)  # 残差连接
        
        return x


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
```

---

## 🔬 前沿研究 (2025)

### 1. Sparse Attention变体

**动机**: 降低 $O(n^2)$ 复杂度

**Longformer** (Beltagy et al., 2020):

- **局部窗口注意力**: 每个token只关注周围 $w$ 个token
- **全局注意力**: 少数特殊token关注所有位置
- 复杂度: $O(nw)$

---

**BigBird** (Zaheer et al., 2020):

结合三种注意力模式:

1. 随机注意力
2. 窗口注意力
3. 全局注意力

**定理**: 这三种模式的组合保持图灵完备性。

---

### 2. 线性Attention近似

**Linformer** (Wang et al., 2020):

将 $K, V$ 投影到低维:

$$
K' = EK, \quad V' = FV
$$

其中 $E, F \in \mathbb{R}^{k \times n}$, $k \ll n$

复杂度: $O(nk)$

---

**Performer** (Choromanski et al., 2021):

使用随机特征近似softmax:

$$
\text{softmax}(qk^T) \approx \phi(q) \phi(k)^T
$$

其中 $\phi$ 是随机特征映射。

复杂度: $O(n)$ (线性!)

---

### 3. State Space Models (Mamba)

**Mamba** (Gu & Dao, 2023):

用状态空间模型替代Attention:

$$
\begin{align}
h_t &= A h_{t-1} + B x_t \\
y_t &= C h_t + D x_t
\end{align}
$$

**优势**:

- $O(n)$ 复杂度
- 更长的上下文
- 更快的推理

**挑战**: 可解释性不如Attention

---

## 🤖 在LLM中的应用

### 1. GPT系列

**GPT-1** (2018): 12层, 117M参数  
**GPT-2** (2019): 48层, 1.5B参数  
**GPT-3** (2020): 96层, 175B参数  
**GPT-4** (2023): ~1.76T参数 (混合专家)

**架构演化**:

- 更深的网络 (96层 → 120层)
- 更大的模型 (175B → 1.76T)
- 混合专家 (MoE)
- 更长的上下文 (2k → 32k → 128k)

---

### 2. In-Context Learning的数学解释

**现象**: LLM可以从few-shot示例中学习,无需梯度更新。

**理论解释1** (Xie et al., 2022):

Transformer在隐式地实现**梯度下降**:

$$
W_{t+1} = W_t - \eta \nabla_W L(x_t, y_t; W_t)
$$

每个Attention层更新"隐式权重"。

---

**理论解释2** (Von Oswald et al., 2023):

Transformer可以模拟**岭回归**等算法:

给定示例 $(x_1, y_1), \ldots, (x_k, y_k)$, 查询 $x_{test}$:

$$
\hat{y}_{test} = (X^TX + \lambda I)^{-1} X^T y \cdot x_{test}
$$

可以用多层Attention精确实现!

---

### 3. Scaling Laws

**Kaplan et al. (2020)** 发现:

$$
L(N) = \left(\frac{N_c}{N}\right)^\alpha
$$

其中:

- $L$: 测试损失
- $N$: 模型参数量
- $N_c, \alpha$: 常数 ($\alpha \approx 0.076$)

**Chinchilla Scaling** (Hoffmann et al., 2022):

最优训练应该平衡参数量和数据量:

$$
N_{\text{optimal}} \propto D_{\text{optimal}}
$$

---

## 📚 相关资源

### 开创性论文

1. **Vaswani et al. (2017)**  
   "Attention Is All You Need"  
   *NeurIPS 2017*  
   → Transformer的开创性论文

2. **Devlin et al. (2019)**  
   "BERT: Pre-training of Deep Bidirectional Transformers"  
   → 双向Transformer

3. **Radford et al. (2019)**  
   "Language Models are Unsupervised Multitask Learners" (GPT-2)  
   → 自回归语言模型

---

### 理论分析论文

1. **Yun et al. (2020)**  
   "Are Transformers Universal Approximators of Sequence-to-Sequence Functions?"  
   → 通用逼近性质

2. **Pérez et al. (2021)**  
   "Attention is Turing Complete"  
   → 图灵完备性

3. **Xie et al. (2022)**  
   "An Explanation of In-context Learning as Implicit Bayesian Inference"  
   → In-context learning理论

---

### 2025年最新研究

1. **Gu & Dao (2023)**  
   "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"  
   → 线性复杂度替代

2. **Chen et al. (2024)**  
   "Training-Free Long-Context Scaling of Large Language Models"  
   → 长上下文扩展

3. **Liu et al. (2025)**  
   "Mathematical Foundations of Transformer Scaling Laws" (arXiv)  
   → Scaling laws的理论基础

---

## 🎓 对标课程

| 大学 | 课程 | 相关内容 |
| ---- |------| ---- |
| Stanford | CS224N | Transformer架构 (Week 5-6) |
| Stanford | CS324 | LLM理论 (全课程) |
| MIT | 6.S898 | 深度学习 (Attention机制) |
| CMU | 11-747 | Neural NLP (Transformer) |

---

## 💡 练习题

### 基础题

**1. 注意力矩阵分析**:

给定序列 "I love AI", 画出可能的注意力矩阵热图,并解释:

- "love" 应该关注哪些词?
- 为什么需要多头注意力?

---

**2. 计算量分析**:

计算一个Transformer块的FLOPs:

- 输入: $(n, d) = (512, 768)$
- 8个注意力头
- FFN隐藏层: $4d$

---

### 进阶题

**3. 实现因果Mask**:

实现GPT风格的因果attention mask:

```python
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    创建因果mask,使得位置i只能看到 i之前的位置
    
    返回: (seq_len, seq_len) 的布尔矩阵
    """
    # TODO: 实现
    pass
```

---

**4. 位置编码可视化**:

绘制正弦位置编码的前8个维度,分析其性质:

- 不同维度的频率如何变化?
- 如何编码相对位置信息?

---

### 挑战题

**5. 证明Attention的通用性**:

证明: 单层Attention + FFN可以实现任意**稀疏连接**的函数。

提示: 构造性证明,展示如何设置权重矩阵。

---

**6. Scaling Law推导**:

推导为什么测试损失 $L(N) \propto N^{-\alpha}$。

考虑:

- 参数量 $N$ 与有效假设类大小的关系
- 统计学习理论的泛化界

---

**📌 下一主题**: [In-Context Learning理论](./02-In-Context-Learning-Theory.md)

**🔙 返回**: [LLM理论](../README.md) | [前沿研究](../../README.md)
