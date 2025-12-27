# 循环神经网络 (RNN/LSTM) 数学原理

> **Recurrent Neural Networks: Mathematics and Theory**
>
> 序列建模的数学基础

---

## 目录

- [循环神经网络 (RNN/LSTM) 数学原理](#循环神经网络-rnnlstm-数学原理)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 序列建模问题](#-序列建模问题)
    - [1. 为什么需要RNN](#1-为什么需要rnn)
    - [2. RNN的优势](#2-rnn的优势)
  - [📊 基础RNN](#-基础rnn)
    - [1. 数学定义](#1-数学定义)
    - [2. 展开图](#2-展开图)
    - [3. 前向传播](#3-前向传播)
  - [🔬 BPTT (反向传播)](#-bptt-反向传播)
    - [1. 时间反向传播](#1-时间反向传播)
    - [2. 梯度计算](#2-梯度计算)
    - [3. 梯度消失/爆炸](#3-梯度消失爆炸)
  - [💻 LSTM (长短期记忆网络)](#-lstm-长短期记忆网络)
    - [1. 动机](#1-动机)
    - [2. LSTM架构](#2-lstm架构)
    - [3. 门控机制](#3-门控机制)
  - [🎨 GRU (门控循环单元)](#-gru-门控循环单元)
    - [1. GRU架构](#1-gru架构)
    - [2. GRU vs LSTM](#2-gru-vs-lstm)
  - [📐 双向RNN](#-双向rnn)
    - [1. 动机1](#1-动机1)
    - [2. 数学定义](#2-数学定义)
  - [🔧 梯度裁剪](#-梯度裁剪)
    - [1. 梯度爆炸问题](#1-梯度爆炸问题)
    - [2. 梯度裁剪方法](#2-梯度裁剪方法)
  - [💡 Python实现](#-python实现)
  - [📚 应用场景](#-应用场景)
    - [1. 语言模型](#1-语言模型)
    - [2. 机器翻译](#2-机器翻译)
    - [3. 时间序列预测](#3-时间序列预测)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**循环神经网络 (RNN)** 通过**循环连接**处理序列数据。

**核心原理**：

```text
输入序列: x₁, x₂, x₃, ..., xₜ
    ↓
隐状态更新: hₜ = f(hₜ₋₁, xₜ)
    ↓
输出: yₜ = g(hₜ)
```

**关键概念**：

- **隐状态**：记忆过去信息
- **参数共享**：所有时间步共享权重
- **时间展开**：循环变为前馈

---

## 🎯 序列建模问题

### 1. 为什么需要RNN

**序列数据的特点**：

- **时间依赖**：当前输出依赖历史输入
- **变长输入**：序列长度不固定
- **顺序重要**：改变顺序改变含义

**示例**：

```text
"The cat sat on the mat"
vs
"The mat sat on the cat"
```

**传统神经网络的问题**：

- 固定输入长度
- 无法捕获时间依赖
- 参数数量随序列长度增长

---

### 2. RNN的优势

**参数共享**：

- 所有时间步共享权重
- 参数数量与序列长度无关

**记忆能力**：

- 隐状态存储历史信息
- 理论上可以捕获任意长度依赖

**灵活性**：

- 多对一：情感分析
- 一对多：图像描述
- 多对多：机器翻译

---

## 📊 基础RNN

### 1. 数学定义

**定义 1.1 (Vanilla RNN)**:

给定输入序列 $\mathbf{x} = (x_1, x_2, \ldots, x_T)$：

$$
\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

$$
\mathbf{y}_t = W_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

**参数**：

- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$：隐状态到隐状态
- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$：输入到隐状态
- $W_{hy} \in \mathbb{R}^{d_y \times d_h}$：隐状态到输出
- $\mathbf{h}_0$：初始隐状态（通常为0）

---

### 2. 展开图

**循环视图**：

```text
    ┌───┐
    │ h │←─┐
    └───┘  │
      ↑    │
      x    └─ (循环连接)
```

**展开视图**：

```text
h₀ → h₁ → h₂ → h₃ → ... → hₜ
     ↑    ↑    ↑         ↑
     x₁   x₂   x₃        xₜ
     ↓    ↓    ↓         ↓
     y₁   y₂   y₃        yₜ
```

**关键**：展开后变为深度前馈网络！

---

### 3. 前向传播

**算法 1.1 (RNN Forward Pass)**:

**输入**：序列 $\mathbf{x} = (x_1, \ldots, x_T)$

**步骤**：

1. 初始化 $\mathbf{h}_0 = \mathbf{0}$
2. **for** $t = 1$ **to** $T$:
   - $\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h)$
   - $\mathbf{y}_t = W_{hy} \mathbf{h}_t + \mathbf{b}_y$
3. **return** $(\mathbf{h}_1, \ldots, \mathbf{h}_T)$, $(\mathbf{y}_1, \ldots, \mathbf{y}_T)$

**复杂度**：$O(T \cdot d_h^2)$

---

## 🔬 BPTT (反向传播)

### 1. 时间反向传播

**BPTT (Backpropagation Through Time)**：

- 将展开的RNN视为深度网络
- 从 $t = T$ 反向传播到 $t = 1$
- 累积所有时间步的梯度

**关键**：梯度需要通过时间回传！

---

### 2. 梯度计算

**损失函数**：

$$
\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t(\mathbf{y}_t, \hat{\mathbf{y}}_t)
$$

**隐状态梯度**：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}
$$

**权重梯度**：

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial W_{hh}}
$$

---

### 3. 梯度消失/爆炸

**定理 3.1 (梯度消失/爆炸)**:

考虑 $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}$ (其中 $t > k$)：

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \prod_{i=k+1}^{t} W_{hh}^T \text{diag}(\tanh'(\cdot))
$$

**分析**：

- 如果 $\|W_{hh}\| < 1$：梯度指数衰减 → **梯度消失**
- 如果 $\|W_{hh}\| > 1$：梯度指数增长 → **梯度爆炸**

**后果**：

- **梯度消失**：无法学习长期依赖
- **梯度爆炸**：训练不稳定

---

## 💻 LSTM (长短期记忆网络)

### 1. 动机

**问题**：Vanilla RNN的梯度消失

**解决方案**：引入**门控机制**和**细胞状态**

**关键思想**：

- **细胞状态** $\mathbf{c}_t$：长期记忆
- **门控单元**：控制信息流

---

### 2. LSTM架构

**定义 2.1 (LSTM)**:

$$
\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(遗忘门)}
$$

$$
\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(输入门)}
$$

$$
\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(候选值)}
$$

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(细胞状态)}
$$

$$
\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(输出门)}
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(隐状态)}
$$

其中 $\odot$ 表示逐元素乘法，$[\cdot, \cdot]$ 表示拼接。

---

### 3. 门控机制

**遗忘门 $\mathbf{f}_t$**：

- 决定丢弃多少旧信息
- $f_t \approx 0$：完全遗忘
- $f_t \approx 1$：完全保留

**输入门 $\mathbf{i}_t$**：

- 决定添加多少新信息
- $i_t \approx 0$：忽略新输入
- $i_t \approx 1$：完全接受

**输出门 $\mathbf{o}_t$**：

- 决定输出多少信息
- $o_t \approx 0$：不输出
- $o_t \approx 1$：完全输出

**关键**：门控值在 $(0, 1)$ 之间，梯度不会消失！

---

## 🎨 GRU (门控循环单元)

### 1. GRU架构

**定义 1.1 (GRU)**:

$$
\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(更新门)}
$$

$$
\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(重置门)}
$$

$$
\tilde{\mathbf{h}}_t = \tanh(W [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(候选隐状态)}
$$

$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

**更新门 $\mathbf{z}_t$**：

- 控制新旧信息的混合
- $z_t \approx 0$：保留旧信息
- $z_t \approx 1$：使用新信息

**重置门 $\mathbf{r}_t$**：

- 控制使用多少历史信息
- $r_t \approx 0$：忽略历史
- $r_t \approx 1$：使用历史

---

### 2. GRU vs LSTM

| 特性 | LSTM | GRU |
|------|------|-----|
| **门数量** | 3个（遗忘、输入、输出） | 2个（更新、重置） |
| **状态** | 细胞状态 + 隐状态 | 仅隐状态 |
| **参数数量** | 更多 | 更少 |
| **计算复杂度** | 更高 | 更低 |
| **表达能力** | 更强 | 略弱 |
| **训练速度** | 较慢 | 较快 |

**实践建议**：

- 数据充足 → LSTM
- 数据有限 → GRU
- 先尝试GRU，不行再用LSTM

---

## 📐 双向RNN

### 1. 动机1

**问题**：单向RNN只能看到过去

**示例**：

```text
"The cat sat on the ___"
```

需要看到后面的词才能预测！

---

### 2. 数学定义

**定义 2.1 (Bidirectional RNN)**:

**前向RNN**：

$$
\overrightarrow{\mathbf{h}}_t = \text{RNN}_{\text{forward}}(\overrightarrow{\mathbf{h}}_{t-1}, \mathbf{x}_t)
$$

**后向RNN**：

$$
\overleftarrow{\mathbf{h}}_t = \text{RNN}_{\text{backward}}(\overleftarrow{\mathbf{h}}_{t+1}, \mathbf{x}_t)
$$

**输出**：

$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]
$$

**应用**：

- 命名实体识别
- 词性标注
- 机器翻译（编码器）

---

## 🔧 梯度裁剪

### 1. 梯度爆炸问题

**现象**：

- 梯度范数突然变得很大
- 参数更新过大
- 训练发散

**检测**：

$$
\|\nabla \mathcal{L}\| > \text{threshold}
$$

---

### 2. 梯度裁剪方法

**方法1：按值裁剪**:

$$
g_i = \begin{cases}
\text{threshold} & \text{if } g_i > \text{threshold} \\
-\text{threshold} & \text{if } g_i < -\text{threshold} \\
g_i & \text{otherwise}
\end{cases}
$$

**方法2：按范数裁剪**:

$$
\mathbf{g} = \begin{cases}
\frac{\text{threshold}}{\|\mathbf{g}\|} \mathbf{g} & \text{if } \|\mathbf{g}\| > \text{threshold} \\
\mathbf{g} & \text{otherwise}
\end{cases}
$$

**推荐**：按范数裁剪（保持梯度方向）

---

## 💡 Python实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. 从零实现Vanilla RNN
class VanillaRNN(nn.Module):
    """从零实现的RNN"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 权重矩阵
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h_prev=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            h_prev: (batch, hidden_size) or None
        
        Returns:
            outputs: (batch, seq_len, output_size)
            h: (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐状态
        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_prev
        
        outputs = []
        
        # 时间步循环
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            
            # RNN更新
            h = torch.tanh(self.W_xh(x_t) + self.W_hh(h))
            
            # 输出
            y_t = self.W_hy(h)
            outputs.append(y_t.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, output_size)
        
        return outputs, h


# 2. 使用PyTorch的LSTM
class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            hidden: (h_0, c_0) or None
        
        Returns:
            output: (batch, seq_len, output_size)
            hidden: (h_n, c_n)
        """
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Dropout + FC
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐状态"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


# 3. GRU实现
class GRUModel(nn.Module):
    """GRU模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        gru_out, hidden = self.gru(x, hidden)
        gru_out = self.dropout(gru_out)
        output = self.fc(gru_out)
        return output, hidden


# 4. 双向LSTM
class BiLSTM(nn.Module):
    """双向LSTM"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 双向
        )
        
        # 注意：双向LSTM的输出维度是 2*hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


# 5. 梯度裁剪
def clip_gradient(model, clip_value):
    """梯度裁剪"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)


# 示例使用
if __name__ == "__main__":
    # 参数
    batch_size = 32
    seq_len = 10
    input_size = 50
    hidden_size = 128
    num_layers = 2
    output_size = 10
    
    # 测试Vanilla RNN
    print("=== Vanilla RNN ===")
    rnn = VanillaRNN(input_size, hidden_size, output_size)
    x = torch.randn(batch_size, seq_len, input_size)
    outputs, h = rnn(x)
    print(f"Input: {x.shape}")
    print(f"Output: {outputs.shape}")
    print(f"Hidden: {h.shape}")
    
    # 测试LSTM
    print("\n=== LSTM ===")
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    hidden = lstm_model.init_hidden(batch_size, x.device)
    outputs, hidden = lstm_model(x, hidden)
    print(f"Output: {outputs.shape}")
    print(f"Hidden h: {hidden[0].shape}")
    print(f"Hidden c: {hidden[1].shape}")
    
    # 测试GRU
    print("\n=== GRU ===")
    gru_model = GRUModel(input_size, hidden_size, num_layers, output_size)
    outputs, hidden = gru_model(x)
    print(f"Output: {outputs.shape}")
    print(f"Hidden: {hidden.shape}")
    
    # 测试双向LSTM
    print("\n=== Bidirectional LSTM ===")
    bilstm = BiLSTM(input_size, hidden_size, num_layers, output_size)
    outputs = bilstm(x)
    print(f"Output: {outputs.shape}")
    
    # 梯度裁剪示例
    print("\n=== Gradient Clipping ===")
    optimizer = torch.optim.Adam(lstm_model.parameters())
    loss = outputs.sum()
    loss.backward()
    
    # 裁剪前
    total_norm_before = 0
    for p in lstm_model.parameters():
        if p.grad is not None:
            total_norm_before += p.grad.data.norm(2).item() ** 2
    total_norm_before = total_norm_before ** 0.5
    print(f"Gradient norm before clipping: {total_norm_before:.4f}")
    
    # 裁剪
    clip_gradient(lstm_model, clip_value=1.0)
    
    # 裁剪后
    total_norm_after = 0
    for p in lstm_model.parameters():
        if p.grad is not None:
            total_norm_after += p.grad.data.norm(2).item() ** 2
    total_norm_after = total_norm_after ** 0.5
    print(f"Gradient norm after clipping: {total_norm_after:.4f}")
```

---

## 📚 应用场景

### 1. 语言模型

**任务**：预测下一个词

**架构**：

```text
输入: "The cat sat on"
    ↓
LSTM
    ↓
输出: "the" (概率最高)
```

**损失函数**：交叉熵

---

### 2. 机器翻译

**Seq2Seq架构**：

```text
编码器 (Encoder):
    英文 → LSTM → 上下文向量

解码器 (Decoder):
    上下文向量 → LSTM → 中文
```

**关键**：编码器的最后隐状态作为解码器的初始隐状态

---

### 3. 时间序列预测

**任务**：预测股价、天气等

**架构**：

```text
历史数据 → LSTM → 未来值
```

**特点**：

- 多对一：预测单个值
- 多对多：预测序列

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS224N Natural Language Processing |
| **MIT** | 6.S191 Introduction to Deep Learning |
| **CMU** | 11-747 Neural Networks for NLP |
| **UC Berkeley** | CS182 Deep Learning |

---

## 📖 参考文献

1. **Hochreiter & Schmidhuber (1997)**. "Long Short-Term Memory". *Neural Computation*.

2. **Cho et al. (2014)**. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation". *EMNLP*. (GRU)

3. **Graves (2013)**. "Generating Sequences With Recurrent Neural Networks". *arXiv*.

4. **Sutskever et al. (2014)**. "Sequence to Sequence Learning with Neural Networks". *NeurIPS*.

5. **Pascanu et al. (2013)**. "On the difficulty of training Recurrent Neural Networks". *ICML*. (梯度消失/爆炸)

---

*最后更新：2025年10月*-
