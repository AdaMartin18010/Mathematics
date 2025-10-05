# 卷积神经网络 (CNN) 数学原理

> **Convolutional Neural Networks: Mathematics and Theory**
>
> 计算机视觉的数学基础

---

## 目录

- [卷积神经网络 (CNN) 数学原理](#卷积神经网络-cnn-数学原理)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 动机与优势](#-动机与优势)
    - [1. 全连接层的问题](#1-全连接层的问题)
    - [2. CNN的优势](#2-cnn的优势)
  - [📊 卷积运算](#-卷积运算)
    - [1. 数学定义](#1-数学定义)
    - [2. 离散卷积](#2-离散卷积)
    - [3. 互相关 vs 卷积](#3-互相关-vs-卷积)
  - [🔬 卷积层的数学](#-卷积层的数学)
    - [1. 前向传播](#1-前向传播)
    - [2. 参数共享](#2-参数共享)
    - [3. 局部连接](#3-局部连接)
  - [💻 感受野分析](#-感受野分析)
    - [1. 感受野定义](#1-感受野定义)
    - [2. 感受野计算](#2-感受野计算)
    - [3. 有效感受野](#3-有效感受野)
  - [🎨 池化层](#-池化层)
    - [1. 最大池化](#1-最大池化)
    - [2. 平均池化](#2-平均池化)
    - [3. 池化的作用](#3-池化的作用)
  - [📐 输出尺寸计算](#-输出尺寸计算)
    - [1. 卷积输出尺寸](#1-卷积输出尺寸)
    - [2. 池化输出尺寸](#2-池化输出尺寸)
    - [3. 填充策略](#3-填充策略)
  - [🔧 反向传播](#-反向传播)
    - [1. 卷积层梯度](#1-卷积层梯度)
    - [2. 池化层梯度](#2-池化层梯度)
  - [💡 Python实现](#-python实现)
  - [📚 经典CNN架构](#-经典cnn架构)
    - [1. LeNet-5](#1-lenet-5)
    - [2. AlexNet](#2-alexnet)
    - [3. VGG](#3-vgg)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**卷积神经网络 (CNN)** 通过**卷积操作**提取图像的局部特征。

**核心原理**：

```text
输入图像
    ↓
卷积层 (特征提取)
    ↓
激活函数 (非线性)
    ↓
池化层 (降维)
    ↓
重复多层
    ↓
全连接层 (分类)
    ↓
输出
```

**关键概念**：

- **局部连接**：每个神经元只连接局部区域
- **参数共享**：同一卷积核在整个图像上共享
- **平移不变性**：对输入的平移具有鲁棒性

---

## 🎯 动机与优势

### 1. 全连接层的问题

**示例**：处理 $224 \times 224 \times 3$ 的RGB图像

**全连接层**：

- 输入维度：$224 \times 224 \times 3 = 150,528$
- 第一隐藏层1000个神经元
- 参数数量：$150,528 \times 1000 = 150M$ 参数！

**问题**：

- 参数过多，容易过拟合
- 计算量巨大
- 忽略空间结构

---

### 2. CNN的优势

**参数共享**：

- 同一卷积核在整个图像上滑动
- 大幅减少参数数量

**局部连接**：

- 每个神经元只关注局部区域
- 符合视觉感知原理

**平移不变性**：

- 特征检测器在图像任何位置都有效
- 对物体位置变化鲁棒

**层次化特征**：

- 低层：边缘、纹理
- 中层：部件、形状
- 高层：物体、场景

---

## 📊 卷积运算

### 1. 数学定义

**连续卷积**：

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
$$

**物理意义**：

- $f$：输入信号
- $g$：卷积核（滤波器）
- $f * g$：滤波后的信号

---

### 2. 离散卷积

**2D离散卷积**：

$$
(I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) K(m, n)
$$

其中：

- $I$：输入图像
- $K$：卷积核
- $(i, j)$：输出位置

**示例**：$3 \times 3$ 卷积核

$$
\begin{bmatrix}
I(i-1, j-1) & I(i-1, j) & I(i-1, j+1) \\
I(i, j-1) & I(i, j) & I(i, j+1) \\
I(i+1, j-1) & I(i+1, j) & I(i+1, j+1)
\end{bmatrix}
\odot
\begin{bmatrix}
K(1, 1) & K(1, 2) & K(1, 3) \\
K(2, 1) & K(2, 2) & K(2, 3) \\
K(3, 1) & K(3, 2) & K(3, 3)
\end{bmatrix}
$$

---

### 3. 互相关 vs 卷积

**互相关 (Cross-Correlation)**：

$$
(I \star K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) K(m, n)
$$

**卷积 (Convolution)**：

$$
(I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) K(m, n)
$$

**关系**：

- 互相关：不翻转卷积核
- 卷积：翻转卷积核

**深度学习中**：

- 通常使用互相关
- 但习惯称为"卷积"
- 因为卷积核是学习的，翻转与否无关紧要

---

## 🔬 卷积层的数学

### 1. 前向传播

**输入**：

- $X \in \mathbb{R}^{H \times W \times C_{in}}$（高度×宽度×输入通道数）

**卷积核**：

- $K \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}$

**输出**：

- $Y \in \mathbb{R}^{H' \times W' \times C_{out}}$

**计算**：

$$
Y_{i,j,c} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c'=0}^{C_{in}-1} X_{i+m, j+n, c'} \cdot K_{m, n, c', c} + b_c
$$

其中 $b_c$ 是偏置。

---

### 2. 参数共享

**全连接层**：

- 每个输出神经元有独立的权重
- 参数数量：$H \times W \times C_{in} \times C_{out}$

**卷积层**：

- 同一卷积核在整个图像上共享
- 参数数量：$k_h \times k_w \times C_{in} \times C_{out}$

**示例**：

- 输入：$224 \times 224 \times 3$
- 卷积核：$3 \times 3$，64个
- 参数：$3 \times 3 \times 3 \times 64 = 1,728$（vs 全连接的150M！）

---

### 3. 局部连接

**全连接层**：

- 每个输出神经元连接所有输入

**卷积层**：

- 每个输出神经元只连接局部感受野
- 感受野大小：$k_h \times k_w$

**优势**：

- 减少计算量
- 符合视觉处理原理（局部特征）
- 更好的泛化能力

---

## 💻 感受野分析

### 1. 感受野定义

**定义 1.1 (感受野, Receptive Field)**:

某层的一个神经元在输入图像上能"看到"的区域大小。

**直觉**：

```text
输入层 → 第1层 → 第2层 → 第3层
  ■       ■       ■       ■
(1×1)   (3×3)   (5×5)   (7×7)
```

---

### 2. 感受野计算

**单层卷积**：

- 卷积核大小：$k$
- 感受野：$r = k$

**两层卷积**：

- 第1层：$k_1 \times k_1$，感受野 $r_1 = k_1$
- 第2层：$k_2 \times k_2$，感受野 $r_2 = r_1 + (k_2 - 1)$

**递推公式**：

$$
r_l = r_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i
$$

其中 $s_i$ 是第 $i$ 层的步长。

**示例**：

- 3层 $3 \times 3$ 卷积，步长1
- 感受野：$r = 1 + 2 + 2 + 2 = 7$

---

### 3. 有效感受野

**问题**：理论感受野 ≠ 有效感受野

**有效感受野 (Effective Receptive Field)**：

- 感受野中心的像素贡献最大
- 边缘像素贡献很小
- 呈高斯分布

**实验发现** (Luo et al., 2016)：

- 有效感受野远小于理论感受野
- 中心区域占主导
- 需要更深的网络才能获得大感受野

---

## 🎨 池化层

### 1. 最大池化

**定义 1.1 (Max Pooling)**:

$$
Y_{i,j} = \max_{m=0}^{k-1} \max_{n=0}^{k-1} X_{i \cdot s + m, j \cdot s + n}
$$

其中 $k$ 是池化窗口大小，$s$ 是步长。

**示例**：$2 \times 2$ 最大池化

$$
\begin{bmatrix}
1 & 3 & 2 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
\to
\begin{bmatrix}
6 & 8 \\
14 & 16
\end{bmatrix}
$$

---

### 2. 平均池化

**定义 2.1 (Average Pooling)**:

$$
Y_{i,j} = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i \cdot s + m, j \cdot s + n}
$$

**对比**：

- **最大池化**：保留最强特征，常用于卷积层后
- **平均池化**：平滑特征，常用于全局池化

---

### 3. 池化的作用

**降维**：

- 减少特征图尺寸
- 减少计算量

**平移不变性**：

- 小的位置变化不影响输出
- 提高鲁棒性

**增大感受野**：

- 间接增加后续层的感受野

**防止过拟合**：

- 减少参数数量
- 正则化效果

---

## 📐 输出尺寸计算

### 1. 卷积输出尺寸

**公式**：

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k_h}{s} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in} + 2p - k_w}{s} \right\rfloor + 1
$$

其中：

- $H_{in}, W_{in}$：输入高度和宽度
- $k_h, k_w$：卷积核高度和宽度
- $p$：填充 (padding)
- $s$：步长 (stride)

**示例**：

- 输入：$32 \times 32$
- 卷积核：$5 \times 5$
- 步长：1
- 填充：0
- 输出：$\lfloor (32 - 5) / 1 \rfloor + 1 = 28$

---

### 2. 池化输出尺寸

**公式**：

$$
H_{out} = \left\lfloor \frac{H_{in} - k}{s} \right\rfloor + 1
$$

**示例**：

- 输入：$28 \times 28$
- 池化窗口：$2 \times 2$
- 步长：2
- 输出：$\lfloor (28 - 2) / 2 \rfloor + 1 = 14$

---

### 3. 填充策略

**Valid Padding** ($p = 0$)：

- 不填充
- 输出尺寸减小

**Same Padding**：

- 填充使输出尺寸 = 输入尺寸（步长=1时）
- $p = \lfloor k / 2 \rfloor$

**Full Padding**：

- 填充使每个输入像素都被卷积核完整覆盖
- $p = k - 1$

---

## 🔧 反向传播

### 1. 卷积层梯度

**前向**：$Y = X * K + b$

**已知**：$\frac{\partial \mathcal{L}}{\partial Y}$

**目标**：计算 $\frac{\partial \mathcal{L}}{\partial X}, \frac{\partial \mathcal{L}}{\partial K}, \frac{\partial \mathcal{L}}{\partial b}$

**梯度计算**：

1. **对偏置**：
   $$
   \frac{\partial \mathcal{L}}{\partial b_c} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial Y_{i,j,c}}
   $$

2. **对卷积核**：
   $$
   \frac{\partial \mathcal{L}}{\partial K_{m,n,c',c}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial Y_{i,j,c}} \cdot X_{i+m, j+n, c'}
   $$

3. **对输入**：
   $$
   \frac{\partial \mathcal{L}}{\partial X_{i,j,c'}} = \sum_{m,n,c} \frac{\partial \mathcal{L}}{\partial Y_{i-m, j-n, c}} \cdot K_{m,n,c',c}
   $$

**关键**：输入梯度是卷积核的"转置卷积"。

---

### 2. 池化层梯度

**最大池化**：

- 梯度只传递给最大值位置
- 其他位置梯度为0

$$
\frac{\partial \mathcal{L}}{\partial X_{i,j}} = \begin{cases}
\frac{\partial \mathcal{L}}{\partial Y_{k,l}} & \text{if } X_{i,j} = \max(\text{pool window}) \\
0 & \text{otherwise}
\end{cases}
$$

**平均池化**：

- 梯度均匀分配给窗口内所有位置

$$
\frac{\partial \mathcal{L}}{\partial X_{i,j}} = \frac{1}{k^2} \frac{\partial \mathcal{L}}{\partial Y_{k,l}}
$$

---

## 💡 Python实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. 从零实现2D卷积
def conv2d_manual(input, kernel, stride=1, padding=0):
    """
    手动实现2D卷积
    
    Args:
        input: (batch, in_channels, H, W)
        kernel: (out_channels, in_channels, kH, kW)
        stride: int
        padding: int
    
    Returns:
        output: (batch, out_channels, H_out, W_out)
    """
    batch, in_channels, H, W = input.shape
    out_channels, _, kH, kW = kernel.shape
    
    # 添加填充
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))
        H, W = H + 2 * padding, W + 2 * padding
    
    # 计算输出尺寸
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    
    # 初始化输出
    output = torch.zeros(batch, out_channels, H_out, W_out)
    
    # 卷积操作
    for b in range(batch):
        for oc in range(out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    
                    # 提取感受野
                    receptive_field = input[b, :, h_start:h_start+kH, w_start:w_start+kW]
                    
                    # 卷积
                    output[b, oc, i, j] = torch.sum(receptive_field * kernel[oc])
    
    return output


# 2. 使用PyTorch的卷积层
class SimpleCNN(nn.Module):
    """简单的CNN示例"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 卷积层1: 3 -> 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 卷积层3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv1: 32x32x3 -> 32x32x32 -> 16x16x32
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Conv2: 16x16x32 -> 16x16x64 -> 8x8x64
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Conv3: 8x8x64 -> 8x8x128 -> 4x4x128
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128*4*4)
        
        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# 3. 感受野计算
def compute_receptive_field(layers_config):
    """
    计算感受野
    
    Args:
        layers_config: list of (kernel_size, stride)
    
    Returns:
        receptive_field: int
    """
    rf = 1
    stride_product = 1
    
    for k, s in layers_config:
        rf = rf + (k - 1) * stride_product
        stride_product *= s
    
    return rf


# 示例使用
if __name__ == "__main__":
    # 测试手动卷积
    print("=== 测试手动卷积 ===")
    input_tensor = torch.randn(1, 3, 5, 5)
    kernel = torch.randn(16, 3, 3, 3)
    
    output_manual = conv2d_manual(input_tensor, kernel, stride=1, padding=1)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {output_manual.shape}")
    
    # 对比PyTorch实现
    conv_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    conv_layer.weight.data = kernel
    output_pytorch = conv_layer(input_tensor)
    
    print(f"PyTorch output shape: {output_pytorch.shape}")
    print(f"Difference: {torch.max(torch.abs(output_manual - output_pytorch)).item():.6f}")
    
    # 测试SimpleCNN
    print("\n=== 测试SimpleCNN ===")
    model = SimpleCNN(num_classes=10)
    x = torch.randn(2, 3, 32, 32)  # CIFAR-10 size
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 计算感受野
    print("\n=== 计算感受野 ===")
    layers = [
        (3, 1),  # conv1
        (2, 2),  # pool1
        (3, 1),  # conv2
        (2, 2),  # pool2
        (3, 1),  # conv3
        (2, 2),  # pool3
    ]
    rf = compute_receptive_field(layers)
    print(f"Receptive field: {rf}x{rf}")


# 4. 可视化卷积核
def visualize_filters(model, layer_name='conv1'):
    """可视化卷积核"""
    import matplotlib.pyplot as plt
    
    # 获取第一层卷积核
    conv_layer = getattr(model, layer_name)
    filters = conv_layer.weight.data.cpu().numpy()
    
    # filters shape: (out_channels, in_channels, kH, kW)
    num_filters = min(filters.shape[0], 32)  # 最多显示32个
    
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # 取第一个输入通道
            filter_img = filters[i, 0, :, :]
            ax.imshow(filter_img, cmap='gray')
            ax.set_title(f'Filter {i}')
        ax.axis('off')
    
    plt.suptitle(f'{layer_name} Filters')
    plt.tight_layout()
    # plt.show()


# 5. 特征图可视化
def visualize_feature_maps(model, input_image, layer_name='conv1'):
    """可视化特征图"""
    import matplotlib.pyplot as plt
    
    # 前向传播到指定层
    x = input_image
    for name, module in model.named_children():
        x = module(x)
        if name == layer_name:
            break
    
    # x shape: (1, channels, H, W)
    feature_maps = x.squeeze(0).detach().cpu().numpy()
    num_maps = min(feature_maps.shape[0], 32)
    
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.set_title(f'Map {i}')
        ax.axis('off')
    
    plt.suptitle(f'{layer_name} Feature Maps')
    plt.tight_layout()
    # plt.show()
```

---

## 📚 经典CNN架构

### 1. LeNet-5

**架构** (LeCun et al., 1998)：

```text
Input (32x32x1)
    ↓
Conv1: 6 filters, 5x5 → (28x28x6)
    ↓
AvgPool: 2x2 → (14x14x6)
    ↓
Conv2: 16 filters, 5x5 → (10x10x16)
    ↓
AvgPool: 2x2 → (5x5x16)
    ↓
FC1: 120
    ↓
FC2: 84
    ↓
Output: 10
```

**特点**：

- 最早的CNN之一
- 用于手写数字识别
- 约60K参数

---

### 2. AlexNet

**架构** (Krizhevsky et al., 2012)：

```text
Input (227x227x3)
    ↓
Conv1: 96 filters, 11x11, stride 4
    ↓
MaxPool: 3x3, stride 2
    ↓
Conv2: 256 filters, 5x5
    ↓
MaxPool: 3x3, stride 2
    ↓
Conv3: 384 filters, 3x3
Conv4: 384 filters, 3x3
Conv5: 256 filters, 3x3
    ↓
MaxPool: 3x3, stride 2
    ↓
FC1: 4096
FC2: 4096
FC3: 1000
```

**创新**：

- ReLU激活
- Dropout
- 数据增强
- GPU训练

---

### 3. VGG

**架构** (Simonyan & Zisserman, 2014)：

```text
多层 3x3 卷积 + 2x2 MaxPool
    ↓
VGG-16: 13个卷积层 + 3个全连接层
VGG-19: 16个卷积层 + 3个全连接层
```

**特点**：

- 全部使用 $3 \times 3$ 卷积
- 更深的网络
- 约138M参数

**洞察**：

- 两个 $3 \times 3$ 卷积 = 一个 $5 \times 5$ 感受野
- 但参数更少：$2 \times (3^2) < 5^2$

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS231n Convolutional Neural Networks |
| **MIT** | 6.S191 Introduction to Deep Learning |
| **UC Berkeley** | CS182 Deep Learning |
| **CMU** | 11-785 Introduction to Deep Learning |

---

## 📖 参考文献

1. **LeCun et al. (1998)**. "Gradient-Based Learning Applied to Document Recognition". *Proceedings of the IEEE*.

2. **Krizhevsky et al. (2012)**. "ImageNet Classification with Deep Convolutional Neural Networks". *NeurIPS*.

3. **Simonyan & Zisserman (2014)**. "Very Deep Convolutional Networks for Large-Scale Image Recognition". *ICLR*.

4. **He et al. (2016)**. "Deep Residual Learning for Image Recognition". *CVPR*.

5. **Luo et al. (2016)**. "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks". *NeurIPS*.

---

*最后更新：2025年10月*-
