# 损失函数理论 (Loss Functions Theory)

> **From Mean Squared Error to Contrastive Learning**
>
> 深度学习的优化目标

---

## 目录

- [损失函数理论 (Loss Functions Theory)](#损失函数理论-loss-functions-theory)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 损失函数的作用](#-损失函数的作用)
  - [📊 回归损失函数](#-回归损失函数)
    - [1. 均方误差 (MSE)](#1-均方误差-mse)
    - [2. 平均绝对误差 (MAE)](#2-平均绝对误差-mae)
    - [3. Huber损失](#3-huber损失)
  - [🔢 分类损失函数](#-分类损失函数)
    - [1. 交叉熵损失](#1-交叉熵损失)
    - [2. Focal Loss](#2-focal-loss)
    - [3. Label Smoothing](#3-label-smoothing)
  - [🎨 对比学习损失](#-对比学习损失)
    - [1. Contrastive Loss](#1-contrastive-loss)
    - [2. Triplet Loss](#2-triplet-loss)
    - [3. InfoNCE Loss](#3-infonce-loss)
  - [🔬 生成模型损失](#-生成模型损失)
    - [1. VAE损失 (ELBO)](#1-vae损失-elbo)
    - [2. GAN损失](#2-gan损失)
    - [3. 感知损失 (Perceptual Loss)](#3-感知损失-perceptual-loss)
  - [💡 损失函数设计原则](#-损失函数设计原则)
    - [1. 可微性](#1-可微性)
    - [2. 凸性](#2-凸性)
    - [3. 鲁棒性](#3-鲁棒性)
    - [4. 任务对齐](#4-任务对齐)
  - [🔧 实践技巧](#-实践技巧)
    - [1. 损失函数组合](#1-损失函数组合)
    - [2. 损失权重调整](#2-损失权重调整)
    - [3. 动态损失权重](#3-动态损失权重)
  - [💻 Python实现](#-python实现)
  - [📚 损失函数选择指南](#-损失函数选择指南)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**损失函数** (Loss Function) 量化模型预测与真实标签之间的差异，是深度学习优化的核心。

**核心原理**：

```text
损失函数的作用:
    1. 量化预测误差
    2. 提供优化方向
    3. 反映任务目标

设计原则:
    可微性 → 梯度下降
    凸性 → 全局最优
    鲁棒性 → 抗噪声
    任务对齐 → 性能提升
```

---

## 🎯 损失函数的作用

**定义**：

给定模型 $f_\theta: \mathcal{X} \to \mathcal{Y}$，损失函数 $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}$ 衡量预测 $\hat{y} = f_\theta(x)$ 与真实标签 $y$ 的差异。

**经验风险**：

$$
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)
$$

**优化目标**：

$$
\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)
$$

---

## 📊 回归损失函数

### 1. 均方误差 (MSE)

**定义 1.1 (Mean Squared Error)**:

$$
\ell_{\text{MSE}}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**性质**：

- **可微**：$\frac{\partial \ell}{\partial \hat{y}} = -2(y - \hat{y})$
- **凸函数**：全局最优
- **对异常值敏感**：平方放大误差

**概率解释**：

假设 $y = f(x) + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$，则最大似然估计等价于最小化MSE。

**应用**：

- 回归任务
- 图像重建
- 信号处理

---

### 2. 平均绝对误差 (MAE)

**定义 2.1 (Mean Absolute Error)**:

$$
\ell_{\text{MAE}}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**性质**：

- **鲁棒性强**：对异常值不敏感
- **非光滑**：在 $y = \hat{y}$ 处不可微
- **中位数估计**：最优解为条件中位数

**对比MSE**：

| 特性 | MSE | MAE |
|------|-----|-----|
| **异常值敏感性** | 高 | 低 |
| **梯度** | 线性 | 常数 |
| **优化难度** | 易 | 难 |

---

### 3. Huber损失

**定义 3.1 (Huber Loss)**:

$$
\ell_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

**特点**：

- **结合MSE和MAE**：小误差用MSE，大误差用MAE
- **平滑可微**：全局可微
- **鲁棒性好**：对异常值不敏感

**超参数 $\delta$**：

- 小 $\delta$：接近MAE
- 大 $\delta$：接近MSE

---

## 🔢 分类损失函数

### 1. 交叉熵损失

**定义 1.1 (Cross-Entropy Loss)**:

**二分类**：

$$
\ell_{\text{CE}}(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

**多分类**：

$$
\ell_{\text{CE}}(y, \hat{y}) = -\sum_{c=1}^{C} y_c \log \hat{y}_c
$$

其中 $y$ 是one-hot编码，$\hat{y} = \text{softmax}(z)$。

**信息论解释**：

交叉熵衡量两个分布的差异：

$$
H(p, q) = -\sum_{x} p(x) \log q(x)
$$

**KL散度**：

$$
D_{\text{KL}}(p \| q) = H(p, q) - H(p)
$$

最小化交叉熵等价于最小化KL散度。

**梯度**：

$$
\frac{\partial \ell_{\text{CE}}}{\partial z_i} = \hat{y}_i - y_i
$$

非常简洁！

---

### 2. Focal Loss

**定义 2.1 (Focal Loss, Lin et al. 2017)**:

$$
\ell_{\text{FL}}(y, \hat{y}) = -\alpha (1 - \hat{y})^\gamma y \log \hat{y}
$$

**动机**：解决类别不平衡问题

**关键思想**：

- $(1 - \hat{y})^\gamma$ 是**调制因子**
- 易分类样本（$\hat{y} \to 1$）：权重小
- 难分类样本（$\hat{y} \to 0$）：权重大

**超参数**：

- $\gamma \in [0, 5]$：聚焦参数（通常2）
- $\alpha \in [0, 1]$：类别权重

**应用**：

- 目标检测（RetinaNet）
- 不平衡分类

---

### 3. Label Smoothing

**定义 3.1 (Label Smoothing)**:

将硬标签 $y$ 平滑为：

$$
y_{\text{smooth}} = (1 - \epsilon) y + \frac{\epsilon}{C}
$$

其中 $\epsilon$ 是平滑参数（如0.1），$C$ 是类别数。

**效果**：

- **防止过拟合**：减少模型过度自信
- **提高泛化**：鼓励模型输出更平滑的分布

**损失函数**：

$$
\ell_{\text{LS}} = -\sum_{c=1}^{C} y_{\text{smooth}, c} \log \hat{y}_c
$$

---

## 🎨 对比学习损失

### 1. Contrastive Loss

**定义 1.1 (Contrastive Loss)**:

给定样本对 $(x_i, x_j)$ 和标签 $y_{ij}$（相似为1，不相似为0）：

$$
\ell_{\text{contrastive}} = y_{ij} d_{ij}^2 + (1 - y_{ij}) \max(0, m - d_{ij})^2
$$

其中 $d_{ij} = \|f(x_i) - f(x_j)\|_2$ 是嵌入距离，$m$ 是边界。

**直觉**：

- 相似样本：拉近
- 不相似样本：推远（至少距离 $m$）

---

### 2. Triplet Loss

**定义 2.1 (Triplet Loss)**:

给定三元组 $(a, p, n)$（锚点、正样本、负样本）：

$$
\ell_{\text{triplet}} = \max(0, d(a, p) - d(a, n) + \alpha)
$$

其中 $\alpha$ 是边界。

**目标**：

$$
d(a, p) + \alpha < d(a, n)
$$

正样本比负样本至少近 $\alpha$。

**难样本挖掘**：

- **Hard negative**：$d(a, n)$ 最小的负样本
- **Semi-hard negative**：$d(a, p) < d(a, n) < d(a, p) + \alpha$

---

### 3. InfoNCE Loss

**定义 3.1 (InfoNCE Loss, Oord et al. 2018)**:

给定查询 $q$ 和一组样本 $\{k_0, k_1, \ldots, k_N\}$，其中 $k_0$ 是正样本：

$$
\ell_{\text{InfoNCE}} = -\log \frac{\exp(q \cdot k_0 / \tau)}{\sum_{i=0}^{N} \exp(q \cdot k_i / \tau)}
$$

其中 $\tau$ 是温度参数。

**信息论解释**：

最大化互信息 $I(q; k_0)$。

**应用**：

- SimCLR
- MoCo
- CLIP

---

## 🔬 生成模型损失

### 1. VAE损失 (ELBO)

**定义 1.1 (Evidence Lower Bound)**:

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))
$$

**两项**：

1. **重构损失**：$\mathbb{E}[\log p(x|z)]$
2. **KL散度**：$D_{\text{KL}}(q(z|x) \| p(z))$

**实践中**：

$$
\mathcal{L}_{\text{VAE}} = \|x - \hat{x}\|^2 + \beta \cdot D_{\text{KL}}
$$

其中 $\beta$ 控制权衡（$\beta$-VAE）。

---

### 2. GAN损失

**定义 2.1 (GAN Loss)**:

**判别器**：

$$
\max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**生成器**：

$$
\min_G \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**非饱和损失** (Non-saturating loss)：

$$
\max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

**WGAN损失**：

$$
\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

---

### 3. 感知损失 (Perceptual Loss)

**定义 3.1 (Perceptual Loss, Johnson et al. 2016)**:

使用预训练网络（如VGG）的特征：

$$
\ell_{\text{perceptual}} = \sum_{l} \lambda_l \|f_l(x) - f_l(\hat{x})\|^2
$$

其中 $f_l$ 是第 $l$ 层的特征。

**优势**：

- 捕获高级语义
- 比像素级损失更好

**应用**：

- 风格迁移
- 超分辨率
- 图像生成

---

## 💡 损失函数设计原则

### 1. 可微性

**要求**：损失函数必须可微，以便梯度下降。

**例外**：

- MAE在0处不可微（次梯度）
- 0-1损失不可微（用交叉熵替代）

---

### 2. 凸性

**凸函数**：任意局部最优即全局最优

**非凸损失**：

- 深度神经网络的损失通常非凸
- 依赖初始化和优化算法

---

### 3. 鲁棒性

**对异常值的敏感性**：

- MSE：敏感
- MAE：鲁棒
- Huber：平衡

---

### 4. 任务对齐

**损失函数应反映任务目标**：

- 分类：交叉熵
- 回归：MSE/MAE
- 排序：Ranking loss
- 生成：ELBO/GAN loss

---

## 🔧 实践技巧

### 1. 损失函数组合

**多任务学习**：

$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_2 + \cdots
$$

**示例**：

- 图像分割：交叉熵 + Dice loss
- 目标检测：分类损失 + 定位损失
- 风格迁移：内容损失 + 风格损失

---

### 2. 损失权重调整

**类别不平衡**：

$$
\mathcal{L} = \sum_{c=1}^{C} w_c \ell_c
$$

其中 $w_c = \frac{n}{C \cdot n_c}$（逆频率加权）。

---

### 3. 动态损失权重

**不确定性加权** (Kendall et al. 2018)：

$$
\mathcal{L} = \sum_{i} \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i
$$

其中 $\sigma_i$ 是可学习的不确定性参数。

---

## 💻 Python实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. Huber Loss
class HuberLoss(nn.Module):
    """Huber Loss"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        abs_error = torch.abs(error)
        
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()


# 2. Focal Loss
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


# 3. Label Smoothing Cross-Entropy
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-Entropy with Label Smoothing"""
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # One-hot encoding
        targets_one_hot = F.one_hot(targets, n_classes).float()
        
        # Label smoothing
        targets_smooth = (1 - self.epsilon) * targets_one_hot + \
                         self.epsilon / n_classes
        
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        return loss.mean()


# 4. Contrastive Loss
class ContrastiveLoss(nn.Module):
    """Contrastive Loss for Siamese Networks"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1, embedding2: (N, D) embeddings
            label: (N,) 1 for similar, 0 for dissimilar
        """
        distance = F.pairwise_distance(embedding1, embedding2)
        
        loss_similar = label * distance.pow(2)
        loss_dissimilar = (1 - label) * F.relu(self.margin - distance).pow(2)
        
        loss = loss_similar + loss_dissimilar
        return loss.mean()


# 5. Triplet Loss
class TripletLoss(nn.Module):
    """Triplet Loss"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: (N, D) embeddings
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


# 6. InfoNCE Loss
class InfoNCELoss(nn.Module):
    """InfoNCE Loss for Contrastive Learning"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query, positive_key, negative_keys):
        """
        Args:
            query: (N, D)
            positive_key: (N, D)
            negative_keys: (N, K, D) or (K, D)
        """
        # Normalize
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        
        # Positive logits: (N,)
        positive_logits = torch.sum(query * positive_key, dim=-1) / self.temperature
        
        # Negative logits
        if negative_keys.dim() == 2:
            # (K, D) -> (N, K)
            negative_keys = F.normalize(negative_keys, dim=-1)
            negative_logits = torch.matmul(query, negative_keys.T) / self.temperature
        else:
            # (N, K, D)
            negative_keys = F.normalize(negative_keys, dim=-1)
            negative_logits = torch.sum(
                query.unsqueeze(1) * negative_keys, dim=-1
            ) / self.temperature
        
        # Concatenate positive and negative logits
        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        
        # Labels: positive is at index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


# 7. Dice Loss (for segmentation)
class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) probabilities
            targets: (N, C, H, W) one-hot encoded
        """
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        
        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# 示例使用
if __name__ == "__main__":
    # 测试Focal Loss
    print("=== Focal Loss ===")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    inputs = torch.randn(32, 10)  # (batch, classes)
    targets = torch.randint(0, 10, (32,))
    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # 测试Label Smoothing
    print("\n=== Label Smoothing ===")
    ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
    loss = ls_loss(inputs, targets)
    print(f"Label Smoothing Loss: {loss.item():.4f}")
    
    # 测试Triplet Loss
    print("\n=== Triplet Loss ===")
    triplet_loss = TripletLoss(margin=1.0)
    anchor = torch.randn(32, 128)
    positive = torch.randn(32, 128)
    negative = torch.randn(32, 128)
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss.item():.4f}")
    
    # 测试InfoNCE Loss
    print("\n=== InfoNCE Loss ===")
    infonce_loss = InfoNCELoss(temperature=0.07)
    query = torch.randn(32, 128)
    positive_key = torch.randn(32, 128)
    negative_keys = torch.randn(32, 100, 128)
    loss = infonce_loss(query, positive_key, negative_keys)
    print(f"InfoNCE Loss: {loss.item():.4f}")
    
    # 可视化不同损失函数
    import matplotlib.pyplot as plt
    
    print("\n=== 可视化损失函数 ===")
    x = np.linspace(-3, 3, 100)
    
    # MSE, MAE, Huber
    mse = x**2
    mae = np.abs(x)
    huber = np.where(np.abs(x) <= 1, 0.5 * x**2, np.abs(x) - 0.5)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, mse, label='MSE', linewidth=2)
    plt.plot(x, mae, label='MAE', linewidth=2)
    plt.plot(x, huber, label='Huber (δ=1)', linewidth=2)
    plt.xlabel('Error (y - ŷ)')
    plt.ylabel('Loss')
    plt.title('Comparison of Regression Loss Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
```

---

## 📚 损失函数选择指南

| 任务 | 推荐损失函数 | 备注 |
|------|-------------|------|
| **回归** | MSE / MAE / Huber | MSE对异常值敏感，MAE鲁棒 |
| **二分类** | Binary Cross-Entropy | 标准选择 |
| **多分类** | Cross-Entropy | 标准选择 |
| **不平衡分类** | Focal Loss | 聚焦难样本 |
| **图像分割** | Cross-Entropy + Dice | 结合像素和区域 |
| **目标检测** | Focal Loss + IoU Loss | 分类 + 定位 |
| **度量学习** | Triplet / Contrastive | 学习嵌入空间 |
| **对比学习** | InfoNCE | 自监督学习 |
| **图像生成** | Perceptual + GAN | 高质量生成 |
| **VAE** | ELBO | 重构 + KL散度 |

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS229 Machine Learning |
| **MIT** | 6.036 Introduction to Machine Learning |
| **UC Berkeley** | CS189 Introduction to Machine Learning |
| **CMU** | 10-701 Introduction to Machine Learning |

---

## 📖 参考文献

1. **Goodfellow et al. (2016)**. *Deep Learning*. MIT Press. (Chapter 5: Machine Learning Basics)

2. **Lin et al. (2017)**. "Focal Loss for Dense Object Detection". *ICCV*. (Focal Loss)

3. **Szegedy et al. (2016)**. "Rethinking the Inception Architecture for Computer Vision". *CVPR*. (Label Smoothing)

4. **Schroff et al. (2015)**. "FaceNet: A Unified Embedding for Face Recognition and Clustering". *CVPR*. (Triplet Loss)

5. **Oord et al. (2018)**. "Representation Learning with Contrastive Predictive Coding". *arXiv*. (InfoNCE)

6. **Johnson et al. (2016)**. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution". *ECCV*. (Perceptual Loss)

7. **Kendall et al. (2018)**. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics". *CVPR*. (Uncertainty Weighting)

---

*最后更新：2025年10月*-
