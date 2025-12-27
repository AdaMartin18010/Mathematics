# 扩散模型与Score-Based SDE

> **Diffusion Models and Score-Based Stochastic Differential Equations**
>
> 2025年生成模型的前沿：从DDPM到连续时间扩散

---

## 目录

- [扩散模型与Score-Based SDE](#扩散模型与score-based-sde)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 扩散过程](#-扩散过程)
    - [1. 前向扩散](#1-前向扩散)
    - [2. 反向扩散](#2-反向扩散)
  - [📊 Score-Based模型](#-score-based模型)
    - [1. Score函数](#1-score函数)
    - [2. Score Matching](#2-score-matching)
  - [🔬 数学理论](#-数学理论)
    - [1. 随机微分方程 (SDE)](#1-随机微分方程-sde)
    - [2. 概率流ODE](#2-概率流ode)
  - [💻 Python实现](#-python实现)
  - [📚 核心模型](#-核心模型)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**扩散模型**通过逐步添加噪声然后学习去噪来生成数据。

**核心流程**：

```text
数据 x₀ → 加噪 → ... → 纯噪声 x_T
         ↓ 学习反向过程
生成 x₀ ← 去噪 ← ... ← 采样 x_T
```

**优势**：

- 高质量生成
- 稳定训练
- 理论保证

---

## 🎯 扩散过程

### 1. 前向扩散

**离散时间 (DDPM)**:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

**封闭形式**：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$。

---

**连续时间 (SDE)**:

$$
dx = f(x, t) dt + g(t) dw
$$

**方差保持 (VP) SDE**：

$$
dx = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)} dw
$$

---

### 2. 反向扩散

**定理 2.1 (反向SDE, Anderson 1982)**:

前向SDE的反向过程为：

$$
dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{w}
$$

其中 $\nabla_x \log p_t(x)$ 是**score函数**。

---

**训练目标**：学习score函数 $s_\theta(x, t) \approx \nabla_x \log p_t(x)$。

**采样**：从 $x_T \sim \mathcal{N}(0, I)$ 开始，沿反向SDE积分。

---

## 📊 Score-Based模型

### 1. Score函数

**定义 1.1 (Score函数)**:

$$
s(x) = \nabla_x \log p(x)
$$

**直觉**：指向数据密度增加最快的方向。

---

### 2. Score Matching

**目标**：最小化Fisher散度。

$$
\mathbb{E}_{p(x)}[\|s_\theta(x) - \nabla_x \log p(x)\|^2]
$$

**去噪Score Matching (DSM)**：

$$
\mathbb{E}_{p(x_0)} \mathbb{E}_{q(x_t|x_0)}[\|s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t|x_0)\|^2]
$$

**简化形式 (DDPM目标)**：

$$
\mathbb{E}_{t, x_0, \epsilon}[\|\epsilon_\theta(x_t, t) - \epsilon\|^2]
$$

其中 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

---

## 🔬 数学理论

### 1. 随机微分方程 (SDE)

**前向VP-SDE**：

$$
dx = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)} dw, \quad t \in [0, T]
$$

**边际分布**：

$$
p_t(x | x_0) = \mathcal{N}(x; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

其中 $\bar{\alpha}_t = \exp\left(-\frac{1}{2} \int_0^t \beta(s) ds\right)$。

---

**反向SDE**：

$$
dx = \left[-\frac{1}{2} \beta(t) x - \beta(t) \nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)} d\bar{w}
$$

---

### 2. 概率流ODE

**定理 2.1 (概率流ODE, Song et al. 2021)**:

与SDE具有相同边际分布的ODE：

$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)
$$

**VP-SDE的概率流ODE**：

$$
\frac{dx}{dt} = -\frac{1}{2} \beta(t) [x + \nabla_x \log p_t(x)]
$$

**优势**：

- 确定性采样
- 精确似然计算
- 可逆性

---

## 💻 Python实现

```python
import torch
import torch.nn as nn
import numpy as np

class DiffusionModel(nn.Module):
    """简化的扩散模型"""
    def __init__(self, data_dim=2, hidden_dim=128, T=1000):
        super().__init__()
        self.T = T
        
        # 噪声调度
        self.beta = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Score网络 (简化为MLP)
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x, t):
        """预测噪声"""
        # 时间嵌入 (简化)
        t_emb = t.float() / self.T
        x_t = torch.cat([x, t_emb.unsqueeze(-1)], dim=-1)
        return self.net(x_t)
    
    def q_sample(self, x0, t, noise=None):
        """前向扩散采样 q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    def p_sample(self, xt, t):
        """反向扩散采样 p(x_{t-1} | x_t)"""
        # 预测噪声
        eps_pred = self.forward(xt, t)
        
        # 计算均值
        alpha_t = self.alpha[t].view(-1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        
        mean = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_t)
        
        # 添加噪声
        if t[0] > 0:
            noise = torch.randn_like(xt)
            sigma = torch.sqrt(self.beta[t].view(-1, 1))
            return mean + sigma * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, batch_size, data_dim):
        """从噪声生成样本"""
        # 从纯噪声开始
        x = torch.randn(batch_size, data_dim)
        
        # 反向扩散
        for t in reversed(range(self.T)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        
        return x


def train_diffusion(model, data_loader, epochs=100, lr=1e-3):
    """训练扩散模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for x0 in data_loader:
            # 随机时间步
            t = torch.randint(0, model.T, (x0.shape[0],))
            
            # 采样噪声
            noise = torch.randn_like(x0)
            
            # 前向扩散
            xt = model.q_sample(x0, t, noise)
            
            # 预测噪声
            noise_pred = model.forward(xt, t)
            
            # 损失 (简单MSE)
            loss = nn.MSELoss()(noise_pred, noise)
            
            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.6f}")


# 示例：2D高斯混合
def generate_2d_data(n_samples=1000):
    """生成2D高斯混合数据"""
    centers = torch.tensor([[2, 2], [-2, -2], [2, -2], [-2, 2]])
    n_per_center = n_samples // 4
    
    data = []
    for center in centers:
        samples = torch.randn(n_per_center, 2) * 0.5 + center
        data.append(samples)
    
    return torch.cat(data, dim=0)


# 训练
data = generate_2d_data(1000)
data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

model = DiffusionModel(data_dim=2, T=100)
train_diffusion(model, data_loader, epochs=100)

# 生成
samples = model.sample(batch_size=100, data_dim=2)
print(f"Generated samples shape: {samples.shape}")
```

---

## 📚 核心模型

| 模型 | 核心思想 | 特点 |
| ---- |----------| ---- |
| **DDPM** | 离散时间扩散 | 简单稳定 |
| **Score-Based** | 连续时间SDE | 理论优雅 |
| **DDIM** | 确定性采样 | 快速生成 |
| **Latent Diffusion** | 潜空间扩散 | 高效训练 |

---

## 🎓 相关课程

| 大学 | 课程 |
| ---- |------|
| **Stanford** | CS236 Deep Generative Models |
| **MIT** | 6.S191 Introduction to Deep Learning |
| **UC Berkeley** | CS294 Deep Unsupervised Learning |

---

## 📖 参考文献

1. **Ho et al. (2020)**. "Denoising Diffusion Probabilistic Models". *NeurIPS*.

2. **Song et al. (2021)**. "Score-Based Generative Modeling through Stochastic Differential Equations". *ICLR*.

3. **Rombach et al. (2022)**. "High-Resolution Image Synthesis with Latent Diffusion Models". *CVPR*.

---

*最后更新：2025年10月*-
