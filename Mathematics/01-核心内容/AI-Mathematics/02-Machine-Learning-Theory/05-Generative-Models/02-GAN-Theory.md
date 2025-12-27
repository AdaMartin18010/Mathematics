# 生成对抗网络 (GAN) 理论

> **Generative Adversarial Networks: Theory and Mathematics**
>
> 对抗训练的数学基础与理论分析

---

## 目录

- [生成对抗网络 (GAN) 理论](#生成对抗网络-gan-理论)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 问题形式化](#-问题形式化)
    - [1. 对抗博弈](#1-对抗博弈)
    - [2. 目标函数](#2-目标函数)
  - [📊 理论分析](#-理论分析)
    - [1. 全局最优解](#1-全局最优解)
    - [2. Nash均衡](#2-nash均衡)
    - [3. 收敛性分析](#3-收敛性分析)
  - [🔬 训练动力学](#-训练动力学)
    - [1. 判别器更新](#1-判别器更新)
    - [2. 生成器更新](#2-生成器更新)
    - [3. 模式坍缩](#3-模式坍缩)
  - [💻 Python实现](#-python实现)
  - [🎨 GAN变体](#-gan变体)
    - [1. DCGAN](#1-dcgan)
    - [2. Wasserstein GAN (WGAN)](#2-wasserstein-gan-wgan)
    - [3. Conditional GAN (cGAN)](#3-conditional-gan-cgan)
  - [📚 理论深化](#-理论深化)
    - [1. f-散度视角](#1-f-散度视角)
    - [2. 积分概率度量 (IPM)](#2-积分概率度量-ipm)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**GAN**通过**对抗训练**学习生成数据分布。

**核心流程**：

```text
噪声 z → 生成器 G → 假样本 G(z)
                         ↓
真样本 x → 判别器 D → 真/假判断
                         ↓
                    反馈给G和D
```

**对抗过程**：

- **生成器 G**：欺骗判别器（生成逼真样本）
- **判别器 D**：区分真假样本

---

## 🎯 问题形式化

### 1. 对抗博弈

**生成器**：$G: \mathcal{Z} \to \mathcal{X}$

- 输入：噪声 $z \sim p_z(z)$（如 $\mathcal{N}(0, I)$）
- 输出：生成样本 $G(z)$

**判别器**：$D: \mathcal{X} \to [0, 1]$

- 输入：样本 $x$
- 输出：$D(x)$ = 样本为真的概率

---

### 2. 目标函数

**定理 2.1 (GAN目标函数, Goodfellow et al. 2014)**:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**解释**：

- **判别器D**：最大化 $V(D, G)$
  - 对真样本输出接近1：$\log D(x) \to 0$
  - 对假样本输出接近0：$\log(1 - D(G(z))) \to 0$

- **生成器G**：最小化 $V(D, G)$
  - 让假样本欺骗判别器：$D(G(z)) \to 1$

---

## 📊 理论分析

### 1. 全局最优解

**定理 1.1 (最优判别器)**:

对于固定的生成器 $G$，最优判别器为：

$$
D_G^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

其中 $p_g$ 是生成器诱导的分布。

**证明**：

目标是最大化：

$$
V(D, G) = \int_x p_{\text{data}}(x) \log D(x) dx + \int_x p_g(x) \log(1 - D(x)) dx
$$

对 $D(x)$ 求导并令其为0：

$$
\frac{p_{\text{data}}(x)}{D(x)} - \frac{p_g(x)}{1 - D(x)} = 0
$$

解得：

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

---

**定理 1.2 (全局最优生成器)**:

当 $D = D_G^*$ 时，$G$ 的目标函数等价于最小化：

$$
C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)
$$

其中 $\text{JSD}$ 是Jensen-Shannon散度。

**推论**：全局最优解为 $p_g = p_{\text{data}}$，此时 $D_G^*(x) = \frac{1}{2}$。

---

### 2. Nash均衡

**定义 2.1 (Nash均衡)**:

$(G^*, D^*)$ 是Nash均衡，若：

$$
\begin{align}
V(D^*, G^*) &\geq V(D, G^*) \quad \forall D \\
V(D^*, G^*) &\leq V(D^*, G) \quad \forall G
\end{align}
$$

**GAN的Nash均衡**：$p_g = p_{\text{data}}$，$D(x) = \frac{1}{2}$。

**问题**：实践中难以达到Nash均衡！

---

### 3. 收敛性分析

**定理 3.1 (收敛性, Goodfellow et al. 2014)**:

如果 $G$ 和 $D$ 有足够容量，且每步更新都能达到最优，则算法收敛到 $p_g = p_{\text{data}}$。

**实践问题**：

- 有限容量
- 有限更新步数
- 非凸优化
- 梯度消失

---

## 🔬 训练动力学

### 1. 判别器更新

**目标**：最大化 $V(D, G)$

**梯度**：

$$
\nabla_\theta V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\nabla_\theta \log D_\theta(x)\right] + \mathbb{E}_{z \sim p_z}\left[\nabla_\theta \log(1 - D_\theta(G(z)))\right]
$$

**更新**：

$$
\theta_D \leftarrow \theta_D + \alpha \nabla_\theta V(D, G)
$$

---

### 2. 生成器更新

**原始目标**：最小化 $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$

**问题**：早期训练时 $D(G(z)) \approx 0$，梯度消失。

**改进目标**：最大化 $\mathbb{E}_{z \sim p_z}[\log D(G(z))]$

**梯度**：

$$
\nabla_\phi \mathbb{E}_{z \sim p_z}[\log D(G_\phi(z))] = \mathbb{E}_{z \sim p_z}\left[\nabla_\phi \log D(G_\phi(z))\right]
$$

---

### 3. 模式坍缩

**问题**：生成器只生成少数几种样本。

**原因**：

- 生成器找到"捷径"欺骗判别器
- 缺乏多样性惩罚

**解决方案**：

1. **Unrolled GAN**：判别器前瞻多步
2. **Minibatch Discrimination**：考虑批内多样性
3. **Mode Regularization**：显式多样性正则化

---

## 💻 Python实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 生成器
class Generator(nn.Module):
    """简单的全连接生成器"""
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, z):
        return self.net(z)


# 判别器
class Discriminator(nn.Module):
    """简单的全连接判别器"""
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出概率
        )
    
    def forward(self, x):
        return self.net(x)


def train_gan(generator, discriminator, train_loader, epochs=50, lr=2e-4, latent_dim=100):
    """训练GAN"""
    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 损失函数
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1)
            
            # 真假标签
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            # ========== 训练判别器 ==========
            d_optimizer.zero_grad()
            
            # 真样本
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # 假样本
            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            # 总判别器损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ========== 训练生成器 ==========
            g_optimizer.zero_grad()
            
            # 生成假样本并欺骗判别器
            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images)
            
            # 生成器损失（希望判别器输出1）
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


def generate_samples(generator, n_samples=16, latent_dim=100):
    """生成样本"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        samples = generator(z)
        samples = samples.view(-1, 28, 28)
        samples = (samples + 1) / 2  # 从 [-1, 1] 转换到 [0, 1]
    return samples


# 示例：在MNIST上训练GAN
if __name__ == "__main__":
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 创建模型
    latent_dim = 100
    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()
    
    # 训练
    print("Training GAN...")
    train_gan(generator, discriminator, train_loader, epochs=50, latent_dim=latent_dim)
    
    # 生成样本
    print("\nGenerating samples...")
    samples = generate_samples(generator, n_samples=16, latent_dim=latent_dim)
    
    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].numpy(), cmap='gray')
        ax.axis('off')
    plt.suptitle('Generated Samples from GAN')
    plt.tight_layout()
    plt.show()
```

---

## 🎨 GAN变体

### 1. DCGAN

**Deep Convolutional GAN** (Radford et al. 2016)

**核心改进**：

- 使用卷积和转置卷积
- Batch Normalization
- LeakyReLU激活
- 移除全连接层

**架构指导**：

```text
生成器:
  - 转置卷积上采样
  - Batch Norm
  - ReLU (最后一层Tanh)

判别器:
  - 卷积下采样
  - Batch Norm
  - LeakyReLU
  - 最后一层Sigmoid
```

---

### 2. Wasserstein GAN (WGAN)

**核心思想**：用Wasserstein距离替代JS散度。

**Wasserstein距离**：

$$
W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]
$$

**Kantorovich-Rubinstein对偶**：

$$
W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]
$$

**WGAN目标**：

$$
\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

其中 $\mathcal{D}$ 是1-Lipschitz函数类。

**优势**：

- 更稳定的训练
- 有意义的损失曲线
- 缓解模式坍缩

**实现**：权重裁剪或梯度惩罚（WGAN-GP）。

---

### 3. Conditional GAN (cGAN)

**核心思想**：条件生成。

**模型**：

$$
\begin{align}
G(z, y) &\quad \text{(条件生成器)} \\
D(x, y) &\quad \text{(条件判别器)}
\end{align}
$$

**目标函数**：

$$
\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x, y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z, y), y))]
$$

**应用**：

- 类别条件图像生成
- 图像到图像翻译（Pix2Pix）
- 文本到图像生成

---

## 📚 理论深化

### 1. f-散度视角

**定理 1.1 (f-GAN, Nowozin et al. 2016)**:

GAN可以最小化任意f-散度：

$$
D_f(p_{\text{data}} \| p_g) = \int p_g(x) f\left(\frac{p_{\text{data}}(x)}{p_g(x)}\right) dx
$$

**常见f-散度**：

| f-散度 | $f(t)$ |
|--------|--------|
| **KL** | $t \log t$ |
| **JS** | $-\log(2) - \frac{1}{2}(t+1)\log\frac{t+1}{2}$ |
| **Total Variation** | $\frac{1}{2}\|t-1\|$ |

---

### 2. 积分概率度量 (IPM)

**定义 2.1 (IPM)**:

$$
d_{\mathcal{F}}(p, q) = \sup_{f \in \mathcal{F}} \left|\mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{x \sim q}[f(x)]\right|
$$

**例子**：

- **Wasserstein距离**：$\mathcal{F}$ = 1-Lipschitz函数
- **Maximum Mean Discrepancy (MMD)**：$\mathcal{F}$ = RKHS单位球

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS236 Deep Generative Models |
| **MIT** | 6.S191 Introduction to Deep Learning |
| **UC Berkeley** | CS294 Deep Unsupervised Learning |
| **NYU** | DS-GA 1008 Deep Learning |

---

## 📖 参考文献

1. **Goodfellow et al. (2014)**. "Generative Adversarial Networks". *NeurIPS*.

2. **Radford et al. (2016)**. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks". *ICLR*.

3. **Arjovsky et al. (2017)**. "Wasserstein GAN". *ICML*.

4. **Gulrajani et al. (2017)**. "Improved Training of Wasserstein GANs". *NeurIPS*.

5. **Nowozin et al. (2016)**. "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization". *NeurIPS*.

---

*最后更新：2025年10月*-
