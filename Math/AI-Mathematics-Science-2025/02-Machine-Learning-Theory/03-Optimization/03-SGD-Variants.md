# SGD及其变体 (SGD and Variants)

> **Stochastic Gradient Descent: From Theory to Practice**
>
> 深度学习优化的核心算法

---

## 目录

- [SGD及其变体 (SGD and Variants)](#sgd及其变体-sgd-and-variants)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 优化问题](#-优化问题)
    - [1. 经验风险最小化](#1-经验风险最小化)
    - [2. 批量梯度下降的问题](#2-批量梯度下降的问题)
  - [📊 随机梯度下降 (SGD)](#-随机梯度下降-sgd)
    - [1. 算法定义](#1-算法定义)
    - [2. 收敛性分析](#2-收敛性分析)
    - [3. 学习率调度](#3-学习率调度)
  - [🔬 动量方法 (Momentum)](#-动量方法-momentum)
    - [1. 标准动量](#1-标准动量)
    - [2. Nesterov加速梯度](#2-nesterov加速梯度)
    - [3. 动量的几何解释](#3-动量的几何解释)
  - [💻 自适应学习率方法](#-自适应学习率方法)
    - [1. AdaGrad](#1-adagrad)
    - [2. RMSprop](#2-rmsprop)
    - [3. Adam](#3-adam)
    - [4. AdamW](#4-adamw)
  - [🎨 学习率调度策略](#-学习率调度策略)
    - [1. 步长衰减](#1-步长衰减)
    - [2. 余弦退火](#2-余弦退火)
    - [3. 预热 (Warmup)](#3-预热-warmup)
    - [4. 循环学习率](#4-循环学习率)
  - [📐 批量大小的影响](#-批量大小的影响)
    - [1. 批量大小与泛化](#1-批量大小与泛化)
    - [2. 线性缩放规则](#2-线性缩放规则)
  - [🔧 实践技巧](#-实践技巧)
    - [1. 梯度裁剪](#1-梯度裁剪)
    - [2. 权重衰减](#2-权重衰减)
    - [3. 梯度累积](#3-梯度累积)
  - [💡 Python实现](#-python实现)
  - [📚 优化器选择指南](#-优化器选择指南)
    - [1. 不同任务的推荐](#1-不同任务的推荐)
    - [2. 超参数调优](#2-超参数调优)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**随机梯度下降 (SGD)** 使用**小批量样本**估计梯度，实现高效优化。

**核心原理**：

```text
批量梯度下降 (BGD):
    使用全部数据 → 准确但慢

随机梯度下降 (SGD):
    使用单个/小批量样本 → 快速但有噪声

关键权衡:
    计算效率 vs 梯度准确性
```

---

## 🎯 优化问题

### 1. 经验风险最小化

**目标**：

$$
\min_{\theta} \mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)
$$

**梯度**：

$$
\nabla \mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla \ell(f_\theta(x_i), y_i)
$$

---

### 2. 批量梯度下降的问题

**算法**：

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)
$$

**问题**：

- **计算成本高**：每步需要遍历全部数据
- **内存需求大**：需要存储所有样本
- **收敛慢**：大数据集下不实用

**示例**：

- 数据集：100万样本
- 每个epoch：100万次前向传播
- 训练100个epoch：1亿次计算！

---

## 📊 随机梯度下降 (SGD)

### 1. 算法定义

**定义 1.1 (Mini-batch SGD)**:

在每次迭代中：

1. 随机采样小批量 $\mathcal{B}_t \subset \{1, \ldots, n\}$，$|\mathcal{B}_t| = b$
2. 计算小批量梯度：
   $$
   g_t = \frac{1}{b} \sum_{i \in \mathcal{B}_t} \nabla \ell(f_{\theta_t}(x_i), y_i)
   $$
3. 更新参数：
   $$
   \theta_{t+1} = \theta_t - \eta_t g_t
   $$

**关键性质**：

$$
\mathbb{E}[g_t] = \nabla \mathcal{L}(\theta_t) \quad \text{(无偏估计)}
$$

---

### 2. 收敛性分析

**定理 2.1 (SGD收敛率, 凸情况)**:

假设 $\mathcal{L}$ 是 $L$-光滑的凸函数，使用固定学习率 $\eta \leq 1/L$：

$$
\mathbb{E}[\mathcal{L}(\bar{\theta}_T)] - \mathcal{L}^* \leq \frac{\|\theta_0 - \theta^*\|^2}{2\eta T} + \frac{\eta \sigma^2}{2b}
$$

其中 $\bar{\theta}_T = \frac{1}{T} \sum_{t=1}^{T} \theta_t$，$\sigma^2$ 是梯度方差。

**解释**：

- 第一项：优化误差，随 $T$ 减小
- 第二项：随机噪声，取决于批量大小

**收敛率**：$O(1/\sqrt{T})$

---

### 3. 学习率调度

**固定学习率问题**：

- 太大：振荡，不收敛
- 太小：收敛慢

**解决方案**：学习率衰减

**常见策略**：

1. **步长衰减**：$\eta_t = \eta_0 / (1 + \alpha t)$
2. **指数衰减**：$\eta_t = \eta_0 \gamma^t$
3. **多项式衰减**：$\eta_t = \eta_0 / (1 + t)^p$

---

## 🔬 动量方法 (Momentum)

### 1. 标准动量

**定义 1.1 (Momentum SGD)**:

$$
v_{t+1} = \beta v_t + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

其中 $\beta \in [0, 1)$ 是动量系数（通常 $\beta = 0.9$）。

**展开形式**：

$$
v_{t+1} = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots
$$

**直觉**：

- 累积历史梯度
- 加速一致方向
- 抑制振荡

---

### 2. Nesterov加速梯度

**定义 2.1 (Nesterov Accelerated Gradient, NAG)**:

$$
v_{t+1} = \beta v_t + \nabla \mathcal{L}(\theta_t - \eta \beta v_t)
$$

$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

**关键思想**：先"预测"一步，再计算梯度

**优势**：

- 更好的收敛性
- 凸情况：$O(1/T^2)$ vs 标准动量的 $O(1/T)$

---

### 3. 动量的几何解释

**物理类比**：

```text
梯度 = 力
动量 = 速度
参数 = 位置

物理方程:
    v_{t+1} = βv_t + F_t  (牛顿第二定律)
    x_{t+1} = x_t + v_{t+1}
```

**效果**：

- 在平坦区域加速
- 在陡峭区域减速
- 跨越局部极小值

---

## 💻 自适应学习率方法

### 1. AdaGrad

**定义 1.1 (AdaGrad)**:

$$
G_t = G_{t-1} + g_t \odot g_t
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
$$

**特点**：

- 频繁更新的参数 → 小学习率
- 稀疏更新的参数 → 大学习率

**问题**：学习率单调递减，可能过早停止

---

### 2. RMSprop

**定义 2.1 (RMSprop)**:

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t \odot g_t
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot g_t
$$

**改进**：

- 使用指数移动平均
- 避免学习率单调递减

**超参数**：$\beta = 0.9$, $\eta = 0.001$

---

### 3. Adam

**定义 3.1 (Adam - Adaptive Moment Estimation)**:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(一阶矩)}
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t \odot g_t \quad \text{(二阶矩)}
$$

**偏差修正**：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**更新**：

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

**默认超参数**：

- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\eta = 0.001$
- $\epsilon = 10^{-8}$

---

### 4. AdamW

**定义 4.1 (AdamW - Adam with Weight Decay)**:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

**关键改进**：

- 将权重衰减从梯度中解耦
- 更好的正则化效果

**推荐**：现代深度学习的首选

---

## 🎨 学习率调度策略

### 1. 步长衰减

**定义**：

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}
$$

其中 $s$ 是步长大小，$\gamma$ 是衰减因子（如0.1）。

**示例**：

- 初始：$\eta_0 = 0.1$
- 每30个epoch：$\eta \times 0.1$

---

### 2. 余弦退火

**定义 2.1 (Cosine Annealing)**:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{t}{T} \pi\right)\right)
$$

**特点**：

- 平滑衰减
- 无需手动调整步长

**变体：Cosine Annealing with Warm Restarts**:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i} \pi\right)\right)
$$

---

### 3. 预热 (Warmup)

**定义 3.1 (Linear Warmup)**:

$$
\eta_t = \begin{cases}
\frac{t}{T_{\text{warmup}}} \eta_0 & \text{if } t \leq T_{\text{warmup}} \\
\eta_0 & \text{otherwise}
\end{cases}
$$

**作用**：

- 避免初期大学习率导致的不稳定
- 特别适用于Transformer训练

---

### 4. 循环学习率

**定义 4.1 (Cyclical Learning Rate)**:

$$
\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \max(0, 1 - |t \bmod (2s) - s| / s)
$$

**特点**：

- 周期性变化
- 帮助跳出局部极小值

---

## 📐 批量大小的影响

### 1. 批量大小与泛化

**观察**：

- **小批量**：泛化更好，但训练慢
- **大批量**：训练快，但泛化差

**理论解释**：

- 小批量：梯度噪声 → 隐式正则化
- 大批量：收敛到尖锐极小值

---

### 2. 线性缩放规则

**定理 2.1 (Linear Scaling Rule, Goyal et al. 2017)**:

当批量大小增加 $k$ 倍时，学习率也应增加 $k$ 倍：

$$
\eta_{\text{new}} = k \cdot \eta_{\text{old}}
$$

**前提**：

- 使用预热
- 批量大小不能太大

**示例**：

- 批量256，学习率0.1
- 批量1024 → 学习率0.4

---

## 🔧 实践技巧

### 1. 梯度裁剪

**按范数裁剪**：

$$
g = \begin{cases}
\frac{c}{\|g\|} g & \text{if } \|g\| > c \\
g & \text{otherwise}
\end{cases}
$$

**作用**：防止梯度爆炸

---

### 2. 权重衰减

**L2正则化**：

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \|\theta\|^2
$$

**等价于**：

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta g_t
$$

---

### 3. 梯度累积

**动机**：模拟大批量，节省内存

**方法**：

```python
for i in range(accumulation_steps):
    loss = compute_loss(batch[i])
    loss.backward()  # 累积梯度

optimizer.step()  # 更新参数
optimizer.zero_grad()
```

**等价批量大小**：$b_{\text{eff}} = b \times \text{accumulation\_steps}$

---

## 💡 Python实现

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 从零实现SGD with Momentum
class SGDMomentum:
    """SGD with Momentum"""
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            # 更新速度
            self.velocities[i] = self.momentum * self.velocities[i] + param.grad
            
            # 更新参数
            param.data -= self.lr * self.velocities[i]
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# 2. 从零实现Adam
class Adam:
    """Adam Optimizer"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        self.m = [torch.zeros_like(p) for p in self.params]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in self.params]  # 二阶矩
        self.t = 0  # 时间步
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # 更新一阶矩和二阶矩
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# 3. 学习率调度器
class CosineAnnealingLR:
    """余弦退火学习率调度"""
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.t = 0
    
    def step(self):
        self.t += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + np.cos(np.pi * self.t / self.T_max)) / 2
        self.optimizer.lr = lr
        return lr


# 4. 对比不同优化器
def compare_optimizers():
    """对比不同优化器的性能"""
    # 定义一个简单的优化问题
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    # 初始点
    x0, y0 = -1.5, 2.0
    
    # 不同优化器
    optimizers = {
        'SGD': lambda p: torch.optim.SGD(p, lr=0.001),
        'SGD+Momentum': lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
        'Adam': lambda p: torch.optim.Adam(p, lr=0.01),
        'RMSprop': lambda p: torch.optim.RMSprop(p, lr=0.01),
    }
    
    trajectories = {}
    
    for name, opt_fn in optimizers.items():
        x = torch.tensor([x0, y0], requires_grad=True)
        optimizer = opt_fn([x])
        
        trajectory = [x.detach().numpy().copy()]
        
        for _ in range(200):
            optimizer.zero_grad()
            loss = rosenbrock(x[0], x[1])
            loss.backward()
            optimizer.step()
            
            trajectory.append(x.detach().numpy().copy())
        
        trajectories[name] = np.array(trajectory)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 绘制等高线
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock(X, Y)
    
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray', alpha=0.3)
    
    # 绘制轨迹
    colors = ['red', 'blue', 'green', 'orange']
    for (name, traj), color in zip(trajectories.items(), colors):
        plt.plot(traj[:, 0], traj[:, 1], '-o', label=name, color=color, 
                 markersize=2, linewidth=1.5)
    
    plt.plot(1, 1, 'r*', markersize=15, label='Optimum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimizer Comparison on Rosenbrock Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()


# 5. 学习率调度可视化
def visualize_lr_schedules():
    """可视化不同的学习率调度策略"""
    T = 100
    eta_0 = 0.1
    
    schedules = {
        'Constant': [eta_0] * T,
        'Step Decay': [eta_0 * (0.5 ** (t // 30)) for t in range(T)],
        'Exponential': [eta_0 * (0.95 ** t) for t in range(T)],
        'Cosine': [eta_0 * (1 + np.cos(np.pi * t / T)) / 2 for t in range(T)],
        'Linear Warmup': [min(t / 10, 1) * eta_0 for t in range(T)],
    }
    
    plt.figure(figsize=(12, 6))
    for name, schedule in schedules.items():
        plt.plot(schedule, label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()


# 示例使用
if __name__ == "__main__":
    print("=== 对比优化器 ===")
    compare_optimizers()
    
    print("\n=== 学习率调度 ===")
    visualize_lr_schedules()
    
    # 测试自定义Adam
    print("\n=== 测试自定义Adam ===")
    model = nn.Linear(10, 1)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    for epoch in range(10):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 📚 优化器选择指南

### 1. 不同任务的推荐

| 任务 | 推荐优化器 | 学习率 |
|------|-----------|--------|
| **图像分类** | SGD+Momentum | 0.1 |
| **目标检测** | SGD+Momentum | 0.02 |
| **语言模型** | Adam/AdamW | 1e-4 |
| **Transformer** | AdamW + Warmup | 1e-4 |
| **GAN** | Adam | 2e-4 |
| **强化学习** | Adam | 3e-4 |

---

### 2. 超参数调优

**学习率**：

- 从大到小尝试：$[1, 0.1, 0.01, 0.001, 0.0001]$
- 使用学习率查找器

**批量大小**：

- 从小开始：32, 64, 128, 256
- 受内存限制

**动量**：

- 默认0.9通常有效
- 可尝试0.95, 0.99

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS229 Machine Learning |
| **MIT** | 6.255J Optimization Methods |
| **CMU** | 10-725 Convex Optimization |
| **UC Berkeley** | CS189 Introduction to Machine Learning |

---

## 📖 参考文献

1. **Robbins & Monro (1951)**. "A Stochastic Approximation Method". *Annals of Mathematical Statistics*.

2. **Polyak (1964)**. "Some methods of speeding up the convergence of iteration methods". *USSR Computational Mathematics and Mathematical Physics*.

3. **Nesterov (1983)**. "A method for solving the convex programming problem with convergence rate O(1/k^2)". *Soviet Mathematics Doklady*.

4. **Duchi et al. (2011)**. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization". *JMLR*. (AdaGrad)

5. **Kingma & Ba (2015)**. "Adam: A Method for Stochastic Optimization". *ICLR*.

6. **Loshchilov & Hutter (2019)**. "Decoupled Weight Decay Regularization". *ICLR*. (AdamW)

7. **Goyal et al. (2017)**. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour". *arXiv*. (Linear Scaling Rule)

---

*最后更新：2025年10月*-
