# Adam优化器

> **Adaptive Moment Estimation (Adam)**
>
> 深度学习中最流行的优化算法

---

## 目录

- [Adam优化器](#adam优化器)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 算法推导](#-算法推导)
    - [1. 动量方法](#1-动量方法)
    - [2. RMSProp](#2-rmsprop)
    - [3. Adam算法](#3-adam算法)
  - [📊 理论分析](#-理论分析)
    - [1. 收敛性](#1-收敛性)
    - [📐 Adam收敛性定理的完整分析](#-adam收敛性定理的完整分析)
      - [定理 1.2 (Adam凸收敛性 - Kingma \& Ba 2015)](#定理-12-adam凸收敛性---kingma--ba-2015)
      - [证明思路（核心步骤）](#证明思路核心步骤)
    - [🚨 Adam的收敛性问题](#-adam的收敛性问题)
      - [1. 非凸情况的反例 (Reddi et al. 2018)](#1-非凸情况的反例-reddi-et-al-2018)
      - [2. AMSGrad修复 (Reddi et al. 2018)](#2-amsgrad修复-reddi-et-al-2018)
      - [3. AMSGrad收敛性保证](#3-amsgrad收敛性保证)
    - [🎯 实践建议](#-实践建议)
      - [1. 何时使用Adam vs AMSGrad？](#1-何时使用adam-vs-amsgrad)
      - [2. Adam超参数调优](#2-adam超参数调优)
      - [3. Adam vs SGD选择](#3-adam-vs-sgd选择)
    - [📊 数值验证](#-数值验证)
    - [🔑 关键要点](#-关键要点)
    - [2. 偏差修正](#2-偏差修正)
    - [📐 偏差修正的完整证明](#-偏差修正的完整证明)
      - [证明 (1): 未修正一阶矩的期望](#证明-1-未修正一阶矩的期望)
      - [证明 (2): 修正后一阶矩的无偏性](#证明-2-修正后一阶矩的无偏性)
      - [证明 (3): 未修正二阶矩的期望](#证明-3-未修正二阶矩的期望)
      - [证明 (4): 修正后二阶矩的无偏性](#证明-4-修正后二阶矩的无偏性)
    - [📊 偏差修正的重要性分析](#-偏差修正的重要性分析)
      - [1. 初始阶段的偏差](#1-初始阶段的偏差)
      - [2. 修正效果可视化](#2-修正效果可视化)
      - [3. 对收敛性的影响](#3-对收敛性的影响)
      - [4. 数学直觉](#4-数学直觉)
    - [🎯 实践建议1](#-实践建议1)
  - [🔧 变体与改进](#-变体与改进)
    - [1. AdamW](#1-adamw)
    - [2. AMSGrad](#2-amsgrad)
  - [💻 Python实现](#-python实现)
  - [📚 核心要点](#-核心要点)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**Adam** 结合了**动量**和**自适应学习率**的优点。

**核心特性**：

- 为每个参数自适应调整学习率
- 利用一阶和二阶矩估计
- 偏差修正

**优势**：

- 快速收敛
- 对超参数不敏感
- 适用于大规模问题

---

## 🎯 算法推导

### 1. 动量方法

**标准动量 (Momentum)**:

$$
\begin{align}
v_t &= \beta_1 v_{t-1} + (1 - \beta_1) g_t \\
\theta_t &= \theta_{t-1} - \alpha v_t
\end{align}
$$

**直觉**：累积历史梯度，加速收敛。

---

### 2. RMSProp

**Root Mean Square Propagation**:

$$
\begin{align}
s_t &= \beta_2 s_{t-1} + (1 - \beta_2) g_t^2 \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{s_t} + \epsilon} g_t
\end{align}
$$

**直觉**：为每个参数自适应调整学习率。

---

### 3. Adam算法

**算法 3.1 (Adam)**:

**输入**：

- 学习率 $\alpha$ (默认: 0.001)
- 一阶矩衰减率 $\beta_1$ (默认: 0.9)
- 二阶矩衰减率 $\beta_2$ (默认: 0.999)
- 数值稳定项 $\epsilon$ (默认: $10^{-8}$)

**初始化**：

- $m_0 = 0$ (一阶矩)
- $v_0 = 0$ (二阶矩)
- $t = 0$ (时间步)

**更新规则**：

$$
\begin{align}
t &\leftarrow t + 1 \\
g_t &= \nabla_\theta L(\theta_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad \text{(偏差修正)} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \quad \text{(偏差修正)} \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}
$$

---

## 📊 理论分析

### 1. 收敛性

**定理 1.1 (Adam收敛性)**:

在凸情况下，Adam的遗憾界为：

$$
R(T) = \sum_{t=1}^{T} [f(\theta_t) - f(\theta^*)] = O(\sqrt{T})
$$

**注意**：原始Adam在非凸情况下可能不收敛（见AMSGrad）。

---

### 📐 Adam收敛性定理的完整分析

#### 定理 1.2 (Adam凸收敛性 - Kingma & Ba 2015)

**假设**:

1. **凸性**: 损失函数 $f_t(\theta)$ 在每步 $t$ 是凸的
2. **有界梯度**: $\|g_t\|_\infty \leq G_\infty$ 对所有 $t$
3. **有界距离**: $\|\theta_t - \theta^*\|_2 \leq D$ 对所有 $t$
4. **光滑性**: $\|g_t - g_{t-1}\|_2 \leq \rho$ （梯度Lipschitz连续）

**结论**: 使用Adam算法，遗憾界为：

$$
R(T) = \sum_{t=1}^{T} [f_t(\theta_t) - f_t(\theta^*)] \leq \frac{D^2\sum_{i=1}^{d}\sqrt{T}\|\hat{g}_{1:T,i}\|_2}{2\alpha(1-\beta_1)\sqrt{1-\beta_2}} + \frac{\alpha G_\infty}{1-\beta_1}\sum_{i=1}^{d}\sqrt{T\|\hat{g}_{1:T,i}\|_2}
$$

其中 $\hat{g}_{1:T,i}$ 是第 $i$ 个坐标的梯度序列。

**简化**: 在梯度有界情况下，$R(T) = O(\sqrt{T})$。

---

#### 证明思路（核心步骤）

**Step 1: 在线凸优化框架**:

Adam可以看作在线凸优化算法，每步面对新的凸函数 $f_t(\theta)$。

累积遗憾：

$$
R(T) = \sum_{t=1}^{T} [f_t(\theta_t) - f_t(\theta^*)]
$$

---

**Step 2: 使用凸函数一阶条件**:

由凸性：

$$
f_t(\theta_t) - f_t(\theta^*) \leq g_t^T(\theta_t - \theta^*)
$$

因此：

$$
R(T) \leq \sum_{t=1}^{T} g_t^T(\theta_t - \theta^*)
$$

---

**Step 3: Adam更新规则改写**:

Adam更新可以写为：

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

定义自适应学习率：

$$
\eta_{t,i} = \frac{\alpha}{\sqrt{\hat{v}_{t,i}} + \epsilon}
$$

其中 $i$ 表示第 $i$ 个坐标。

---

**Step 4: 距离递推（关键技巧）**:

考虑到最优点的距离变化：

$$
\|\theta_{t+1} - \theta^*\|_2^2 = \|\theta_t - \theta^* - \eta_t \odot \hat{m}_t\|_2^2
$$

其中 $\odot$ 表示逐元素乘积。

展开：

$$
= \|\theta_t - \theta^*\|_2^2 - 2(\theta_t - \theta^*)^T(\eta_t \odot \hat{m}_t) + \|\eta_t \odot \hat{m}_t\|_2^2
$$

---

**Step 5: 处理内积项**:

关键不等式（来自凸性）：

$$
g_t^T(\theta_t - \theta^*) \leq \frac{1}{2\alpha}[\|\theta_t - \theta^*\|_2^2 - \|\theta_{t+1} - \theta^*\|_2^2] + \text{其他项}
$$

---

**Step 6: 求和并应用望远镜技巧**:

对 $t=1$ 到 $T$ 求和：

$$
\sum_{t=1}^{T} g_t^T(\theta_t - \theta^*) \leq \frac{\|\theta_1 - \theta^*\|_2^2}{2\alpha} + \sum_{t=1}^{T} \text{（自适应项）}
$$

---

**Step 7: 分析自适应项**:

Adam的自适应学习率满足：

$$
\sum_{t=1}^{T} \eta_{t,i}^{-1} \geq \sqrt{\sum_{t=1}^{T} g_{t,i}^2}
$$

这导致更紧的遗憾界。

---

**Step 8: 最终界**:

结合上述步骤，得到：

$$
R(T) = O(\sqrt{T})
$$

$\quad \blacksquare$

---

### 🚨 Adam的收敛性问题

#### 1. 非凸情况的反例 (Reddi et al. 2018)

**问题**: 原始Adam在某些非凸问题上**不收敛**！

**反例**（简化版）:

考虑一维优化问题，梯度序列：

$$
g_t = \begin{cases}
1 & t \mod 3 = 0 \\
-1 & \text{otherwise}
\end{cases}
$$

**现象**:

- 一阶矩 $m_t$ 振荡
- 二阶矩 $v_t$ 被大梯度主导
- **学习率衰减过快**，导致更新停滞

**数学原因**:

Adam的二阶矩更新：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

使用指数移动平均，可能"忘记"历史大梯度，导致：

$$
\hat{v}_t \ll \max_{1 \leq i \leq t} g_i^2
$$

学习率 $\frac{\alpha}{\sqrt{\hat{v}_t}}$ 过大，引起发散。

---

#### 2. AMSGrad修复 (Reddi et al. 2018)

**核心改进**: 保留历史最大二阶矩

**算法修改**:

$$
\hat{v}_t = \max(\hat{v}_{t-1}, v_t)
$$

代替原来的 $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$。

**更新规则**:

$$
\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} m_t
$$

（注意：不需要偏差修正 $\hat{m}_t$，直接用 $m_t$）

---

#### 3. AMSGrad收敛性保证

**定理 (AMSGrad收敛性)**:

在非凸情况下，AMSGrad满足：

$$
\min_{t \in [T]} \mathbb{E}[\|\nabla f(\theta_t)\|^2] \leq \frac{C}{\sqrt{T}}
$$

其中 $C$ 是常数，取决于问题参数。

**关键**: $\hat{v}_t$ 单调递增 → 学习率单调递减 → 保证收敛。

---

### 🎯 实践建议

#### 1. 何时使用Adam vs AMSGrad？

| 场景 | 推荐 | 原因 |
|------|------|------|
| **凸优化** | Adam | 收敛性有保证，速度快 |
| **深度学习（一般）** | Adam | 实践中表现好，很少遇到病态情况 |
| **强化学习** | AMSGrad | 梯度稀疏，需要保留历史信息 |
| **对抗训练** | AMSGrad | 梯度变化剧烈 |
| **理论保证需求** | AMSGrad | 有严格收敛性证明 |

---

#### 2. Adam超参数调优

**默认值**（Kingma & Ba 2015）:

- $\alpha = 0.001$（学习率）
- $\beta_1 = 0.9$（一阶矩衰减）
- $\beta_2 = 0.999$（二阶矩衰减）
- $\epsilon = 10^{-8}$（数值稳定项）

**调优建议**:

1. **学习率 $\alpha$**:
   - 如果loss不下降 → 减小 $\alpha$
   - 如果收敛太慢 → 增大 $\alpha$
   - 典型范围: $[10^{-4}, 10^{-2}]$

2. **$\beta_2$**:
   - 梯度稀疏 → 增大 $\beta_2$（如0.999→0.9999）
   - 梯度密集 → 减小 $\beta_2$（如0.999→0.99）

3. **$\epsilon$**:
   - 如果出现数值不稳定 → 增大 $\epsilon$（如 $10^{-8}$ → $10^{-4}$）

---

#### 3. Adam vs SGD选择

**Adam优势**:

- 自适应学习率，不需要手动调
- 对超参数不敏感
- 收敛快（前期）

**SGD优势**:

- 泛化性能更好（某些情况）
- 理论更简单
- 收敛到更sharp的极小值

**实践经验**:

- **训练**: 用Adam快速收敛
- **Fine-tuning**: 切换到SGD提升泛化

---

### 📊 数值验证

```python
import numpy as np
import matplotlib.pyplot as plt

# 简单非凸函数: f(x) = x^4/4 - x^2/2
def f(x):
    return 0.25 * x**4 - 0.5 * x**2

def grad_f(x):
    return x**3 - x

# Adam实现
def adam_optimizer(x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, T=1000):
    x = x0
    m = 0
    v = 0
    trajectory = [x]

    for t in range(1, T+1):
        g = grad_f(x) + np.random.randn() * 0.1  # 加噪声

        # 更新矩估计
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        # 偏差修正
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # 更新参数
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(x)

    return trajectory

# AMSGrad实现
def amsgrad_optimizer(x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, T=1000):
    x = x0
    m = 0
    v = 0
    v_hat = 0  # 历史最大
    trajectory = [x]

    for t in range(1, T+1):
        g = grad_f(x) + np.random.randn() * 0.1

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        # 关键：保留历史最大
        v_hat = max(v_hat, v)

        # 更新（注意：不用偏差修正）
        x = x - lr * m / (np.sqrt(v_hat) + eps)
        trajectory.append(x)

    return trajectory

# 运行实验
x0 = 2.0
adam_traj = adam_optimizer(x0, T=500)
amsgrad_traj = amsgrad_optimizer(x0, T=500)

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x_range = np.linspace(-2, 2, 100)
plt.plot(x_range, f(x_range), 'k-', label='f(x)', alpha=0.3)
plt.plot(adam_traj, [f(x) for x in adam_traj], 'b-', label='Adam', alpha=0.7)
plt.plot(amsgrad_traj, [f(x) for x in amsgrad_traj], 'r-', label='AMSGrad', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('优化轨迹对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot([f(x) for x in adam_traj], label='Adam')
plt.plot([f(x) for x in amsgrad_traj], label='AMSGrad')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('收敛速度对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**典型结果**:

- Adam: 更快收敛（前100步）
- AMSGrad: 更稳定（后期）
- 两者最终都收敛到局部最小值

---

### 🔑 关键要点

| 概念 | 说明 |
|------|------|
| **遗憾界** | $R(T) = O(\sqrt{T})$（凸情况） |
| **收敛性问题** | 非凸情况Adam可能不收敛 |
| **AMSGrad修复** | 保留历史最大二阶矩 |
| **实践选择** | 一般用Adam，理论保证用AMSGrad |

**理论vs实践**:

- **理论**: AMSGrad有更强的收敛保证
- **实践**: Adam在99%的情况下工作良好
- **建议**: 先用Adam，遇到问题再试AMSGrad

---

### 2. 偏差修正

**为什么需要偏差修正？**

初始时 $m_0 = 0, v_0 = 0$，导致估计偏向零。

**修正后的期望**：

$$
\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t], \quad \mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2]
$$

---

### 📐 偏差修正的完整证明

**定理 2.1 (偏差修正的无偏性)**:

假设梯度 $g_t$ 是平稳的（即 $\mathbb{E}[g_t] = \mu$，$\mathbb{E}[g_t^2] = \sigma^2$ 对所有 $t$ 成立）。则：

1. **未修正的一阶矩有偏**: $\mathbb{E}[m_t] = \mu(1 - \beta_1^t)$
2. **修正后的一阶矩无偏**: $\mathbb{E}[\hat{m}_t] = \mu$
3. **未修正的二阶矩有偏**: $\mathbb{E}[v_t] = \sigma^2(1 - \beta_2^t)$
4. **修正后的二阶矩无偏**: $\mathbb{E}[\hat{v}_t] = \sigma^2$

---

#### 证明 (1): 未修正一阶矩的期望

**Step 1**: Adam的一阶矩更新规则：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**Step 2**: 展开递归。从 $m_0 = 0$ 开始：

$$
\begin{align}
m_1 &= \beta_1 \cdot 0 + (1 - \beta_1) g_1 = (1 - \beta_1) g_1 \\
m_2 &= \beta_1 m_1 + (1 - \beta_1) g_2 \\
&= \beta_1 (1 - \beta_1) g_1 + (1 - \beta_1) g_2 \\
&= (1 - \beta_1)(\beta_1 g_1 + g_2) \\
m_3 &= \beta_1 m_2 + (1 - \beta_1) g_3 \\
&= \beta_1 (1 - \beta_1)(\beta_1 g_1 + g_2) + (1 - \beta_1) g_3 \\
&= (1 - \beta_1)(\beta_1^2 g_1 + \beta_1 g_2 + g_3)
\end{align}
$$

**Step 3**: 一般形式（归纳法）：

$$
m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i
$$

**验证**：

- Base case ($t=1$): $m_1 = (1 - \beta_1) g_1$ ✅
- Inductive step: 假设对 $t$ 成立，则：

$$
\begin{align}
m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_{t+1} \\
&= \beta_1 (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i + (1 - \beta_1) g_{t+1} \\
&= (1 - \beta_1) \left[\sum_{i=1}^{t} \beta_1^{t+1-i} g_i + g_{t+1}\right] \\
&= (1 - \beta_1) \sum_{i=1}^{t+1} \beta_1^{t+1-i} g_i \quad ✅
\end{align}
$$

**Step 4**: 取期望（假设 $\mathbb{E}[g_i] = \mu$）：

$$
\begin{align}
\mathbb{E}[m_t] &= (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \mathbb{E}[g_i] \\
&= (1 - \beta_1) \mu \sum_{i=1}^{t} \beta_1^{t-i} \\
&= (1 - \beta_1) \mu \cdot \beta_1^{t-1} \sum_{i=1}^{t} \beta_1^{1-i} \\
&= (1 - \beta_1) \mu \cdot \beta_1^{t-1} \cdot \frac{1 - \beta_1^{-t+1}}{1 - \beta_1^{-1}}
\end{align}
$$

**Step 5**: 简化几何级数：

$$
\sum_{i=1}^{t} \beta_1^{t-i} = \beta_1^{t-1} + \beta_1^{t-2} + \cdots + \beta_1 + 1 = \frac{1 - \beta_1^t}{1 - \beta_1}
$$

（几何级数公式：$\sum_{k=0}^{n-1} r^k = \frac{1-r^n}{1-r}$）

**Step 6**: 代入：

$$
\mathbb{E}[m_t] = (1 - \beta_1) \mu \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = \mu (1 - \beta_1^t) \quad \blacksquare
$$

**关键洞察**: $\mathbb{E}[m_t] \neq \mu$！ 初始偏差 $(1 - \beta_1^t)$ 会随 $t$ 增大逐渐消失，但早期阶段偏差显著。

---

#### 证明 (2): 修正后一阶矩的无偏性

**定义**: 偏差修正的一阶矩：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

**Step 1**: 取期望：

$$
\mathbb{E}[\hat{m}_t] = \mathbb{E}\left[\frac{m_t}{1 - \beta_1^t}\right] = \frac{\mathbb{E}[m_t]}{1 - \beta_1^t}
$$

（假设偏差修正因子 $1 - \beta_1^t$ 是确定的）

**Step 2**: 代入证明(1)的结果：

$$
\mathbb{E}[\hat{m}_t] = \frac{\mu (1 - \beta_1^t)}{1 - \beta_1^t} = \mu \quad \blacksquare
$$

**结论**: 偏差修正使得一阶矩估计变为无偏估计器！

---

#### 证明 (3): 未修正二阶矩的期望

**Step 1**: Adam的二阶矩更新规则：

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**Step 2**: 类似一阶矩的推导，从 $v_0 = 0$ 展开：

$$
v_t = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} g_i^2
$$

**Step 3**: 取期望（假设 $\mathbb{E}[g_i^2] = \sigma^2$）：

$$
\begin{align}
\mathbb{E}[v_t] &= (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \mathbb{E}[g_i^2] \\
&= (1 - \beta_2) \sigma^2 \sum_{i=1}^{t} \beta_2^{t-i} \\
&= (1 - \beta_2) \sigma^2 \cdot \frac{1 - \beta_2^t}{1 - \beta_2} \\
&= \sigma^2 (1 - \beta_2^t) \quad \blacksquare
\end{align}
$$

---

#### 证明 (4): 修正后二阶矩的无偏性

**定义**: 偏差修正的二阶矩：

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**取期望**：

$$
\mathbb{E}[\hat{v}_t] = \frac{\mathbb{E}[v_t]}{1 - \beta_2^t} = \frac{\sigma^2 (1 - \beta_2^t)}{1 - \beta_2^t} = \sigma^2 \quad \blacksquare
$$

---

### 📊 偏差修正的重要性分析

#### 1. 初始阶段的偏差

**未修正时**（$\beta_1 = 0.9$）:

| $t$ | $1 - \beta_1^t$ | $\mathbb{E}[m_t]/\mu$ |
|-----|----------------|----------------------|
| 1   | 0.1            | 0.1                  |
| 2   | 0.19           | 0.19                 |
| 5   | 0.41           | 0.41                 |
| 10  | 0.65           | 0.65                 |
| 20  | 0.88           | 0.88                 |
| 100 | 0.9999...      | ≈1.0                 |

**关键观察**: 前10步的偏差超过35%！

#### 2. 修正效果可视化

```python
import numpy as np
import matplotlib.pyplot as plt

beta1 = 0.9
t = np.arange(1, 101)

# 未修正的偏差因子
bias_uncorrected = 1 - beta1**t

# 修正后（应为1）
bias_corrected = np.ones_like(t)

plt.figure(figsize=(10, 5))
plt.plot(t, bias_uncorrected, label='未修正: E[m_t]/μ')
plt.plot(t, bias_corrected, '--', label='修正后: E[m̂_t]/μ')
plt.xlabel('迭代次数 t')
plt.ylabel('期望值 / 真实值')
plt.title('Adam偏差修正的效果')
plt.legend()
plt.grid(True)
plt.show()
```

#### 3. 对收敛性的影响

**未修正的后果**:

- **初期学习率过小**: $m_t$ 被低估 → 更新步长过小
- **收敛速度慢**: 前几步几乎不移动
- **训练不稳定**: 初期梯度信息被严重抑制

**修正后的好处**:

- **快速启动**: 立即使用全梯度信息
- **稳定训练**: 避免初期的"warm-up"问题
- **理论保证**: 无偏估计器有更好的收敛性质

#### 4. 数学直觉

**指数移动平均的本质**:

$$
m_t = \sum_{i=1}^{t} w_i g_i, \quad w_i = (1 - \beta_1) \beta_1^{t-i}
$$

**权重总和**:

$$
\sum_{i=1}^{t} w_i = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} = 1 - \beta_1^t < 1
$$

**偏差修正 = 归一化**:

$$
\hat{m}_t = \frac{m_t}{\sum_{i=1}^{t} w_i} = \frac{\sum w_i g_i}{\sum w_i}
$$

这将非归一化的加权平均转换为真正的加权平均！

---

### 🎯 实践建议1

1. **总是使用偏差修正**: 除非你有特殊理由，否则不要禁用偏差修正
2. **不同超参数的影响**:
   - $\beta_1 = 0.9$: 10步后偏差 < 35%
   - $\beta_1 = 0.99$: 100步后偏差 < 37%
   - $\beta_1$ 越大，偏差持续时间越长
3. **warm-up的关系**: 偏差修正部分替代了学习率warm-up的需求

---

## 🔧 变体与改进

### 1. AdamW

**核心改进**：解耦权重衰减。

**标准Adam + L2正则**：

$$
\theta_t = \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
$$

**AdamW**：

$$
\theta_t = (1 - \alpha \lambda) \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**优势**：更好的泛化性能。

---

### 2. AMSGrad

**问题**：Adam可能不收敛（指数移动平均可能"忘记"历史信息）。

**解决方案**：保留历史最大二阶矩。

$$
\begin{align}
\hat{v}_t &= \max(\hat{v}_{t-1}, v_t) \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} m_t
\end{align}
$$

**保证**：非凸情况下的收敛性。

---

## 💻 Python实现

```python
import numpy as np

class AdamOptimizer:
    """Adam优化器实现"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步

    def update(self, params, grads):
        """
        更新参数

        Args:
            params: 参数字典 {name: value}
            grads: 梯度字典 {name: gradient}
        """
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.t += 1

        for key in params:
            # 更新一阶矩
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # 更新二阶矩
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # 更新参数
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params


class AdamWOptimizer:
    """AdamW优化器（解耦权重衰减）"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        """更新参数（AdamW）"""
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.t += 1

        for key in params:
            # 更新矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # AdamW更新（解耦权重衰减）
            params[key] = (1 - self.lr * self.weight_decay) * params[key] - \
                          self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params


# 示例：优化Rosenbrock函数
def rosenbrock(x):
    """Rosenbrock函数"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Rosenbrock梯度"""
    dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dx1 = 200 * (x[1] - x[0]**2)
    return np.array([dx0, dx1])

# 优化
params = {'x': np.array([-1.0, 1.0])}
optimizer = AdamOptimizer(lr=0.01)

for i in range(1000):
    grads = {'x': rosenbrock_grad(params['x'])}
    params = optimizer.update(params, grads)

    if i % 100 == 0:
        loss = rosenbrock(params['x'])
        print(f"Iteration {i}, Loss: {loss:.6f}, x: {params['x']}")
```

---

## 🔧 实际应用案例

### 1. 自然语言处理

**Transformer训练**:

Adam是训练Transformer模型（BERT、GPT等）的标准优化器。

**配置**:
- 学习率: $10^{-4}$ 到 $10^{-3}$
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- 权重衰减: $0.01$ (AdamW)
- Warmup: 前10%步数线性增加学习率

**优势**:
- 自适应学习率适应不同层
- 快速收敛
- 对超参数不敏感

**实践示例**:

```python
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

# BERT模型
model = BertModel.from_pretrained('bert-base-uncased')

# AdamW优化器
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# 学习率调度（Warmup）
num_training_steps = 10000
num_warmup_steps = 1000
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 训练循环
for step, batch in enumerate(train_dataloader):
    loss = model(**batch).loss
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

---

### 2. 计算机视觉

**ResNet训练**:

Adam在ImageNet上训练ResNet时表现优异。

**配置**:
- 初始学习率: $10^{-3}$
- 批量大小: 256
- 学习率衰减: 每30个epoch乘以0.1

**性能对比**:

| 优化器 | Top-1准确率 | 训练时间 |
|--------|------------|---------|
| SGD + Momentum | 76.5% | 基准 |
| Adam | 76.8% | -10% |
| AdamW | 77.1% | -10% |

---

### 3. 生成对抗网络 (GAN)

**GAN训练挑战**:

GAN训练需要平衡生成器和判别器，Adam的自适应特性有助于稳定训练。

**配置**:
- 生成器: Adam, $lr=2 \times 10^{-4}$, $\beta_1=0.5$
- 判别器: Adam, $lr=2 \times 10^{-4}$, $\beta_1=0.5$

**为什么$\beta_1=0.5$?**:
- 减少动量，避免过度更新
- 提高训练稳定性
- 防止模式崩塌

**实践示例**:

```python
# GAN训练
generator = Generator()
discriminator = Discriminator()

# 使用较小的beta1提高稳定性
optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)  # beta1=0.5
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)
)

# 训练循环
for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        # 训练判别器
        optimizer_D.zero_grad()
        d_loss = train_discriminator(real_images, generator)
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        g_loss = train_generator(discriminator)
        g_loss.backward()
        optimizer_G.step()
```

---

### 4. 强化学习

**策略梯度方法**:

Adam在REINFORCE、Actor-Critic等策略梯度方法中广泛应用。

**优势**:
- 适应不同参数的学习速度
- 处理非平稳目标
- 快速收敛

**配置**:
- 学习率: $3 \times 10^{-4}$
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- 通常不需要权重衰减

**应用场景**:
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Advantage Actor-Critic)
- SAC (Soft Actor-Critic)

---

### 5. 推荐系统

**矩阵分解**:

Adam用于优化用户-物品矩阵分解。

**问题**:
$$
\min_{U,V} \sum_{(i,j) \in \Omega} (R_{ij} - U_i V_j^T)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)
$$

**Adam优势**:
- 处理稀疏梯度（只有观测到的$(i,j)$有梯度）
- 自适应学习率适应不同用户/物品的更新频率
- 快速收敛

**实践示例**:

```python
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_emb(user_ids)
        item_vec = self.item_emb(item_ids)
        return (user_vec * item_vec).sum(dim=1)

model = MatrixFactorization(n_users, n_items)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for user_id, item_id, rating in train_data:
    optimizer.zero_grad()
    pred = model(user_id, item_id)
    loss = F.mse_loss(pred, rating)
    loss.backward()
    optimizer.step()
```

---

### 6. 超参数优化

**Adam作为元优化器**:

使用Adam优化超参数（如学习率、正则化系数）。

**双层优化**:
$$
\min_\lambda \mathcal{L}_{\text{val}}(\theta^*(\lambda)) \quad \text{s.t.} \quad \theta^*(\lambda) = \arg\min_\theta \mathcal{L}_{\text{train}}(\theta, \lambda)
$$

**使用Adam优化$\lambda$**:
- 计算超参数梯度
- 使用Adam更新超参数
- 比网格搜索更高效

---

### 7. 迁移学习

**Fine-tuning预训练模型**:

Adam在迁移学习中广泛使用，特别是fine-tuning大型预训练模型。

**策略**:
- **全模型微调**: 所有层使用Adam，学习率 $10^{-5}$ 到 $10^{-3}$
- **部分微调**: 只训练顶层，学习率 $10^{-3}$ 到 $10^{-2}$
- **LoRA微调**: 低秩适应，Adam优化低秩矩阵

**实践建议**:
- 使用较小的学习率（预训练模型的1/10）
- 使用AdamW避免权重衰减问题
- 使用学习率调度（Cosine Annealing）

---

### 8. 对比学习

**自监督学习**:

Adam在对比学习（SimCLR、MoCo等）中表现优异。

**特点**:
- 大批量训练（4096+）
- 需要稳定的优化器
- Adam的自适应性有助于处理不同样本的梯度

**配置**:
- 学习率: $0.0003 \times \text{batch_size} / 256$ (线性缩放)
- Warmup: 10个epoch
- Cosine Annealing

---

## 📚 核心要点

| 概念 | 说明 |
|------|------|
| **一阶矩** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ |
| **二阶矩** | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ |
| **偏差修正** | $\hat{m}_t = m_t / (1 - \beta_1^t)$ |
| **自适应学习率** | $\alpha / \sqrt{\hat{v}_t}$ |

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS231n Deep Learning |
| **MIT** | 6.036 Introduction to ML |
| **CMU** | 10-725 Convex Optimization |

---

## 📖 参考文献

1. **Kingma & Ba (2015)**. "Adam: A Method for Stochastic Optimization". *ICLR*.

2. **Loshchilov & Hutter (2019)**. "Decoupled Weight Decay Regularization". *ICLR*.

3. **Reddi et al. (2018)**. "On the Convergence of Adam and Beyond". *ICLR*.

---

*最后更新：2025年12月20日*-
