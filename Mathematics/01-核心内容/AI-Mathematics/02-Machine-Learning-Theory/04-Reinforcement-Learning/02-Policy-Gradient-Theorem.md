# 策略梯度定理

> **Policy Gradient Theorem**
>
> 直接优化策略的强化学习方法

---

## 目录

- [策略梯度定理](#策略梯度定理)
  - [目录](#目录)
  - [📋 核心思想](#-核心思想)
  - [🎯 策略梯度定理](#-策略梯度定理)
    - [1. 问题设定](#1-问题设定)
    - [2. 定理陈述](#2-定理陈述)
    - [3. 证明思路](#3-证明思路)
  - [📊 经典算法](#-经典算法)
    - [1. REINFORCE](#1-reinforce)
    - [2. Actor-Critic](#2-actor-critic)
    - [3. 优势函数](#3-优势函数)
  - [🔧 现代变体](#-现代变体)
    - [1. PPO (Proximal Policy Optimization)](#1-ppo-proximal-policy-optimization)
    - [2. TRPO (Trust Region Policy Optimization)](#2-trpo-trust-region-policy-optimization)
  - [💻 Python实现](#-python实现)
  - [📚 核心定理总结](#-核心定理总结)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心思想

**策略梯度方法**直接优化参数化策略 $\pi_\theta(a|s)$。

**核心优势**：

- 可处理连续动作空间
- 可学习随机策略
- 更好的收敛性质

**挑战**：

- 高方差
- 样本效率低

---

## 🎯 策略梯度定理

### 1. 问题设定

**目标**：最大化期望累积奖励

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

其中 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ 是轨迹。

---

### 2. 定理陈述

**定理 2.1 (策略梯度定理)**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时刻 $t$ 开始的累积奖励。

**等价形式**（使用Q函数）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s, a)\right]
$$

其中 $d^\pi(s)$ 是在策略 $\pi$ 下的状态分布。

---

### 3. 证明思路

**关键步骤**：

1. **轨迹概率**：
   $$
   P(\tau|\theta) = \rho_0(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
   $$

2. **对数技巧**：
   $$
   \nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)
   $$

3. **状态转移独立于 $\theta$**：
   $$
   \nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)
   $$

4. **期望梯度**：
   $$
   \nabla_\theta J(\theta) = \mathbb{E}_\tau\left[R(\tau) \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]
   $$

---

## 📊 经典算法

### 1. REINFORCE

**算法 1.1 (REINFORCE / Monte Carlo Policy Gradient)**:

```text
对于每个episode:
  1. 采样轨迹 τ ~ π_θ
  2. 计算累积奖励 G_t
  3. 更新: θ ← θ + α ∇_θ log π_θ(a_t|s_t) G_t
```

**特点**：

- 无偏估计
- 高方差

**基线技巧**：减去基线 $b(s_t)$ 降低方差：

$$
\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t))
$$

常用基线：$b(s_t) = V(s_t)$

---

### 2. Actor-Critic

**核心思想**：结合策略梯度和价值函数。

- **Actor**：策略 $\pi_\theta(a|s)$
- **Critic**：价值函数 $V_w(s)$ 或 $Q_w(s, a)$

**更新规则**：

```text
Critic: w ← w + α_w δ_t ∇_w V_w(s_t)
Actor:  θ ← θ + α_θ δ_t ∇_θ log π_θ(a_t|s_t)
```

其中 TD误差：$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$

---

### 3. 优势函数

**定义 3.1 (优势函数)**:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

**意义**：动作 $a$ 相对于平均的优势。

**优势Actor-Critic (A2C)**：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s, a)\right]
$$

**广义优势估计 (GAE)**：

$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

---

## 🔧 现代变体

### 1. PPO (Proximal Policy Optimization)

**核心思想**：限制策略更新幅度。

**目标函数**：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

其中：

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比率
- $\epsilon$ 是裁剪参数（通常0.2）

**优势**：

- 简单实现
- 稳定训练
- 样本效率高

---

### 2. TRPO (Trust Region Policy Optimization)

**核心思想**：约束KL散度。

**优化问题**：

$$
\max_\theta \mathbb{E}\left[r_t(\theta) \hat{A}_t\right] \quad \text{s.t.} \quad \mathbb{E}[D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta)] \leq \delta
$$

**实现**：使用共轭梯度法求解。

**优势**：

- 理论保证
- 单调改进

---

## 💻 Python实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)


class REINFORCE:
    """REINFORCE算法"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[action]
    
    def update(self, states, actions, rewards):
        """更新策略"""
        # 计算累积奖励
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        
        # 计算损失
        loss = 0
        for state, action, G in zip(states, actions, returns):
            state = torch.FloatTensor(state)
            probs = self.policy(state)
            log_prob = torch.log(probs[action])
            loss -= log_prob * G
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class ActorCritic:
    """Actor-Critic算法"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def update(self, state, action, reward, next_state, done):
        """更新Actor和Critic"""
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        
        # Critic更新
        value = self.critic(state)
        next_value = self.critic(next_state)
        
        td_target = reward + self.gamma * next_value * (1 - done)
        td_error = td_target - value
        
        critic_loss = td_error.pow(2)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor更新
        probs = self.actor(state)
        log_prob = torch.log(probs[action])
        actor_loss = -log_prob * td_error.detach()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


# 示例：CartPole环境
import gym

env = gym.make('CartPole-v1')
agent = REINFORCE(state_dim=4, action_dim=2)

for episode in range(1000):
    state = env.reset()
    states, actions, rewards = [], [], []
    
    for t in range(500):
        action, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
        
        if done:
            break
    
    loss = agent.update(states, actions, rewards)
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}, Loss: {loss:.4f}")
```

---

## 📚 核心定理总结

| 定理/算法 | 核心思想 | 优缺点 |
|-----------|----------|--------|
| **策略梯度定理** | $\nabla J = \mathbb{E}[\nabla \log \pi \cdot Q]$ | 理论基础 |
| **REINFORCE** | Monte Carlo采样 | 无偏但高方差 |
| **Actor-Critic** | 结合策略和价值 | 降低方差 |
| **PPO** | 裁剪重要性采样 | 简单稳定 |
| **TRPO** | KL散度约束 | 理论保证 |

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **UC Berkeley** | CS285 Deep Reinforcement Learning |
| **Stanford** | CS234 Reinforcement Learning |
| **DeepMind** | UCL Course on RL |

---

## 📖 参考文献

1. **Sutton et al. (2000)**. "Policy Gradient Methods for Reinforcement Learning with Function Approximation". *NeurIPS*.

2. **Schulman et al. (2017)**. "Proximal Policy Optimization Algorithms". *arXiv*.

3. **Schulman et al. (2015)**. "Trust Region Policy Optimization". *ICML*.

---

*最后更新：2025年10月*-
