# 马尔可夫决策过程与Bellman方程

> **Markov Decision Processes and Bellman Equations**
>
> 强化学习的数学基础

---

## 目录

- [马尔可夫决策过程与Bellman方程](#马尔可夫决策过程与bellman方程)
  - [目录](#目录)
  - [📋 核心概念](#-核心概念)
  - [🎯 马尔可夫决策过程 (MDP)](#-马尔可夫决策过程-mdp)
    - [1. 形式化定义](#1-形式化定义)
    - [2. 策略](#2-策略)
    - [3. 价值函数](#3-价值函数)
  - [📊 Bellman方程](#-bellman方程)
    - [1. Bellman期望方程](#1-bellman期望方程)
    - [2. Bellman最优方程](#2-bellman最优方程)
    - [3. 最优策略](#3-最优策略)
  - [🔧 求解方法](#-求解方法)
    - [1. 动态规划](#1-动态规划)
    - [2. 蒙特卡洛方法](#2-蒙特卡洛方法)
    - [3. 时序差分学习](#3-时序差分学习)
  - [🤖 深度强化学习](#-深度强化学习)
  - [💻 Python实现](#-python实现)
  - [📚 核心定理总结](#-核心定理总结)
  - [🎓 相关课程](#-相关课程)
  - [📖 参考文献](#-参考文献)

---

## 📋 核心概念

**强化学习**研究智能体如何通过与环境交互学习最优行为策略。

**核心要素**：
- **状态 (State)**：环境的描述
- **动作 (Action)**：智能体的选择
- **奖励 (Reward)**：即时反馈
- **策略 (Policy)**：状态到动作的映射
- **价值 (Value)**：长期累积奖励

---

## 🎯 马尔可夫决策过程 (MDP)

### 1. 形式化定义

**定义 1.1 (MDP)**:

MDP是一个五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$：

- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间
- $P(s'|s, a)$：状态转移概率
- $R(s, a, s')$：奖励函数
- $\gamma \in [0, 1)$：折扣因子

**马尔可夫性质**：

$$
\mathbb{P}(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = \mathbb{P}(s_{t+1} | s_t, a_t)
$$

---

### 2. 策略

**定义 2.1 (策略)**:

策略 $\pi$ 是从状态到动作的映射：

- **确定性策略**：$\pi : \mathcal{S} \to \mathcal{A}$
- **随机策略**：$\pi(a|s) = \mathbb{P}(a_t = a | s_t = s)$

---

### 3. 价值函数

**定义 3.1 (状态价值函数)**:

$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s\right]
$$

**定义 3.2 (动作价值函数)**:

$$
Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s, A_0 = a\right]
$$

**关系**：

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^\pi(s, a)
$$

---

## 📊 Bellman方程

### 1. Bellman期望方程

**定理 1.1 (Bellman期望方程)**:

对于任意策略 $\pi$：

$$
V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]
$$

$$
Q^\pi(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]
$$

**矩阵形式**：

$$
V^\pi = R^\pi + \gamma P^\pi V^\pi
$$

**解析解**：

$$
V^\pi = (I - \gamma P^\pi)^{-1} R^\pi
$$

---

### 2. Bellman最优方程

**定义 2.1 (最优价值函数)**:

$$
V^*(s) = \max_\pi V^\pi(s)
$$

$$
Q^*(s, a) = \max_\pi Q^\pi(s, a)
$$

**定理 2.2 (Bellman最优方程)**:

$$
V^*(s) = \max_{a \in \mathcal{A}} \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^*(s')]
$$

$$
Q^*(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \max_{a'} Q^*(s', a')]
$$

---

### 3. 最优策略

**定理 3.1 (最优策略存在性)**:

对于任意有限MDP，存在确定性最优策略 $\pi^*$ 使得：

$$
\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s, a)
$$

**唯一性**：最优价值函数唯一，但最优策略可能不唯一。

---

## 🔧 求解方法

### 1. 动态规划

**值迭代 (Value Iteration)**:

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_k(s')]
$$

**收敛性**：$V_k \to V^*$ 以几何速率收敛。

**策略迭代 (Policy Iteration)**:

1. **策略评估**：解 $V^\pi = R^\pi + \gamma P^\pi V^\pi$
2. **策略改进**：$\pi'(s) = \arg\max_a Q^\pi(s, a)$

---

### 2. 蒙特卡洛方法

**思想**：通过采样轨迹估计价值函数。

$$
V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i(s)
$$

其中 $G_i(s)$ 是第 $i$ 条轨迹中从状态 $s$ 开始的累积奖励。

---

### 3. 时序差分学习

**TD(0) 更新**:

$$
V(s_t) \leftarrow V(s_t) + \alpha [R_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
$$

**Q-Learning**:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
$$

**SARSA**:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
$$

---

## 🤖 深度强化学习

**DQN (Deep Q-Network)**:

用神经网络逼近 $Q^*(s, a)$：

$$
L(\theta) = \mathbb{E}[(R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]
$$

**关键技术**：
- Experience Replay
- Target Network
- Double DQN

---

## 💻 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 简单Grid World MDP
class GridWorldMDP:
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上下左右
        self.gamma = 0.9
        
        # 定义奖励
        self.goal_state = (size-1, size-1)
        self.trap_state = (2, 2)
    
    def state_to_coord(self, state):
        return (state // self.size, state % self.size)
    
    def coord_to_state(self, coord):
        return coord[0] * self.size + coord[1]
    
    def get_next_state(self, state, action):
        """状态转移"""
        x, y = self.state_to_coord(state)
        
        # 动作: 0=上, 1=下, 2=左, 3=右
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.size - 1, y + 1)
        
        return self.coord_to_state((x, y))
    
    def get_reward(self, state, action, next_state):
        """奖励函数"""
        coord = self.state_to_coord(next_state)
        
        if coord == self.goal_state:
            return 10.0
        elif coord == self.trap_state:
            return -10.0
        else:
            return -0.1  # 每步小惩罚


# 2. 值迭代
def value_iteration(mdp, max_iter=1000, tol=1e-6):
    """值迭代算法"""
    V = np.zeros(mdp.n_states)
    
    for iteration in range(max_iter):
        V_new = np.zeros(mdp.n_states)
        
        for s in range(mdp.n_states):
            # Bellman最优方程
            q_values = []
            for a in range(mdp.n_actions):
                s_next = mdp.get_next_state(s, a)
                r = mdp.get_reward(s, a, s_next)
                q_values.append(r + mdp.gamma * V[s_next])
            
            V_new[s] = max(q_values)
        
        # 检查收敛
        if np.max(np.abs(V_new - V)) < tol:
            print(f"Converged in {iteration} iterations")
            break
        
        V = V_new
    
    # 提取最优策略
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        q_values = []
        for a in range(mdp.n_actions):
            s_next = mdp.get_next_state(s, a)
            r = mdp.get_reward(s, a, s_next)
            q_values.append(r + mdp.gamma * V[s_next])
        policy[s] = np.argmax(q_values)
    
    return V, policy


# 3. Q-Learning
def q_learning(mdp, n_episodes=1000, alpha=0.1, epsilon=0.1):
    """Q-Learning算法"""
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    
    for episode in range(n_episodes):
        state = 0  # 起始状态
        
        for step in range(100):  # 最多100步
            # ε-贪心策略
            if np.random.rand() < epsilon:
                action = np.random.randint(mdp.n_actions)
            else:
                action = np.argmax(Q[state])
            
            # 执行动作
            next_state = mdp.get_next_state(state, action)
            reward = mdp.get_reward(state, action, next_state)
            
            # Q-Learning更新
            Q[state, action] += alpha * (
                reward + mdp.gamma * np.max(Q[next_state]) - Q[state, action]
            )
            
            # 终止条件
            coord = mdp.state_to_coord(next_state)
            if coord == mdp.goal_state or coord == mdp.trap_state:
                break
            
            state = next_state
    
    # 提取策略
    policy = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    
    return V, policy, Q


# 4. 可视化
def visualize_policy(mdp, V, policy):
    """可视化价值函数和策略"""
    V_grid = V.reshape(mdp.size, mdp.size)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 价值函数
    im = axes[0].imshow(V_grid, cmap='viridis')
    axes[0].set_title('Value Function', fontsize=14)
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im, ax=axes[0])
    
    # 策略
    policy_grid = policy.reshape(mdp.size, mdp.size)
    arrows = ['↑', '↓', '←', '→']
    
    axes[1].imshow(V_grid, cmap='gray', alpha=0.3)
    for i in range(mdp.size):
        for j in range(mdp.size):
            state = mdp.coord_to_state((i, j))
            axes[1].text(j, i, arrows[policy[state]], 
                        ha='center', va='center', fontsize=20)
    
    axes[1].set_title('Optimal Policy', fontsize=14)
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    
    plt.tight_layout()
    plt.savefig('mdp_solution.png', dpi=150)
    plt.show()


# 运行示例
if __name__ == "__main__":
    mdp = GridWorldMDP(size=5)
    
    print("Running Value Iteration...")
    V_vi, policy_vi = value_iteration(mdp)
    
    print("\nRunning Q-Learning...")
    V_ql, policy_ql, Q = q_learning(mdp, n_episodes=5000)
    
    visualize_policy(mdp, V_vi, policy_vi)
```

---

## 📚 核心定理总结

| 定理 | 陈述 | 意义 |
|------|------|------|
| **Bellman期望方程** | $V^\pi = R^\pi + \gamma P^\pi V^\pi$ | 策略评估 |
| **Bellman最优方程** | $V^* = \max_a [R^a + \gamma P^a V^*]$ | 最优性条件 |
| **最优策略存在** | $\exists \pi^*: V^{\pi^*} = V^*$ | 可求解性 |
| **值迭代收敛** | $V_k \to V^*$ 指数收敛 | 算法保证 |

---

## 🎓 相关课程

| 大学 | 课程 |
|------|------|
| **Stanford** | CS234 Reinforcement Learning |
| **UC Berkeley** | CS285 Deep Reinforcement Learning |
| **MIT** | 6.246 Reinforcement Learning |
| **DeepMind** | UCL Course on RL |

---

## 📖 参考文献

1. **Sutton & Barto (2018)**. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

2. **Puterman (2014)**. *Markov Decision Processes*. Wiley.

3. **Bertsekas (2019)**. *Reinforcement Learning and Optimal Control*. Athena Scientific.

---

*最后更新：2025年10月*
