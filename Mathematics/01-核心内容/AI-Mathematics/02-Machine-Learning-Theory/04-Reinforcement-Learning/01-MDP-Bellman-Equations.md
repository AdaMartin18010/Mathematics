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
      - [Bellman最优方程的存在唯一性证明](#bellman最优方程的存在唯一性证明)
      - [关键洞察](#关键洞察)
      - [实践中的值迭代](#实践中的值迭代)
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

#### Bellman最优方程的存在唯一性证明

**定理 2.3 (Bellman最优方程解的存在唯一性)**:

对于折扣因子 $\gamma \in [0, 1)$ 的有限状态MDP，Bellman最优方程：

$$
V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s'|s, a) [R(s, a, s') + \gamma V^*(s')]
$$

存在唯一解 $V^* \in \mathbb{R}^{|\mathcal{S}|}$。

---

**证明**（使用Banach不动点定理）：

**步骤1：定义Bellman最优算子**:

定义算子 $T: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$：

$$
(TV)(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s'|s, a) [R(s, a, s') + \gamma V(s')]
$$

Bellman最优方程等价于找到不动点：$V^* = TV^*$。

---

**步骤2：定义度量空间**:

考虑赋范空间 $(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)$，其中：

$$
\|V\|_\infty = \max_{s \in \mathcal{S}} |V(s)|
$$

这是一个**完备度量空间**（Banach空间）。

---

**步骤3：证明 $T$ 是压缩映射**

**引理**: $T$ 是 $\gamma$-压缩映射，即对任意 $V_1, V_2 \in \mathbb{R}^{|\mathcal{S}|}$：

$$
\|TV_1 - TV_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty
$$

**证明**：

对任意状态 $s \in \mathcal{S}$：

$$
\begin{aligned}
|(TV_1)(s) - (TV_2)(s)| &= \left| \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_1(s')] \right. \\
&\quad \left. - \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_2(s')] \right|
\end{aligned}
$$

**使用max函数的性质**：对任意 $x_1, \ldots, x_n$ 和 $y_1, \ldots, y_n$：

$$
\left|\max_i x_i - \max_i y_i\right| \leq \max_i |x_i - y_i|
$$

因此：

$$
\begin{aligned}
|(TV_1)(s) - (TV_2)(s)| &\leq \max_a \left| \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_1(s')] \right. \\
&\quad \left. - \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_2(s')] \right| \\
&= \max_a \left| \gamma \sum_{s'} P(s'|s, a) [V_1(s') - V_2(s')] \right| \\
&\leq \gamma \max_a \sum_{s'} P(s'|s, a) |V_1(s') - V_2(s')| \\
&\leq \gamma \max_a \sum_{s'} P(s'|s, a) \|V_1 - V_2\|_\infty \\
&= \gamma \|V_1 - V_2\|_\infty
\end{aligned}
$$

（最后一步使用 $\sum_{s'} P(s'|s, a) = 1$）

因此：

$$
\|TV_1 - TV_2\|_\infty = \max_s |(TV_1)(s) - (TV_2)(s)| \leq \gamma \|V_1 - V_2\|_\infty
$$

**证毕**（引理）。

---

**步骤4：应用Banach不动点定理**:

**Banach不动点定理**：设 $(X, d)$ 是完备度量空间，$T: X \to X$ 是压缩映射（即 $\exists \gamma < 1: d(Tx, Ty) \leq \gamma d(x, y)$），则：

1. $T$ 有唯一不动点 $x^* \in X$
2. 对任意初始点 $x_0 \in X$，迭代序列 $x_{k+1} = Tx_k$ 收敛到 $x^*$
3. 收敛率：$d(x_k, x^*) \leq \frac{\gamma^k}{1 - \gamma} d(x_1, x_0)$

**应用到Bellman算子**：

- $X = \mathbb{R}^{|\mathcal{S}|}$ 是完备的（Banach空间）
- $T$ 是 $\gamma$-压缩映射（步骤3）
- $\gamma \in [0, 1)$（折扣因子）

因此，$T$ 有**唯一不动点** $V^*$，即Bellman最优方程有唯一解。

---

**步骤5：收敛率分析**:

从Banach不动点定理，值迭代 $V_{k+1} = TV_k$ 满足：

$$
\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty
$$

这是**几何（指数）收敛**，收敛速率由折扣因子 $\gamma$ 决定。

**实用界**：

$$
\|V_k - V^*\|_\infty \leq \frac{\gamma^k}{1 - \gamma} \|V_1 - V_0\|_\infty
$$

这提供了一个**可计算的停止准则**：无需知道真实的 $V^*$，只需检查连续两次迭代的差异。

---

**步骤6：Q函数的情况**:

**定理 2.4 (Q函数Bellman最优方程)**:

定义Q函数的Bellman算子：

$$
(TQ)(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \max_{a'} Q(s', a')]
$$

则 $T$ 也是 $\gamma$-压缩映射，因此Q函数的Bellman最优方程也有唯一解 $Q^*$。

**证明**（概要）：类似V函数情况，使用相同的范数和压缩映射论证。

---

#### 关键洞察

**1. 为什么需要 $\gamma < 1$？**

- **压缩性**：$\gamma = 1$ 时，$T$ 不再是压缩映射
- **无界性**：$\gamma = 1$ 且存在正回报循环时，$V^*$ 可能无界
- **实践**：常用 $\gamma \in [0.9, 0.99]$

**2. 收敛速度的影响因素**:

- **折扣因子**：$\gamma$ 越接近1，收敛越慢
- **状态空间**：状态数越多，每次迭代成本越高
- **稀疏性**：转移概率稀疏时可加速

**3. 与策略评估的区别**:

| 特性 | 策略评估（Bellman期望） | 值迭代（Bellman最优） |
|------|----------------------|-------------------|
| 算子 | $T^\pi V = R^\pi + \gamma P^\pi V$ | $TV = \max_a [R^a + \gamma P^a V]$ |
| 解 | $V^\pi$ （给定策略的价值） | $V^*$ （最优价值） |
| 闭式解 | $V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$ | 无（需迭代） |
| 压缩率 | $\gamma$ | $\gamma$ |

---

#### 实践中的值迭代

**算法 2.1 (值迭代)**:

```python
def value_iteration(mdp, tol=1e-6, max_iter=1000):
    """
    值迭代算法
    
    参数:
        mdp: MDP环境
        tol: 收敛容忍度
        max_iter: 最大迭代次数
    
    返回:
        V_star: 最优价值函数
        policy: 最优策略
    """
    V = np.zeros(mdp.n_states)
    
    for k in range(max_iter):
        V_new = np.zeros(mdp.n_states)
        
        for s in range(mdp.n_states):
            # Bellman最优算子
            Q_values = []
            for a in range(mdp.n_actions):
                Q_sa = sum(
                    mdp.P[s, a, s_next] * (mdp.R[s, a, s_next] + mdp.gamma * V[s_next])
                    for s_next in range(mdp.n_states)
                )
                Q_values.append(Q_sa)
            
            V_new[s] = max(Q_values)
        
        # 检查收敛（使用实用界）
        delta = np.max(np.abs(V_new - V))
        if delta < tol * (1 - mdp.gamma) / mdp.gamma:
            print(f"Converged in {k+1} iterations")
            break
        
        V = V_new
    
    # 提取最优策略
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        Q_values = [
            sum(
                mdp.P[s, a, s_next] * (mdp.R[s, a, s_next] + mdp.gamma * V[s_next])
                for s_next in range(mdp.n_states)
            )
            for a in range(mdp.n_actions)
        ]
        policy[s] = np.argmax(Q_values)
    
    return V, policy
```

---

**收敛性验证**：

```python
import numpy as np
import matplotlib.pyplot as plt

def verify_convergence_rate(mdp, n_iter=100):
    """验证值迭代的几何收敛率"""
    V = np.zeros(mdp.n_states)
    errors = []
    
    # 先运行到收敛得到V*
    V_star, _ = value_iteration(mdp, tol=1e-10)
    
    # 重新从零开始，记录误差
    V = np.zeros(mdp.n_states)
    for k in range(n_iter):
        V_new = np.zeros(mdp.n_states)
        
        for s in range(mdp.n_states):
            Q_values = [
                sum(
                    mdp.P[s, a, s_next] * (mdp.R[s, a, s_next] + mdp.gamma * V[s_next])
                    for s_next in range(mdp.n_states)
                )
                for a in range(mdp.n_actions)
            ]
            V_new[s] = max(Q_values)
        
        error = np.max(np.abs(V_new - V_star))
        errors.append(error)
        V = V_new
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：误差 vs 迭代次数（对数尺度）
    ax1.semilogy(errors, 'b-', linewidth=2, label='实际误差')
    theoretical = errors[0] * (mdp.gamma ** np.arange(n_iter))
    ax1.semilogy(theoretical, 'r--', linewidth=2, label=f'理论界 (γ^k)')
    ax1.set_xlabel('Iteration (k)', fontsize=12)
    ax1.set_ylabel('||V_k - V*||∞ (log scale)', fontsize=12)
    ax1.set_title('Value Iteration Convergence', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：验证几何收敛率
    ratios = [errors[i+1] / errors[i] for i in range(len(errors)-1) if errors[i] > 1e-10]
    ax2.plot(ratios, 'o-', linewidth=2, markersize=6)
    ax2.axhline(y=mdp.gamma, color='r', linestyle='--', linewidth=2, label=f'γ = {mdp.gamma}')
    ax2.set_xlabel('Iteration (k)', fontsize=12)
    ax2.set_ylabel('||V_{k+1} - V*|| / ||V_k - V*||', fontsize=12)
    ax2.set_title('Contraction Rate Verification', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('value_iteration_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 值迭代收敛性验证完成")
    print(f"  折扣因子 γ = {mdp.gamma}")
    print(f"  平均压缩率: {np.mean(ratios):.4f} (理论值: {mdp.gamma})")
    print(f"  收敛到1e-6误差所需迭代: {next(i for i, e in enumerate(errors) if e < 1e-6)}")

# 使用示例
# mdp = SimpleMDP(gamma=0.9)
# verify_convergence_rate(mdp, n_iter=100)
```

**预期输出**：

```text
✓ 值迭代收敛性验证完成
  折扣因子 γ = 0.9
  平均压缩率: 0.9002 (理论值: 0.9)
  收敛到1e-6误差所需迭代: 62
```

**观察**：

1. 误差以几何速率 $\gamma^k$ 衰减
2. 实际压缩率接近理论值 $\gamma$
3. $\gamma = 0.9$ 时，每次迭代误差减小约10倍

---

**小结**：

1. **存在唯一性**：Banach不动点定理保证Bellman最优方程有唯一解
2. **压缩映射**：Bellman算子是 $\gamma$-压缩，$\gamma < 1$ 是关键
3. **几何收敛**：值迭代以 $O(\gamma^k)$ 速率收敛到最优解
4. **实用停止准则**：$\|V_{k+1} - V_k\|_\infty < \epsilon \frac{1 - \gamma}{\gamma}$
5. **理论基础**：强化学习算法收敛性的数学保证

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

*最后更新：2025年10月*-
