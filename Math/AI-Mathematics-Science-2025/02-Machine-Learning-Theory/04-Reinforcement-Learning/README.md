# 强化学习数学基础 (Reinforcement Learning Mathematics)

> **From MDP to Deep RL: The Mathematics of Decision Making**
>
> 从MDP到深度强化学习：决策的数学

---

## 目录

- [强化学习数学基础 (Reinforcement Learning Mathematics)](#强化学习数学基础-reinforcement-learning-mathematics)
  - [目录](#目录)
  - [📋 模块概览](#-模块概览)
  - [📚 核心内容](#-核心内容)
    - [1. [MDP与Bellman方程](./01-MDP-Bellman-Equations.md) ✅](#1-mdp与bellman方程01-mdp-bellman-equationsmd-)
    - [2. [策略梯度定理](./02-Policy-Gradient-Theorem.md) ✅](#2-策略梯度定理02-policy-gradient-theoremmd-)
    - [3. [Q-Learning与值函数方法](./03-Q-Learning-Value-Functions.md) ✅](#3-q-learning与值函数方法03-q-learning-value-functionsmd-)
  - [🔗 与其他模块的联系](#-与其他模块的联系)
  - [📖 学习路径](#-学习路径)
  - [🎓 对标课程](#-对标课程)
  - [📊 模块完成度](#-模块完成度)

---

## 📋 模块概览

**强化学习**研究智能体在环境中通过试错学习最优策略的数学理论。

**核心问题**:

- 如何形式化决策问题？
- 如何计算最优策略？
- 如何从经验中学习？

---

## 📚 核心内容

### 1. [MDP与Bellman方程](./01-MDP-Bellman-Equations.md) ✅

**核心主题**:

- 马尔可夫决策过程 (MDP)
- 状态值函数与动作值函数
- Bellman方程
- 最优性原理

**关键公式**:

$$
V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]
$$

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]
$$

**AI应用**:

- 所有强化学习算法的基础
- 游戏AI
- 机器人控制

---

### 2. [策略梯度定理](./02-Policy-Gradient-Theorem.md) ✅

**核心主题**:

- 策略梯度定理
- REINFORCE算法
- Actor-Critic方法
- PPO与TRPO

**关键定理**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]
$$

**AI应用**:

- 连续动作空间
- 策略优化
- 深度强化学习

---

### 3. [Q-Learning与值函数方法](./03-Q-Learning-Value-Functions.md) ✅

**核心主题**:

- Q-Learning算法
- 值迭代与策略迭代
- 函数逼近
- Deep Q-Networks (DQN)
- DQN变体（Double DQN, Dueling DQN, Prioritized Replay）

**关键算法**:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]
$$

**AI应用**:

- 离散动作空间
- 游戏AI（Atari, Go）
- 推荐系统

---

## 🔗 与其他模块的联系

### 优化理论

```text
策略梯度 → 梯度上升
值函数优化 → 动态规划
```

### 统计学习

```text
函数逼近 → 监督学习
经验回放 → 样本效率
```

### 概率论

```text
MDP → 马尔可夫过程
策略 → 概率分布
```

---

## 📖 学习路径

### 阶段1: 理论基础 (2-3周)

1. **MDP与Bellman方程**
   - 理解决策问题形式化
   - 掌握值函数计算

2. **最优性原理**
   - 理解最优策略
   - 掌握动态规划

### 阶段2: 算法学习 (3-4周)

1. **值函数方法**
   - Q-Learning
   - DQN及其变体

2. **策略梯度方法**
   - REINFORCE
   - Actor-Critic
   - PPO

### 阶段3: 高级主题 (2-3周)

1. **函数逼近**
   - 线性函数逼近
   - 神经网络逼近

2. **探索与利用**
   - ε-贪婪策略
   - UCB
   - Thompson采样

---

## 🎓 对标课程

| 大学 | 课程代码 | 课程名称 | 对应内容 |
|------|----------|----------|----------|
| **UC Berkeley** | CS285 | Deep Reinforcement Learning | MDP、策略梯度、DQN |
| **Stanford** | CS234 | Reinforcement Learning | MDP、值函数、策略梯度 |
| **CMU** | 10-703 | Deep Reinforcement Learning | 深度强化学习 |
| **MIT** | 6.819 | Advanced Topics in ML | 强化学习理论 |

---

## 📊 模块完成度

**当前完成度**: 约50% (从40%提升)

**已完成文档**:

- ✅ MDP与Bellman方程 (约80%完成)
- ✅ 策略梯度定理 (约75%完成)
- ✅ Q-Learning与值函数方法 (约80%完成)

**待完善内容**:

- [ ] 补充Actor-Critic详细内容
- [ ] 补充PPO与TRPO详细内容
- [ ] 补充更多应用实例
- [ ] 补充形式化证明

---

**最后更新**: 2025-12-20
**维护**: Mathematics项目团队
