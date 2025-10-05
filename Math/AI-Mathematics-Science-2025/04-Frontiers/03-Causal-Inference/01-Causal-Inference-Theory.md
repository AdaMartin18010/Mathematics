# 因果推断理论 (Causal Inference Theory)

> **From Correlation to Causation: The Mathematics of Causal Reasoning**
>
> 从相关到因果：因果推理的数学基础

---

## 目录

- [因果推断理论 (Causal Inference Theory)](#因果推断理论-causal-inference-theory)
  - [目录](#目录)
  - [📋 概述](#-概述)
  - [🎯 核心概念](#-核心概念)
    - [1. 因果关系 vs 相关关系](#1-因果关系-vs-相关关系)
    - [2. 反事实推理](#2-反事实推理)
    - [3. 因果效应](#3-因果效应)
  - [📐 因果图模型](#-因果图模型)
    - [1. 结构因果模型 (SCM)](#1-结构因果模型-scm)
    - [2. 有向无环图 (DAG)](#2-有向无环图-dag)
    - [3. d-分离](#3-d-分离)
  - [🔬 因果推断框架](#-因果推断框架)
    - [1. Rubin因果模型 (潜在结果框架)](#1-rubin因果模型-潜在结果框架)
    - [2. Pearl因果模型 (结构方程框架)](#2-pearl因果模型-结构方程框架)
    - [3. 两种框架的统一](#3-两种框架的统一)
  - [📊 因果识别](#-因果识别)
    - [1. 后门准则 (Backdoor Criterion)](#1-后门准则-backdoor-criterion)
    - [2. 前门准则 (Frontdoor Criterion)](#2-前门准则-frontdoor-criterion)
    - [3. do-演算 (do-Calculus)](#3-do-演算-do-calculus)
  - [💡 因果效应估计](#-因果效应估计)
    - [1. 随机对照试验 (RCT)](#1-随机对照试验-rct)
    - [2. 倾向得分匹配 (PSM)](#2-倾向得分匹配-psm)
    - [3. 工具变量 (IV)](#3-工具变量-iv)
    - [4. 双重差分 (DID)](#4-双重差分-did)
    - [5. 回归不连续 (RD)](#5-回归不连续-rd)
  - [🧠 机器学习中的因果推断](#-机器学习中的因果推断)
    - [1. 因果表示学习](#1-因果表示学习)
    - [2. 反事实推理与解释性](#2-反事实推理与解释性)
    - [3. 因果强化学习](#3-因果强化学习)
    - [4. 迁移学习与域适应](#4-迁移学习与域适应)
  - [💻 Python实现](#-python实现)
    - [示例1: 因果图与d-分离](#示例1-因果图与d-分离)
    - [示例2: 倾向得分匹配](#示例2-倾向得分匹配)
    - [示例3: 工具变量估计](#示例3-工具变量估计)
    - [示例4: 因果发现](#示例4-因果发现)
  - [🎓 对标课程](#-对标课程)
  - [📖 核心教材与论文](#-核心教材与论文)
    - [教材](#教材)
    - [经典论文](#经典论文)
    - [最新进展 (2024-2025)](#最新进展-2024-2025)
  - [🔗 相关主题](#-相关主题)
  - [📝 总结](#-总结)
    - [核心概念](#核心概念)
    - [理论框架](#理论框架)
    - [识别方法](#识别方法)
    - [估计方法](#估计方法)
    - [AI应用](#ai应用)

---

## 📋 概述

**因果推断**研究如何从观察数据中推断因果关系，是统计学、机器学习、经济学、流行病学等领域的核心问题。

**核心问题**:

1. **因果识别**: 给定观察数据，能否识别因果效应？
2. **因果估计**: 如何从数据中估计因果效应？
3. **因果发现**: 如何从数据中发现因果结构？

**为什么重要**:

- **预测 vs 干预**: 相关性预测未来，因果性指导干预
- **可解释性**: 理解"为什么"而非仅仅"是什么"
- **泛化能力**: 因果模型在分布变化下更鲁棒
- **决策支持**: 评估政策、治疗、干预的效果

---

## 🎯 核心概念

### 1. 因果关系 vs 相关关系

**相关关系** (Correlation):

$$
\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

**因果关系** (Causation):

$$
X \to Y: \text{改变 } X \text{ 会导致 } Y \text{ 的改变}
$$

**Simpson悖论**: 相关性可能误导因果推断

**示例**:

```text
总体: Corr(治疗, 康复) < 0  (负相关)
男性: Corr(治疗, 康复) > 0  (正相关)
女性: Corr(治疗, 康复) > 0  (正相关)
```

原因: 性别是混淆因子 (confounder)

### 2. 反事实推理

**反事实** (Counterfactual): "如果当时...会怎样?"

**定义**: 个体 $i$ 在接受治疗 $T=1$ 时的结果 $Y_i(1)$ 和未接受治疗 $T=0$ 时的结果 $Y_i(0)$

**根本问题**: 我们只能观察到 $Y_i(T_i)$，无法同时观察 $Y_i(1)$ 和 $Y_i(0)$

### 3. 因果效应

**平均因果效应** (ATE):

$$
\text{ATE} = \mathbb{E}[Y(1) - Y(0)]
$$

**条件平均因果效应** (CATE):

$$
\text{CATE}(x) = \mathbb{E}[Y(1) - Y(0) | X = x]
$$

**处理组平均因果效应** (ATT):

$$
\text{ATT} = \mathbb{E}[Y(1) - Y(0) | T = 1]
$$

---

## 📐 因果图模型

### 1. 结构因果模型 (SCM)

**定义**: SCM是一个三元组 $(U, V, F)$

- $U$: 外生变量 (unobserved)
- $V$: 内生变量 (observed)
- $F$: 结构方程 $V_i = f_i(\text{PA}_i, U_i)$

**示例**:

$$
\begin{align}
X &= U_X \\
Y &= \beta X + U_Y
\end{align}
$$

### 2. 有向无环图 (DAG)

**定义**: DAG $G = (V, E)$ 表示变量间的因果关系

- 节点 $V$: 变量
- 有向边 $E$: 因果关系 $X \to Y$

**示例**:

```text
    Z
   ↙ ↘
  X → Y
```

- $X$: 治疗
- $Y$: 结果
- $Z$: 混淆因子

### 3. d-分离

**定义**: 给定集合 $Z$，路径 $p$ 被 $Z$ 阻断 (blocked) 如果:

1. **链** (Chain): $X \to Z \to Y$，$Z \in \mathbf{Z}$
2. **叉** (Fork): $X \leftarrow Z \to Y$，$Z \in \mathbf{Z}$
3. **对撞** (Collider): $X \to Z \leftarrow Y$，$Z \notin \mathbf{Z}$ 且 $Z$ 的后代 $\notin \mathbf{Z}$

**定理** (d-分离准则):

$$
X \perp_G Y | Z \iff X \perp Y | Z \text{ (在 } G \text{ 对应的分布中)}
$$

---

## 🔬 因果推断框架

### 1. Rubin因果模型 (潜在结果框架)

**核心思想**: 每个个体都有潜在结果 $Y_i(0), Y_i(1)$

**观察数据**:

$$
Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)
$$

**识别假设**:

1. **SUTVA** (Stable Unit Treatment Value Assumption)
   - 无干扰: 个体 $i$ 的结果不受其他个体治疗的影响
   - 治疗一致性: 治疗只有一个版本

2. **可忽略性** (Ignorability):

    $$
    (Y(0), Y(1)) \perp T | X
    $$

    即给定协变量 $X$，治疗分配与潜在结果独立。

3. **正性** (Positivity):

    $$
    0 < P(T = 1 | X = x) < 1, \quad \forall x
    $$

### 2. Pearl因果模型 (结构方程框架)

**核心思想**: 因果关系由结构方程表示

**do-算子**: $P(Y | do(X = x))$ 表示干预 $X$ 为 $x$ 后 $Y$ 的分布

**do-演算规则**:

1. **插入/删除观察**:

    $$
    P(y | do(x), z, w) = P(y | do(x), w) \quad \text{if } (Y \perp Z | X, W)_{\overline{G_X}}
    $$

2. **动作/观察交换**:

    $$
    P(y | do(x), do(z), w) = P(y | do(x), z, w) \quad \text{if } (Y \perp Z | X, W)_{\overline{G_{XZ}}}
    $$

3. **插入/删除动作**:

    $$
    P(y | do(x), do(z), w) = P(y | do(x), w) \quad \text{if } (Y \perp Z | X, W)_{\overline{G_X}, \underline{G_{Z(W)}}}
    $$

### 3. 两种框架的统一

**定理**: 在可忽略性假设下，

$$
\mathbb{E}[Y(1) - Y(0)] = \mathbb{E}_X[\mathbb{E}[Y | T=1, X] - \mathbb{E}[Y | T=0, X]]
$$

这连接了Rubin的潜在结果和Pearl的条件期望。

---

## 📊 因果识别

### 1. 后门准则 (Backdoor Criterion)

**定义**: 集合 $Z$ 满足后门准则如果:

1. $Z$ 阻断所有从 $X$ 到 $Y$ 的后门路径 (包含指向 $X$ 的箭头的路径)
2. $Z$ 不包含 $X$ 的后代

**定理**: 如果 $Z$ 满足后门准则，则

$$
P(y | do(x)) = \sum_z P(y | x, z) P(z)
$$

**示例**:

```text
    Z
   ↙ ↘
  X → Y
```

$Z$ 满足后门准则，因此:

$$
P(y | do(x)) = \sum_z P(y | x, z) P(z)
$$

### 2. 前门准则 (Frontdoor Criterion)

**定义**: 集合 $M$ 满足前门准则如果:

1. $M$ 阻断所有从 $X$ 到 $Y$ 的有向路径
2. $X$ 阻断所有从 $M$ 到 $Y$ 的后门路径
3. 没有从 $X$ 到 $M$ 的后门路径

**定理**: 如果 $M$ 满足前门准则，则

$$
P(y | do(x)) = \sum_m P(m | x) \sum_{x'} P(y | m, x') P(x')
$$

**示例**:

```text
  U
 ↙ ↘
X → M → Y
```

$U$ 是未观察的混淆因子，但 $M$ 满足前门准则。

### 3. do-演算 (do-Calculus)

**目标**: 将 $P(y | do(x))$ 化简为观察分布的函数

**三条规则** (见上文)

**完备性定理** (Shpitser & Pearl, 2006):

如果 $P(y | do(x))$ 可识别，则可通过do-演算推导。

---

## 💡 因果效应估计

### 1. 随机对照试验 (RCT)

**黄金标准**: 随机分配治疗

$$
T \perp (Y(0), Y(1))
$$

**估计**:

$$
\widehat{\text{ATE}} = \frac{1}{n_1} \sum_{i: T_i=1} Y_i - \frac{1}{n_0} \sum_{i: T_i=0} Y_i
$$

**优点**: 无偏估计
**缺点**: 昂贵、不可行、伦理问题

### 2. 倾向得分匹配 (PSM)

**倾向得分** (Propensity Score):

$$
e(x) = P(T = 1 | X = x)
$$

**定理** (Rosenbaum & Rubin, 1983):

如果 $(Y(0), Y(1)) \perp T | X$，则 $(Y(0), Y(1)) \perp T | e(X)$

**估计步骤**:

1. 估计倾向得分 $\hat{e}(x)$ (如逻辑回归)
2. 匹配: 为每个治疗组个体找到倾向得分相近的对照组个体
3. 计算匹配后的平均差异

**ATT估计**:

$$
\widehat{\text{ATT}} = \frac{1}{n_1} \sum_{i: T_i=1} \left[ Y_i - \sum_{j: T_j=0} w_{ij} Y_j \right]
$$

其中 $w_{ij}$ 是匹配权重。

### 3. 工具变量 (IV)

**定义**: $Z$ 是 $X$ 对 $Y$ 的工具变量如果:

1. **相关性**: $Z$ 与 $X$ 相关
2. **排他性**: $Z$ 只通过 $X$ 影响 $Y$
3. **独立性**: $Z$ 与未观察混淆因子独立

**两阶段最小二乘 (2SLS)**:

**第一阶段**: 回归 $X$ 对 $Z$

$$
X = \alpha_0 + \alpha_1 Z + \nu
$$

**第二阶段**: 回归 $Y$ 对 $\hat{X}$

$$
Y = \beta_0 + \beta_1 \hat{X} + \epsilon
$$

**因果效应**: $\beta_1$

**Wald估计**:

$$
\beta = \frac{\mathbb{E}[Y | Z=1] - \mathbb{E}[Y | Z=0]}{\mathbb{E}[X | Z=1] - \mathbb{E}[X | Z=0]}
$$

### 4. 双重差分 (DID)

**场景**: 面板数据，治疗组和对照组，前后对比

**模型**:

$$
Y_{it} = \alpha + \beta \cdot \text{Treat}_i + \gamma \cdot \text{Post}_t + \delta \cdot (\text{Treat}_i \times \text{Post}_t) + \epsilon_{it}
$$

**因果效应**: $\delta$

**平行趋势假设**: 在无干预情况下，治疗组和对照组的趋势相同

### 5. 回归不连续 (RD)

**场景**: 治疗分配基于某个连续变量的阈值

**模型**:

$$
Y_i = \alpha + \tau D_i + \beta (X_i - c) + \epsilon_i
$$

其中 $D_i = \mathbb{1}(X_i \geq c)$

**因果效应**: $\tau$ (在阈值 $c$ 处的局部效应)

---

## 🧠 机器学习中的因果推断

### 1. 因果表示学习

**目标**: 学习因果不变的表示

**独立因果机制 (ICM)** 原则:

$$
P(X_1, \ldots, X_n) = \prod_i P(X_i | \text{PA}_i)
$$

每个机制 $P(X_i | \text{PA}_i)$ 独立变化。

**因果VAE**:

$$
\begin{align}
z &\sim P(z) \\
x &\sim P(x | z, u)
\end{align}
$$

其中 $z$ 是因果因子，$u$ 是非因果因子。

### 2. 反事实推理与解释性

**反事实解释**: "如果特征 $X_i$ 不同，预测会如何变化?"

**示例**: LIME, SHAP

**数学形式**:

$$
\text{Explanation}_i = f(x) - f(x_{-i}, x'_i)
$$

其中 $x'_i$ 是 $x_i$ 的反事实值。

### 3. 因果强化学习

**目标**: 学习因果策略，泛化到新环境

**因果MDP**:

$$
\begin{align}
s_{t+1} &= f_s(s_t, a_t, u_t) \\
r_t &= f_r(s_t, a_t, u_t)
\end{align}
$$

其中 $u_t$ 是未观察的混淆因子。

**因果Q-learning**:

$$
Q(s, a) = \mathbb{E}[R | do(s, a)]
$$

### 4. 迁移学习与域适应

**因果视角**: 因果关系在域间不变

**协变量偏移** (Covariate Shift):

$$
P_{\text{source}}(X) \neq P_{\text{target}}(X), \quad P_{\text{source}}(Y | X) = P_{\text{target}}(Y | X)
$$

**因果不变性**:

$$
P(Y | do(X)) \text{ 在域间不变}
$$

---

## 💻 Python实现

### 示例1: 因果图与d-分离

```python
import networkx as nx
import matplotlib.pyplot as plt

class CausalGraph:
    """因果图类"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_edge(self, cause, effect):
        """添加因果边"""
        self.graph.add_edge(cause, effect)
    
    def visualize(self):
        """可视化因果图"""
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, 
                node_color='lightblue', node_size=1500,
                font_size=12, font_weight='bold',
                arrows=True, arrowsize=20)
        plt.title("Causal Graph")
        plt.show()
    
    def is_d_separated(self, X, Y, Z):
        """
        检查X和Y是否被Z d-分离
        
        简化实现：检查是否存在未被阻断的路径
        """
        # 移除Z及其后代
        G_minus_Z = self.graph.copy()
        descendants_Z = set()
        for z in Z:
            descendants_Z.update(nx.descendants(G_minus_Z, z))
            descendants_Z.add(z)
        
        G_minus_Z.remove_nodes_from(descendants_Z)
        
        # 检查X到Y是否有路径
        try:
            path = nx.shortest_path(G_minus_Z.to_undirected(), X, Y)
            return False  # 存在路径，未d-分离
        except nx.NetworkXNoPath:
            return True  # 无路径，d-分离

# 示例：混淆因子
G = CausalGraph()
G.add_edge('Z', 'X')  # Z -> X
G.add_edge('Z', 'Y')  # Z -> Y
G.add_edge('X', 'Y')  # X -> Y

print("因果图结构:")
print("Z -> X, Z -> Y, X -> Y")

# 检查d-分离
print(f"\nX和Y是否被空集d-分离? {G.is_d_separated('X', 'Y', [])}")
print(f"X和Y是否被{{Z}}d-分离? {G.is_d_separated('X', 'Y', ['Z'])}")

G.visualize()
```

### 示例2: 倾向得分匹配

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(X, T, Y, n_neighbors=1):
    """
    倾向得分匹配
    
    Args:
        X: 协变量 (n_samples, n_features)
        T: 治疗指示 (n_samples,)
        Y: 结果 (n_samples,)
        n_neighbors: 匹配的邻居数
    
    Returns:
        ATT: 处理组平均因果效应
    """
    # 步骤1: 估计倾向得分
    ps_model = LogisticRegression()
    ps_model.fit(X, T)
    propensity_scores = ps_model.predict_proba(X)[:, 1]
    
    # 步骤2: 匹配
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    # 为每个治疗组个体找到最近的对照组个体
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(propensity_scores[control_idx].reshape(-1, 1))
    
    distances, indices = nn.kneighbors(
        propensity_scores[treated_idx].reshape(-1, 1)
    )
    
    # 步骤3: 计算ATT
    treated_outcomes = Y[treated_idx]
    matched_control_outcomes = np.mean(
        Y[control_idx[indices]], axis=1
    )
    
    ATT = np.mean(treated_outcomes - matched_control_outcomes)
    
    return ATT, propensity_scores

# 模拟数据
np.random.seed(42)
n = 1000

# 协变量
X = np.random.randn(n, 3)

# 倾向得分（真实）
true_ps = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))

# 治疗分配
T = (np.random.rand(n) < true_ps).astype(int)

# 潜在结果
Y0 = X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5
Y1 = Y0 + 2  # 真实ATE = 2

# 观察结果
Y = T * Y1 + (1 - T) * Y0

# 真实ATT
true_ATT = np.mean(Y1[T == 1] - Y0[T == 1])

# PSM估计
estimated_ATT, ps = propensity_score_matching(X, T, Y)

print(f"真实ATT: {true_ATT:.4f}")
print(f"估计ATT (PSM): {estimated_ATT:.4f}")
print(f"估计误差: {abs(estimated_ATT - true_ATT):.4f}")

# 可视化倾向得分分布
plt.figure(figsize=(10, 5))
plt.hist(ps[T == 0], bins=30, alpha=0.5, label='Control', density=True)
plt.hist(ps[T == 1], bins=30, alpha=0.5, label='Treated', density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distribution')
plt.legend()
plt.show()
```

### 示例3: 工具变量估计

```python
from sklearn.linear_model import LinearRegression

def instrumental_variable_2sls(Z, X, Y):
    """
    两阶段最小二乘 (2SLS) 工具变量估计
    
    Args:
        Z: 工具变量 (n_samples, n_instruments)
        X: 内生变量 (n_samples, n_endogenous)
        Y: 结果变量 (n_samples,)
    
    Returns:
        beta: 因果效应估计
    """
    # 第一阶段: X ~ Z
    first_stage = LinearRegression()
    first_stage.fit(Z, X)
    X_hat = first_stage.predict(Z)
    
    # 第二阶段: Y ~ X_hat
    second_stage = LinearRegression()
    second_stage.fit(X_hat.reshape(-1, 1), Y)
    beta = second_stage.coef_[0]
    
    return beta

# 模拟数据
np.random.seed(42)
n = 1000

# 工具变量
Z = np.random.randn(n)

# 未观察混淆因子
U = np.random.randn(n)

# 内生变量 (受Z和U影响)
X = 0.5 * Z + 0.3 * U + np.random.randn(n) * 0.1

# 结果变量 (真实因果效应 = 2)
Y = 2 * X + 0.5 * U + np.random.randn(n) * 0.5

# OLS估计 (有偏)
ols = LinearRegression()
ols.fit(X.reshape(-1, 1), Y)
beta_ols = ols.coef_[0]

# IV估计 (无偏)
beta_iv = instrumental_variable_2sls(Z.reshape(-1, 1), X, Y)

print(f"真实因果效应: 2.0000")
print(f"OLS估计 (有偏): {beta_ols:.4f}")
print(f"IV估计 (无偏): {beta_iv:.4f}")
```

### 示例4: 因果发现

```python
from itertools import permutations

def pc_algorithm_simple(data, alpha=0.05):
    """
    简化的PC算法用于因果发现
    
    Args:
        data: 数据矩阵 (n_samples, n_variables)
        alpha: 显著性水平
    
    Returns:
        adjacency_matrix: 邻接矩阵
    """
    from scipy.stats import pearsonr
    
    n_vars = data.shape[1]
    
    # 初始化完全图
    adj_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    
    # 步骤1: 移除条件独立的边
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # 测试边缘独立性
            corr, p_value = pearsonr(data[:, i], data[:, j])
            
            if p_value > alpha:
                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
                continue
            
            # 测试条件独立性 (简化: 只测试一阶)
            for k in range(n_vars):
                if k != i and k != j:
                    # 部分相关
                    corr_ik, _ = pearsonr(data[:, i], data[:, k])
                    corr_jk, _ = pearsonr(data[:, j], data[:, k])
                    corr_ij, _ = pearsonr(data[:, i], data[:, j])
                    
                    partial_corr = (corr_ij - corr_ik * corr_jk) / \
                                   np.sqrt((1 - corr_ik**2) * (1 - corr_jk**2))
                    
                    # Fisher's z-transformation
                    n = data.shape[0]
                    z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
                    p_value = 2 * (1 - norm.cdf(abs(z) * np.sqrt(n - 3)))
                    
                    if p_value > alpha:
                        adj_matrix[i, j] = 0
                        adj_matrix[j, i] = 0
                        break
    
    return adj_matrix

# 模拟因果数据
from scipy.stats import norm

np.random.seed(42)
n = 500

# 真实因果结构: X -> Y -> Z
X = np.random.randn(n)
Y = 0.8 * X + np.random.randn(n) * 0.3
Z = 0.7 * Y + np.random.randn(n) * 0.3

data = np.column_stack([X, Y, Z])

# 因果发现
adj_matrix = pc_algorithm_simple(data, alpha=0.05)

print("发现的因果结构 (邻接矩阵):")
print(adj_matrix)
print("\n真实因果结构: X -> Y -> Z")
```

---

## 🎓 对标课程

| 大学 | 课程代码 | 课程名称 |
|------|----------|----------|
| **Stanford** | STATS361 | Causal Inference |
| **MIT** | 14.387 | Applied Econometrics: Mostly Harmless |
| **UC Berkeley** | PH252D | Causal Inference |
| **Harvard** | STAT186 | Causal Inference |
| **CMU** | 10-708 | Probabilistic Graphical Models |

---

## 📖 核心教材与论文

### 教材

1. **Pearl, J.** *Causality: Models, Reasoning, and Inference*. Cambridge University Press, 2009.
   - 因果推断的经典教材

2. **Imbens, G. & Rubin, D.** *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press, 2015.
   - 潜在结果框架

3. **Hernán, M. & Robins, J.** *Causal Inference: What If*. Chapman & Hall/CRC, 2020.
   - 流行病学视角

4. **Peters, J., Janzing, D., & Schölkopf, B.** *Elements of Causal Inference*. MIT Press, 2017.
   - 机器学习视角

### 经典论文

1. **Rubin, D. (1974)** - *Estimating Causal Effects of Treatments*
   - 潜在结果框架

2. **Pearl, J. (1995)** - *Causal Diagrams for Empirical Research*
   - 因果图模型

3. **Rosenbaum, P. & Rubin, D. (1983)** - *The Central Role of the Propensity Score*
   - 倾向得分

4. **Angrist, J. et al. (1996)** - *Identification of Causal Effects Using Instrumental Variables*
   - 工具变量

### 最新进展 (2024-2025)

1. **因果表示学习**
   - 学习因果不变的表示

2. **因果强化学习**
   - 泛化到新环境的策略

3. **因果LLM**
   - 大语言模型的因果推理能力

4. **因果发现的深度学习方法**
   - 神经网络用于因果结构学习

---

## 🔗 相关主题

- [概率统计](../../01-Mathematical-Foundations/02-Probability-Statistics/)
- [统计学习理论](../../02-Machine-Learning-Theory/01-Statistical-Learning/)
- [强化学习](../../02-Machine-Learning-Theory/04-Reinforcement-Learning/)

---

## 📝 总结

**因果推断**是从观察数据中推断因果关系的科学，核心包括:

### 核心概念

1. **因果关系 vs 相关关系**: 相关不等于因果
2. **反事实推理**: "如果...会怎样?"
3. **因果效应**: ATE, ATT, CATE

### 理论框架

1. **Rubin因果模型**: 潜在结果框架
2. **Pearl因果模型**: 结构方程与因果图
3. **统一**: 两种框架的等价性

### 识别方法

1. **后门准则**: 调整混淆因子
2. **前门准则**: 利用中介变量
3. **do-演算**: 系统化的识别方法

### 估计方法

1. **RCT**: 黄金标准
2. **PSM**: 倾向得分匹配
3. **IV**: 工具变量
4. **DID**: 双重差分
5. **RD**: 回归不连续

### AI应用

1. **因果表示学习**: 学习因果不变特征
2. **反事实解释**: 可解释AI
3. **因果强化学习**: 泛化策略
4. **域适应**: 利用因果不变性

**未来方向**:

- 因果发现的深度学习方法
- 因果LLM
- 可验证的因果推断
- 因果与公平性

因果推断为AI提供了从"预测"到"理解"和"干预"的桥梁！

---

**© 2025 AI Mathematics and Science Knowledge System**-

*Building the mathematical foundations for the AI era*-

*最后更新：2025年10月5日*-
