# 因果推断 (Causal Inference)

**创建日期**: 2025-12-20
**版本**: v1.0
**状态**: 进行中
**适用范围**: AI-Mathematics模块 - 因果推断子模块

> **From Correlation to Causation**
>
> 从相关到因果：理解"为什么"而非仅仅"是什么"

---

## 📋 模块概述

**因果推断**研究如何从观察数据中推断因果关系，是统计学、机器学习、经济学、流行病学等领域的核心问题。

**为什么重要**:

- **预测 vs 干预**: 相关性预测未来，因果性指导干预
- **可解释性**: 理解"为什么"而非仅仅"是什么"
- **泛化能力**: 因果模型在分布变化下更鲁棒
- **决策支持**: 评估政策、治疗、干预的效果

---

## 📚 核心内容

### 1. [因果推断理论](./01-Causal-Inference-Theory.md) ✅

**核心主题**:

- **基础概念**
  - 因果关系 vs 相关关系
  - 反事实推理 (Counterfactual)
  - 因果效应 (ATE, ATT, CATE)

- **理论框架**
  - Rubin因果模型 (潜在结果框架)
  - Pearl因果模型 (结构方程框架)
  - 两种框架的统一

- **因果图模型**
  - 结构因果模型 (SCM)
  - 有向无环图 (DAG)
  - d-分离准则

- **因果识别**
  - 后门准则 (Backdoor Criterion)
  - 前门准则 (Frontdoor Criterion)
  - do-演算 (do-Calculus)

- **因果估计方法**
  - 随机对照试验 (RCT)
  - 倾向得分匹配 (PSM)
  - 工具变量 (IV)
  - 双重差分 (DID)
  - 回归不连续 (RD)

- **机器学习中的因果推断**
  - 因果表示学习
  - 反事实推理与解释性
  - 因果强化学习
  - 迁移学习与域适应

**Python实现**:

- 因果图与d-分离
- 倾向得分匹配
- 工具变量估计 (2SLS)
- 因果发现 (PC算法)

**对标课程**:

- Stanford STATS361 - Causal Inference
- MIT 14.387 - Applied Econometrics
- UC Berkeley PH252D - Causal Inference
- Harvard STAT186 - Causal Inference
- CMU 10-708 - Probabilistic Graphical Models

---

## 🎯 学习路径

### 初级 (基础概念)

1. **因果关系 vs 相关关系**
   - Simpson悖论
   - 混淆因子 (Confounder)

2. **反事实推理**
   - 潜在结果 $Y(0), Y(1)$
   - 因果效应定义

3. **随机对照试验 (RCT)**
   - 黄金标准
   - 无偏估计

### 中级 (理论框架)

1. **Rubin因果模型**
   - SUTVA假设
   - 可忽略性 (Ignorability)
   - 正性 (Positivity)

2. **Pearl因果模型**
   - 结构因果模型 (SCM)
   - do-算子
   - 因果图 (DAG)

3. **因果识别**
   - 后门准则
   - 前门准则
   - do-演算

### 高级 (估计与应用)

1. **因果估计方法**
   - 倾向得分匹配 (PSM)
   - 工具变量 (IV)
   - 双重差分 (DID)
   - 回归不连续 (RD)

2. **因果发现**
   - PC算法
   - FCI算法
   - 约束条件与独立性测试

3. **机器学习中的因果推断**
   - 因果表示学习
   - 因果强化学习
   - 反事实解释 (LIME, SHAP)
   - 域适应与迁移学习

---

## 🔗 与其他模块的联系

### 数学基础

- **概率统计** → 条件独立、贝叶斯推断
- **优化理论** → 因果效应估计

### 机器学习理论

- **统计学习理论** → 泛化与因果不变性
- **强化学习** → 因果MDP、因果Q-learning

### 前沿研究

- **LLM理论** → 因果推理能力
- **生成模型** → 因果VAE、因果GAN

---

## 📊 模块统计

| 指标 | 数值 |
|------|------|
| **文档数** | 1 |
| **内容量** | ~78 KB |
| **数学公式** | 150+ |
| **代码示例** | 4 |
| **Python实现** | 4 |
| **对标课程** | 5 |
| **完成度** | **100%** ✅ |

---

## 🎓 推荐学习资源

### 教材

1. **Pearl, J.** *Causality: Models, Reasoning, and Inference*. Cambridge University Press, 2009.
   - 因果推断的经典教材

2. **Imbens, G. & Rubin, D.** *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press, 2015.
   - 潜在结果框架

3. **Hernán, M. & Robins, J.** *Causal Inference: What If*. Chapman & Hall/CRC, 2020.
   - 流行病学视角

4. **Peters, J., Janzing, D., & Schölkopf, B.** *Elements of Causal Inference*. MIT Press, 2017.
   - 机器学习视角

### 在线课程

- **Stanford STATS361** - Causal Inference
- **MIT 14.387** - Applied Econometrics: Mostly Harmless
- **UC Berkeley PH252D** - Causal Inference
- **Harvard STAT186** - Causal Inference

### Python库

- **DoWhy** - 微软开源的因果推断库
- **CausalML** - Uber开源的因果机器学习库
- **EconML** - 微软开源的经济因果推断库
- **pgmpy** - 概率图模型库

---

## 🚀 应用场景

### 医疗健康

- **临床试验**: 评估治疗效果
- **流行病学**: 识别疾病风险因素
- **个性化医疗**: 异质性治疗效应 (HTE)

### 经济学

- **政策评估**: 评估政策干预效果
- **劳动经济学**: 教育回报、工资差异
- **发展经济学**: 扶贫项目评估

### 科技产业

- **A/B测试**: 因果效应估计
- **推荐系统**: 去偏与因果推荐
- **广告投放**: ROI评估
- **用户增长**: 干预策略优化

### 机器学习

- **可解释AI**: 反事实解释
- **公平性**: 去除歧视性偏见
- **域适应**: 利用因果不变性
- **强化学习**: 因果策略学习

---

## 💡 核心洞察

### 1. 相关不等于因果

**Simpson悖论**告诉我们，相关性可能误导因果推断。必须通过因果图、实验设计或识别假设来建立因果关系。

### 2. 反事实推理的根本问题

我们只能观察到 $Y_i(T_i)$，无法同时观察 $Y_i(1)$ 和 $Y_i(0)$。因果推断的核心是如何从观察数据中推断反事实。

### 3. 两种框架的统一

Rubin的潜在结果框架和Pearl的结构方程框架在可忽略性假设下是等价的，提供了不同的视角和工具。

### 4. 因果识别的关键

- **后门准则**: 调整混淆因子
- **前门准则**: 利用中介变量
- **do-演算**: 系统化的识别方法

### 5. 机器学习中的因果革命

因果推断为机器学习提供了从"预测"到"理解"和"干预"的桥梁，是可解释AI、公平性、域适应的理论基础。

---

## 📝 更新记录

### 2025-10-05

- ✅ 创建因果推断理论文档 (78KB)
- ✅ 涵盖Rubin/Pearl框架、因果图、识别与估计
- ✅ 包含4个Python实现示例
- ✅ 对标5门世界顶尖大学课程
- ✅ **模块100%完成** 🎉

---

## 🎯 下一步

因果推断模块已100%完成！可以继续探索：

1. **因果发现的深度学习方法**
   - 神经网络用于因果结构学习
   - 基于注意力机制的因果发现

2. **因果LLM**
   - 大语言模型的因果推理能力
   - Prompt Engineering for Causal Reasoning

3. **因果与公平性**
   - 因果公平性定义
   - 去除歧视性偏见

4. **因果强化学习**
   - 因果MDP
   - 泛化到新环境的策略

---

**© 2025 AI Mathematics and Science Knowledge System**-

*Building the mathematical foundations for the AI era*-

**模块状态**: ✅ **100% 完成**

**最后更新**: 2025年10月5日
