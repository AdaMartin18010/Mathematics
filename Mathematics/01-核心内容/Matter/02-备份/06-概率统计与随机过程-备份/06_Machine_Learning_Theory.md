# Machine Learning Theory

## Overview

Machine learning theory provides the mathematical foundation for understanding how algorithms can learn from data. It encompasses statistical learning theory, computational learning theory, and optimization theory, establishing fundamental limits and guarantees for learning algorithms.

## Historical Development

### Early Foundations

- **1950s**: Perceptron learning algorithm
- **1960s**: Pattern recognition theory
- **1970s**: PAC learning framework
- **1980s**: Statistical learning theory

### Modern Development

- **1990s**: Support vector machines and kernel methods
- **2000s**: Boosting and ensemble methods
- **2010s**: Deep learning theory
- **2020s**: Federated learning and privacy-preserving ML

## Statistical Learning Theory

### Learning Framework

**Definition 6.1** (Learning Problem)
A tuple (X, Y, P, H, L) where:

- X: Input space
- Y: Output space
- P: Data distribution on X × Y
- H: Hypothesis class
- L: Loss function L: Y × Y → ℝ

**Definition 6.2** (Risk)
R(h) = E_{(x,y)~P}[L(h(x), y)]

**Definition 6.3** (Empirical Risk)
R̂(h) = (1/n)Σᵢ₌₁ⁿ L(h(xᵢ), yᵢ)

**Definition 6.4** (ERM Principle)
ĥ = argmin_{h∈H} R̂(h)

### Generalization Bounds

**Definition 6.5** (Generalization Gap)
R(ĥ) - R̂(ĥ)

**Theorem 6.1** (Uniform Convergence)
With probability at least 1-δ:
sup_{h∈H} |R(h) - R̂(h)| ≤ O(√(log|H|/n))

**Theorem 6.2** (VC Dimension Bound)
For binary classification with VC dimension d:
R(ĥ) ≤ R̂(ĥ) + O(√(d log(n/d)/n))

**Definition 6.6** (VC Dimension)
The largest number of points that can be shattered by H.

**Example 6.1** (Linear Classifiers)
VC dimension of linear classifiers in ℝ^d is d+1.

### Rademacher Complexity

**Definition 6.7** (Rademacher Complexity)
Rₙ(H) = E[sup_{h∈H} (1/n)Σᵢ₌₁ⁿ σᵢh(xᵢ)]

where σᵢ are i.i.d. Rademacher random variables.

**Theorem 6.3** (Rademacher Bound)
With probability at least 1-δ:
R(ĥ) ≤ R̂(ĥ) + 2Rₙ(H) + 3√(log(2/δ)/(2n))

**Theorem 6.4** (Rademacher for Linear Classes)
For H = {x ↦ ⟨w,x⟩ : ||w||₂ ≤ B}:
Rₙ(H) ≤ B maxᵢ ||xᵢ||₂ / √n

## PAC Learning

### PAC Framework

**Definition 6.8** (PAC Learnable)
A concept class C is PAC learnable if there exists an algorithm A such that for all ε, δ > 0 and distributions P, with probability at least 1-δ:
R(A(S)) ≤ min_{c∈C} R(c) + ε

**Definition 6.9** (Sample Complexity)
The minimum number of samples needed for PAC learning.

**Theorem 6.5** (Sample Complexity Bound)
For finite hypothesis class H:
m(ε, δ) = O((log|H| + log(1/δ))/ε²)

### Agnostic PAC Learning

**Definition 6.10** (Agnostic PAC Learnable)
A hypothesis class H is agnostically PAC learnable if for all ε, δ > 0 and distributions P, with probability at least 1-δ:
R(A(S)) ≤ min_{h∈H} R(h) + ε

**Theorem 6.6** (Agnostic Sample Complexity)
For finite hypothesis class H:
m(ε, δ) = O((log|H| + log(1/δ))/ε²)

## Online Learning

### Online Learning Framework

**Definition 6.11** (Online Learning)
Learning in rounds where in each round t:

1. Learner predicts ŷₜ
2. Nature reveals yₜ
3. Learner suffers loss L(ŷₜ, yₜ)

**Definition 6.12** (Regret)
Regret = Σₜ₌₁ᵀ L(ŷₜ, yₜ) - min_{h∈H} Σₜ₌₁ᵀ L(h(xₜ), yₜ)

### Online Gradient Descent

**Algorithm 6.1** (Online Gradient Descent)

1. Initialize w₁ = 0
2. For t = 1, 2, ..., T:
   - Predict ŷₜ = ⟨wₜ, xₜ⟩
   - Receive yₜ and loss L(ŷₜ, yₜ)
   - Update w_{t+1} = wₜ - η∇L(ŷₜ, yₜ)

**Theorem 6.7** (OGD Regret Bound)
For convex loss functions:
Regret ≤ O(√T)

### Follow the Regularized Leader

**Algorithm 6.2** (FTRL)
w_{t+1} = argmin_w [Σₛ₌₁ᵗ L(⟨w, xₛ⟩, yₛ) + R(w)]

**Theorem 6.8** (FTRL Regret Bound)
For strongly convex regularizer R:
Regret ≤ O(log T)

## Kernel Methods

### Reproducing Kernel Hilbert Spaces

**Definition 6.13** (Kernel Function)
A symmetric function k: X × X → ℝ is a kernel if it is positive semidefinite.

**Definition 6.14** (RKHS)
A Hilbert space H of functions f: X → ℝ with reproducing property:
f(x) = ⟨f, k(x,·)⟩_H

**Theorem 6.9** (Representer Theorem)
For any loss function L and regularization term R:
f* = argmin_{f∈H} [Σᵢ₌₁ⁿ L(f(xᵢ), yᵢ) + R(||f||_H)]

has the form f* = Σᵢ₌₁ⁿ αᵢk(xᵢ,·).

### Support Vector Machines

**Definition 6.15** (SVM Optimization)
min_{w,b} (1/2)||w||² + C Σᵢ₌₁ⁿ ξᵢ
subject to yᵢ(⟨w, xᵢ⟩ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

**Theorem 6.10** (SVM Dual)
The dual problem is:
max_α Σᵢ₌₁ⁿ αᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼk(xᵢ, xⱼ)
subject to 0 ≤ αᵢ ≤ C, Σᵢ αᵢyᵢ = 0

## Boosting

### AdaBoost Algorithm

**Algorithm 6.3** (AdaBoost)

1. Initialize D₁(i) = 1/n
2. For t = 1, 2, ..., T:
   - Train weak learner hₜ with distribution Dₜ
   - Compute error εₜ = Σᵢ Dₜ(i)I(hₜ(xᵢ) ≠ yᵢ)
   - Set αₜ = (1/2)log((1-εₜ)/εₜ)
   - Update D_{t+1}(i) ∝ Dₜ(i)exp(-αₜyᵢhₜ(xᵢ))
3. Output H(x) = sign(Σₜ αₜhₜ(x))

**Theorem 6.11** (AdaBoost Bound)
Training error ≤ exp(-2Σₜ γₜ²) where γₜ = 1/2 - εₜ

### Gradient Boosting

**Algorithm 6.4** (Gradient Boosting)

1. Initialize F₀(x) = 0
2. For t = 1, 2, ..., T:
   - Compute residuals rᵢ = yᵢ - F_{t-1}(xᵢ)
   - Fit weak learner hₜ to (xᵢ, rᵢ)
   - Update Fₜ(x) = F_{t-1}(x) + ηhₜ(x)

## Deep Learning Theory

### Universal Approximation

**Theorem 6.12** (Universal Approximation)
For any continuous function f: [0,1]^d → ℝ and ε > 0, there exists a neural network with one hidden layer that approximates f within ε.

**Theorem 6.13** (Deep Universal Approximation)
Deep networks can approximate functions with exponentially fewer parameters than shallow networks.

### Optimization Landscape

**Definition 6.16** (Critical Point)
A point w where ∇f(w) = 0.

**Definition 6.17** (Saddle Point)
A critical point that is neither a local minimum nor maximum.

**Theorem 6.14** (Saddle Point Escape)
Gradient descent can escape strict saddle points almost surely.

### Generalization in Deep Learning

**Definition 6.18** (Double Descent)
The phenomenon where test error decreases, then increases, then decreases again as model complexity increases.

**Theorem 6.15** (Interpolation Threshold)
For overparameterized models, interpolation (zero training error) can lead to good generalization.

## Federated Learning

### Federated Learning Framework

**Definition 6.19** (Federated Learning)
Learning from decentralized data without sharing raw data.

**Algorithm 6.5** (FedAvg)

1. Server initializes global model w⁰
2. For t = 1, 2, ..., T:
   - Server sends w^t to clients
   - Each client k updates w_k^t = w^t - η∇f_k(w^t)
   - Server aggregates w^{t+1} = (1/K)Σₖ w_k^t

**Theorem 6.16** (FedAvg Convergence)
Under certain conditions, FedAvg converges to a stationary point of the global objective.

## Privacy-Preserving Learning

### Differential Privacy

**Definition 6.20** (Differential Privacy)
An algorithm A is (ε, δ)-differentially private if for all neighboring datasets D, D':
P(A(D) ∈ S) ≤ e^ε P(A(D') ∈ S) + δ

**Definition 6.21** (Neighboring Datasets)
Two datasets are neighboring if they differ in at most one record.

**Theorem 6.17** (Composition)
If A₁ is (ε₁, δ₁)-DP and A₂ is (ε₂, δ₂)-DP, then (A₁, A₂) is (ε₁+ε₂, δ₁+δ₂)-DP.

### Private Learning

**Algorithm 6.6** (Private SGD)

1. Initialize w₁ = 0
2. For t = 1, 2, ..., T:
   - Sample batch Bₜ
   - Compute gradient gₜ = (1/|Bₜ|)Σᵢ∈Bₜ ∇L(wₜ, xᵢ, yᵢ)
   - Add noise: g̃ₜ = gₜ + N(0, σ²I)
   - Update w_{t+1} = wₜ - ηg̃ₜ

**Theorem 6.18** (Privacy Guarantee)
Private SGD is (ε, δ)-differentially private for appropriate noise scale σ.

## Applications

### Computer Vision

- Image classification
- Object detection
- Semantic segmentation

### Natural Language Processing

- Language modeling
- Machine translation
- Question answering

### Speech Recognition

- Automatic speech recognition
- Speaker identification
- Emotion recognition

### Healthcare

- Medical diagnosis
- Drug discovery
- Personalized medicine

## References

1. Vapnik, V.N. (1998). "Statistical Learning Theory"
2. Shalev-Shwartz, S. & Ben-David, S. (2014). "Understanding Machine Learning"
3. Mohri, M. et al. (2018). "Foundations of Machine Learning"
4. Goodfellow, I. et al. (2016). "Deep Learning"

## Cross-References

- [Basic Probability Theory](../01_Basic_Probability_Theory.md)
- [Statistics and Data Analysis](../02_Statistics_Data_Analysis.md)
- [Optimization Theory](../14_Optimization_Theory.md)
- [Neural Network Theory](../15_Neural_Network_Theory.md)
