# Basic Probability Theory

## Overview

Basic probability theory provides the mathematical foundation for understanding randomness, uncertainty, and stochastic phenomena. It establishes the formal framework for quantifying likelihood and analyzing random events through rigorous mathematical structures.

## Historical Development

### Early Origins

- **Ancient civilizations**: Games of chance, divination practices
- **17th century**: Pascal and Fermat's correspondence on gambling problems
- **18th century**: Bernoulli's work on the law of large numbers
- **19th century**: Laplace's comprehensive treatment of probability

### Modern Formalization

- **1933**: Kolmogorov's axiomatic foundation
- **20th century**: Measure-theoretic approach
- **Contemporary**: Integration with statistics and applications

## Fundamental Concepts

### Sample Space and Events

**Definition 1.1** (Sample Space)
A sample space Ω is a non-empty set representing all possible outcomes of a random experiment.

**Definition 1.2** (Event)
An event is a subset of the sample space Ω. The collection of all events forms a σ-algebra ℱ.

**Example 1.1** (Coin Toss)

- Sample space: Ω = {H, T}
- Events: ∅, {H}, {T}, {H, T}

### Probability Measure

**Axiom 1.1** (Kolmogorov Axioms)
A probability measure P is a function P: ℱ → [0,1] satisfying:

1. P(Ω) = 1
2. P(A) ≥ 0 for all A ∈ ℱ
3. Countable additivity: P(∪ᵢAᵢ) = ΣᵢP(Aᵢ) for disjoint events

**Theorem 1.1** (Basic Properties)
For any events A, B ∈ ℱ:

1. P(∅) = 0
2. P(Aᶜ) = 1 - P(A)
3. P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

## Conditional Probability and Independence

### Conditional Probability

**Definition 1.3** (Conditional Probability)
For events A, B with P(B) > 0:
P(A|B) = P(A ∩ B) / P(B)

**Theorem 1.2** (Bayes' Theorem)
P(A|B) = P(B|A)P(A) / P(B)

### Independence

**Definition 1.4** (Independence)
Events A and B are independent if P(A ∩ B) = P(A)P(B).

**Definition 1.5** (Conditional Independence)
Events A and B are conditionally independent given C if:
P(A ∩ B|C) = P(A|C)P(B|C)

## Random Variables

### Definition and Types

**Definition 1.6** (Random Variable)
A random variable X is a measurable function X: Ω → ℝ.

**Definition 1.7** (Types of Random Variables)

- **Discrete**: Takes countable values
- **Continuous**: Takes uncountable values
- **Mixed**: Combination of discrete and continuous

### Distribution Functions

**Definition 1.8** (Cumulative Distribution Function)
F_X(x) = P(X ≤ x)

**Properties:**

1. Non-decreasing
2. Right-continuous
3. lim_{x→-∞} F_X(x) = 0
4. lim_{x→∞} F_X(x) = 1

**Definition 1.9** (Probability Mass Function)
For discrete X: p_X(x) = P(X = x)

**Definition 1.10** (Probability Density Function)
For continuous X: f_X(x) = dF_X(x)/dx

## Expectation and Moments

### Expectation

**Definition 1.11** (Expectation)
E[X] = ∫ x dF_X(x) = Σ x p_X(x) (discrete) or ∫ x f_X(x) dx (continuous)

**Properties:**

1. Linearity: E[aX + bY] = aE[X] + bE[Y]
2. Monotonicity: If X ≤ Y, then E[X] ≤ E[Y]

### Moments and Variance

**Definition 1.12** (k-th Moment)
μ_k = E[X^k]

**Definition 1.13** (Variance)
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²

**Properties:**

1. Var(aX + b) = a²Var(X)
2. Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)

## Important Distributions

### Discrete Distributions

**Definition 1.14** (Bernoulli Distribution)
X ~ Bernoulli(p): P(X = 1) = p, P(X = 0) = 1-p

- E[X] = p
- Var(X) = p(1-p)

**Definition 1.15** (Binomial Distribution)
X ~ Binomial(n,p): P(X = k) = C(n,k) p^k (1-p)^(n-k)

- E[X] = np
- Var(X) = np(1-p)

**Definition 1.16** (Poisson Distribution)
X ~ Poisson(λ): P(X = k) = e^(-λ) λ^k / k!

- E[X] = λ
- Var(X) = λ

### Continuous Distributions

**Definition 1.17** (Uniform Distribution)
X ~ Uniform(a,b): f_X(x) = 1/(b-a) for x ∈ [a,b]

- E[X] = (a+b)/2
- Var(X) = (b-a)²/12

**Definition 1.18** (Normal Distribution)
X ~ N(μ,σ²): f_X(x) = (1/√(2πσ²)) e^(-(x-μ)²/(2σ²))

- E[X] = μ
- Var(X) = σ²

**Definition 1.19** (Exponential Distribution)
X ~ Exponential(λ): f_X(x) = λe^(-λx) for x ≥ 0

- E[X] = 1/λ
- Var(X) = 1/λ²

## Convergence Concepts

### Types of Convergence

**Definition 1.20** (Almost Sure Convergence)
X_n → X a.s. if P(lim_{n→∞} X_n = X) = 1

**Definition 1.21** (Convergence in Probability)
X_n → X in probability if lim_{n→∞} P(|X_n - X| > ε) = 0

**Definition 1.22** (Convergence in Distribution)
X_n → X in distribution if lim_{n→∞} F_{X_n}(x) = F_X(x)

### Laws of Large Numbers

**Theorem 1.3** (Weak Law of Large Numbers)
For i.i.d. random variables X₁, X₂, ... with finite mean μ:
(1/n)Σᵢ₌₁ⁿ Xᵢ → μ in probability

**Theorem 1.4** (Strong Law of Large Numbers)
For i.i.d. random variables X₁, X₂, ... with finite mean μ:
(1/n)Σᵢ₌₁ⁿ Xᵢ → μ almost surely

### Central Limit Theorem

**Theorem 1.5** (Central Limit Theorem)
For i.i.d. random variables X₁, X₂, ... with mean μ and variance σ²:
√n((1/n)Σᵢ₌₁ⁿ Xᵢ - μ) → N(0,σ²) in distribution

## Applications

### Risk Assessment

- Financial modeling
- Insurance calculations
- Quality control

### Decision Theory

- Bayesian inference
- Statistical hypothesis testing
- Machine learning

### Information Theory

- Entropy calculations
- Data compression
- Communication systems

## References

1. Kolmogorov, A.N. (1933). "Grundbegriffe der Wahrscheinlichkeitsrechnung"
2. Billingsley, P. (1995). "Probability and Measure"
3. Durrett, R. (2019). "Probability: Theory and Examples"
4. Feller, W. (1968). "An Introduction to Probability Theory and Its Applications"

## Cross-References

- [Statistics and Data Analysis](../02_Statistics_Data_Analysis.md)
- [Stochastic Processes](../03_Stochastic_Processes.md)
- [Information Theory](../04_Information_Theory.md)
- [Decision Theory](../05_Decision_Theory.md)
