# Decision Theory

## Overview

Decision theory provides the mathematical framework for making optimal decisions under uncertainty. It combines probability theory, utility theory, and optimization to analyze decision-making processes in economics, statistics, artificial intelligence, and other fields.

## Historical Development

### Early Foundations

- **1738**: Bernoulli's utility theory
- **1921**: Knight's distinction between risk and uncertainty
- **1944**: von Neumann and Morgenstern's expected utility theory
- **1954**: Savage's subjective expected utility theory

### Modern Development

- **1960s**: Development of Bayesian decision theory
- **1970s**: Prospect theory and behavioral economics
- **1980s**: Multi-criteria decision analysis
- **1990s-present**: Computational decision theory

## Utility Theory

### Expected Utility Theory

**Definition 5.1** (Lottery)
A lottery L = (p₁, x₁; p₂, x₂; ...; pₙ, xₙ) where pᵢ ≥ 0, Σpᵢ = 1, and xᵢ are outcomes.

**Definition 5.2** (Preference Relation)
A binary relation ⪰ on the set of lotteries satisfying:

1. Completeness: L ⪰ M or M ⪰ L
2. Transitivity: If L ⪰ M and M ⪰ N, then L ⪰ N

**Axiom 5.1** (Independence Axiom)
If L ⪰ M, then αL + (1-α)N ⪰ αM + (1-α)N for all α ∈ (0,1) and N.

**Axiom 5.2** (Continuity Axiom)
If L ⪰ M ⪰ N, then there exists α ∈ [0,1] such that M ~ αL + (1-α)N.

**Theorem 5.1** (Expected Utility Theorem)
If preferences satisfy completeness, transitivity, independence, and continuity, then there exists a utility function u such that:
L ⪰ M if and only if Σᵢ pᵢu(xᵢ) ≥ Σᵢ qᵢu(xᵢ)

### Risk Attitudes

**Definition 5.3** (Risk Aversion)
A decision maker is risk averse if u(E[X]) ≥ E[u(X)] for all X.

**Definition 5.4** (Risk Seeking)
A decision maker is risk seeking if u(E[X]) ≤ E[u(X)] for all X.

**Definition 5.5** (Risk Neutral)
A decision maker is risk neutral if u(E[X]) = E[u(X)] for all X.

**Theorem 5.2** (Jensen's Inequality)
For a concave utility function u:
u(E[X]) ≥ E[u(X)]

### Certainty Equivalent

**Definition 5.6** (Certainty Equivalent)
The certainty equivalent CE of a lottery L is the amount such that:
u(CE) = Σᵢ pᵢu(xᵢ)

**Definition 5.7** (Risk Premium)
RP = E[X] - CE

**Theorem 5.3** (Risk Premium Approximation)
For small risks: RP ≈ -(1/2)u''(E[X])/u'(E[X]) × Var(X)

## Bayesian Decision Theory

### Decision Framework

**Definition 5.8** (Decision Problem)
A tuple (Θ, A, L) where:

- Θ: State space
- A: Action space
- L: Loss function L: Θ × A → ℝ

**Definition 5.9** (Decision Rule)
A function δ: X → A that maps observations to actions.

**Definition 5.10** (Risk Function)
R(θ, δ) = E[L(θ, δ(X))]

### Bayes Risk and Optimality

**Definition 5.11** (Bayes Risk)
r(π, δ) = E_π[R(θ, δ)] = ∫ R(θ, δ) π(θ) dθ

**Definition 5.12** (Bayes Rule)
δ^π = argmin_δ r(π, δ)

**Theorem 5.4** (Bayes Rule Characterization)
The Bayes rule minimizes posterior expected loss:
δ^π(x) = argmin_a E[L(θ, a) | X = x]

### Common Loss Functions

**Definition 5.13** (Squared Error Loss)
L(θ, a) = (θ - a)²

**Definition 5.14** (Absolute Error Loss)
L(θ, a) = |θ - a|

**Definition 5.15** (0-1 Loss)
L(θ, a) = I(θ ≠ a)

**Theorem 5.5** (Optimal Estimators)

- For squared error loss: δ^π(x) = E[θ | X = x]
- For absolute error loss: δ^π(x) = median(θ | X = x)
- For 0-1 loss: δ^π(x) = mode(θ | X = x)

## Hypothesis Testing

### Bayesian Testing

**Definition 5.16** (Posterior Probability)
P(H₀ | x) = P(x | H₀)π₀ / P(x)

**Definition 5.17** (Bayes Factor)
B = P(x | H₁) / P(x | H₀)

**Theorem 5.6** (Posterior Odds)
P(H₁ | x) / P(H₀ | x) = B × π₁ / π₀

### Decision-Theoretic Testing

**Definition 5.18** (Testing Loss Function)
L(θ, a) = c₀I(θ ∈ Θ₀, a = 1) + c₁I(θ ∈ Θ₁, a = 0)

**Theorem 5.7** (Optimal Test)
Reject H₀ if P(H₀ | x) < c₁/(c₀ + c₁)

## Game Theory

### Strategic Games

**Definition 5.19** (Strategic Game)
A tuple (N, A, u) where:

- N: Set of players
- A: Action profiles
- u: Utility functions uᵢ: A → ℝ

**Definition 5.20** (Nash Equilibrium)
A strategy profile a* is a Nash equilibrium if:
uᵢ(a*_i, a**{-i}) ≥ uᵢ(aᵢ, a**{-i}) for all i and aᵢ

**Theorem 5.8** (Nash Existence)
Every finite strategic game has at least one Nash equilibrium.

### Bayesian Games

**Definition 5.21** (Bayesian Game)
A tuple (N, Θ, A, p, u) where:

- N: Set of players
- Θ: Type space
- A: Action profiles
- p: Prior distribution on types
- u: Utility functions uᵢ: Θ × A → ℝ

**Definition 5.22** (Bayesian Nash Equilibrium)
A strategy profile s*is a Bayesian Nash equilibrium if:
E[uᵢ(θ, s*_i(θᵢ), s**{-i}(θ*{-i})) | θᵢ] ≥ E[uᵢ(θ, sᵢ(θᵢ), s**{-i}(θ*{-i})) | θᵢ]

## Multi-Criteria Decision Analysis

### Pareto Optimality

**Definition 5.23** (Pareto Dominance)
Alternative a dominates b if fᵢ(a) ≥ fᵢ(b) for all i and fⱼ(a) > fⱼ(b) for some j.

**Definition 5.24** (Pareto Optimal)
An alternative is Pareto optimal if it is not dominated by any other alternative.

### Weighted Sum Method

**Definition 5.25** (Weighted Sum)
U(a) = Σᵢ wᵢfᵢ(a) where wᵢ ≥ 0 and Σwᵢ = 1

**Theorem 5.9** (Weighted Sum Optimality)
The solution to max U(a) is Pareto optimal.

### TOPSIS Method

**Algorithm 5.1** (TOPSIS)

1. Normalize decision matrix
2. Calculate weighted normalized matrix
3. Identify ideal and anti-ideal solutions
4. Calculate distances to ideal and anti-ideal
5. Calculate relative closeness
6. Rank alternatives by relative closeness

## Prospect Theory

### Value Function

**Definition 5.26** (Value Function)
v(x) = x^α for x ≥ 0, -λ(-x)^β for x < 0

**Properties:**

1. Reference dependence
2. Loss aversion (λ > 1)
3. Diminishing sensitivity (α, β < 1)

### Weighting Function

**Definition 5.27** (Weighting Function)
w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)

**Properties:**

1. Overweighting of small probabilities
2. Underweighting of large probabilities
3. Subadditivity

### Prospect Theory Value

**Definition 5.28** (Prospect Theory Value)
V = Σᵢ w(pᵢ)v(xᵢ)

## Sequential Decision Making

### Markov Decision Processes

**Definition 5.29** (MDP)
A tuple (S, A, P, R, γ) where:

- S: State space
- A: Action space
- P: Transition probabilities
- R: Reward function
- γ: Discount factor

**Definition 5.30** (Policy)
A function π: S → A mapping states to actions.

**Definition 5.31** (Value Function)
V^π(s) = E[Σₜ γ^t R(sₜ, aₜ) | s₀ = s, π]

**Theorem 5.10** (Bellman Equation)
V^π(s) = R(s, π(s)) + γ Σ_s' P(s' | s, π(s)) V^π(s')

### Optimal Policy

**Definition 5.32** (Optimal Value Function)
V*(s) = max_π V^π(s)

**Definition 5.33** (Optimal Policy)
π*(s) = argmax_a [R(s, a) + γ Σ_s' P(s' | s, a) V*(s')]

**Theorem 5.11** (Policy Iteration)

1. Initialize policy π₀
2. Policy evaluation: Compute V^π
3. Policy improvement: π' = argmax_a Q^π(s, a)
4. Repeat until convergence

## Applications

### Economics

- Consumer choice theory
- Investment decisions
- Risk management

### Statistics

- Bayesian inference
- Experimental design
- Model selection

### Artificial Intelligence

- Reinforcement learning
- Multi-agent systems
- Automated decision making

### Operations Research

- Resource allocation
- Scheduling problems
- Supply chain management

## References

1. Berger, J.O. (1985). "Statistical Decision Theory and Bayesian Analysis"
2. von Neumann, J. & Morgenstern, O. (1944). "Theory of Games and Economic Behavior"
3. Savage, L.J. (1954). "The Foundations of Statistics"
4. Kahneman, D. & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk"

## Cross-References

- [Basic Probability Theory](../01_Basic_Probability_Theory.md)
- [Statistics and Data Analysis](../02_Statistics_Data_Analysis.md)
- [Game Theory](../12_Game_Theory.md)
- [Reinforcement Learning](../13_Reinforcement_Learning.md)
