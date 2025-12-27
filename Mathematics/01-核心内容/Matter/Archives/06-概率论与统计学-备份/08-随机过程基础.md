# Stochastic Processes

## Overview

Stochastic processes provide the mathematical framework for modeling systems that evolve randomly over time. They extend probability theory to dynamic systems and form the foundation for understanding complex phenomena in physics, finance, biology, and engineering.

## Historical Development

### Early Foundations

- **1900**: Bachelier's thesis on Brownian motion in finance
- **1905**: Einstein's explanation of Brownian motion
- **1906**: Markov's introduction of Markov chains
- **1923**: Wiener's rigorous construction of Brownian motion

### Modern Development

- **1930s-40s**: Kolmogorov's work on Markov processes
- **1950s-60s**: Ito's stochastic calculus
- **1970s-80s**: Development of martingale theory
- **1990s-present**: Applications in finance and biology

## Basic Definitions

### Stochastic Process

**Definition 3.1** (Stochastic Process)
A stochastic process is a collection of random variables {X_t : t ∈ T} indexed by a parameter set T.

**Definition 3.2** (State Space)
The set S of possible values for X_t is called the state space.

**Definition 3.3** (Index Set)
The set T is called the index set (time parameter).

**Example 3.1** (Types of Processes)

- **Discrete time**: T = {0,1,2,...}
- **Continuous time**: T = [0,∞)
- **Discrete state**: S = {0,1,2,...}
- **Continuous state**: S = ℝ

### Finite-Dimensional Distributions

**Definition 3.4** (Finite-Dimensional Distribution)
For t₁ < t₂ < ... < tₙ, the joint distribution of (X_{t₁}, X_{t₂}, ..., X_{tₙ}).

**Definition 3.5** (Consistency)
A family of finite-dimensional distributions is consistent if they satisfy Kolmogorov's consistency conditions.

**Theorem 3.1** (Kolmogorov Extension Theorem)
Given a consistent family of finite-dimensional distributions, there exists a stochastic process with these distributions.

## Markov Processes

### Markov Property

**Definition 3.6** (Markov Property)
A process {X_t} has the Markov property if:
P(X_{t+s} ∈ A | X_u, u ≤ t) = P(X_{t+s} ∈ A | X_t)

**Definition 3.7** (Markov Chain)
A discrete-time Markov process with countable state space.

### Transition Probabilities

**Definition 3.8** (Transition Probability)
P_{ij}(n) = P(X_{n+1} = j | X_n = i)

**Definition 3.9** (Transition Matrix)
P(n) = [P_{ij}(n)]_{i,j∈S}

**Theorem 3.2** (Chapman-Kolmogorov Equation)
P_{ij}(m+n) = Σ_k P_{ik}(m)P_{kj}(n)

### Classification of States

**Definition 3.10** (Accessible State)
State j is accessible from state i if P_{ij}(n) > 0 for some n ≥ 0.

**Definition 3.11** (Communicating States)
States i and j communicate if i is accessible from j and j is accessible from i.

**Definition 3.12** (Irreducible Chain)
A Markov chain is irreducible if all states communicate.

**Definition 3.13** (Recurrent State)
State i is recurrent if P(X_n = i for infinitely many n | X_0 = i) = 1.

**Definition 3.14** (Transient State)
State i is transient if P(X_n = i for infinitely many n | X_0 = i) = 0.

### Stationary Distribution

**Definition 3.15** (Stationary Distribution)
A probability distribution π is stationary if πP = π.

**Theorem 3.3** (Existence of Stationary Distribution)
For an irreducible, positive recurrent Markov chain, there exists a unique stationary distribution.

**Theorem 3.4** (Convergence to Stationary Distribution)
For an irreducible, aperiodic, positive recurrent chain:
lim_{n→∞} P_{ij}(n) = π_j

## Continuous-Time Markov Chains

### Definition and Properties

**Definition 3.16** (Continuous-Time Markov Chain)
A continuous-time process {X_t} with countable state space satisfying the Markov property.

**Definition 3.17** (Transition Probability)
P_{ij}(t) = P(X_{t+s} = j | X_s = i)

**Definition 3.18** (Transition Rate Matrix)
Q = [q_{ij}] where q_{ij} = lim_{t→0} (P_{ij}(t) - δ_{ij})/t

**Theorem 3.5** (Kolmogorov Forward Equation)
dP(t)/dt = P(t)Q

**Theorem 3.6** (Kolmogorov Backward Equation)
dP(t)/dt = QP(t)

### Birth-Death Processes

**Definition 3.19** (Birth-Death Process)
A continuous-time Markov chain with state space {0,1,2,...} and transitions only to neighboring states.

**Definition 3.20** (Birth Rate)
λ_i: Rate of transition from state i to state i+1.

**Definition 3.21** (Death Rate)
μ_i: Rate of transition from state i to state i-1.

**Example 3.2** (M/M/1 Queue)

- Birth rate: λ_i = λ (constant)
- Death rate: μ_i = μ for i ≥ 1, μ_0 = 0

## Poisson Processes

### Definition and Properties

**Definition 3.22** (Poisson Process)
A counting process {N_t} with:

1. N_0 = 0
2. Independent increments
3. Stationary increments
4. P(N_{t+h} - N_t = 1) = λh + o(h)
5. P(N_{t+h} - N_t ≥ 2) = o(h)

**Theorem 3.7** (Poisson Distribution)
For a Poisson process with rate λ:
P(N_t = k) = e^(-λt)(λt)^k / k!

**Theorem 3.8** (Interarrival Times)
The interarrival times T₁, T₂, ... are i.i.d. exponential with rate λ.

### Compound Poisson Process

**Definition 3.23** (Compound Poisson Process)
X_t = Σ_{i=1}^{N_t} Y_i, where {N_t} is a Poisson process and {Y_i} are i.i.d. random variables.

**Theorem 3.9** (Characteristic Function)
φ_{X_t}(u) = exp(λt(φ_Y(u) - 1))

## Brownian Motion

### Definition and Properties

**Definition 3.24** (Standard Brownian Motion)
A continuous-time process {B_t} with:

1. B_0 = 0
2. Independent increments
3. Stationary increments
4. B_t - B_s ~ N(0, t-s) for s < t
5. Continuous sample paths

**Theorem 3.10** (Properties of Brownian Motion)

1. E[B_t] = 0
2. Var(B_t) = t
3. Cov(B_s, B_t) = min(s,t)
4. Sample paths are nowhere differentiable

### Geometric Brownian Motion

**Definition 3.25** (Geometric Brownian Motion)
S_t = S_0 exp((μ - σ²/2)t + σB_t)

**Theorem 3.11** (Properties of GBM)

1. E[S_t] = S_0 e^(μt)
2. Var(S_t) = S_0² e^(2μt)(e^(σ²t) - 1)

### Ito's Lemma

**Theorem 3.12** (Ito's Lemma)
For a function f(t,B_t):
df = (∂f/∂t + (1/2)∂²f/∂x²)dt + (∂f/∂x)dB_t

**Example 3.3** (Ito's Lemma Application)
For f(t,x) = x²:
d(B_t²) = dt + 2B_t dB_t

## Martingales

### Definition and Properties

**Definition 3.26** (Martingale)
A process {M_t} is a martingale if:

1. E[|M_t|] < ∞ for all t
2. E[M_t | F_s] = M_s for s ≤ t

**Definition 3.27** (Submartingale)
E[M_t | F_s] ≥ M_s for s ≤ t

**Definition 3.28** (Supermartingale)
E[M_t | F_s] ≤ M_s for s ≤ t

**Theorem 3.13** (Optional Stopping Theorem)
For a martingale {M_t} and stopping time τ:
E[M_τ] = E[M_0] if τ is bounded

### Examples of Martingales

**Example 3.4** (Random Walk)
S_n = Σ_{i=1}^n X_i, where {X_i} are i.i.d. with E[X_i] = 0

**Example 3.5** (Exponential Martingale)
M_t = exp(σB_t - σ²t/2)

## Diffusion Processes

### Stochastic Differential Equations

**Definition 3.29** (SDE)
dX_t = μ(t,X_t)dt + σ(t,X_t)dB_t

**Definition 3.30** (Drift)
μ(t,x): Deterministic component

**Definition 3.31** (Diffusion Coefficient)
σ(t,x): Stochastic component

### Fokker-Planck Equation

**Theorem 3.14** (Fokker-Planck Equation)
For a diffusion process with density p(t,x):
∂p/∂t = -∂(μp)/∂x + (1/2)∂²(σ²p)/∂x²

### Ornstein-Uhlenbeck Process

**Definition 3.32** (Ornstein-Uhlenbeck Process)
dX_t = -αX_t dt + σ dB_t

**Theorem 3.15** (OU Process Properties)

1. E[X_t] = X_0 e^(-αt)
2. Var(X_t) = (σ²/(2α))(1 - e^(-2αt))
3. Stationary distribution: N(0, σ²/(2α))

## Renewal Processes

### Definition and Properties

**Definition 3.33** (Renewal Process)
A counting process {N_t} where interarrival times {T_i} are i.i.d. positive random variables.

**Definition 3.34** (Renewal Function)
m(t) = E[N_t]

**Theorem 3.16** (Elementary Renewal Theorem)
lim_{t→∞} m(t)/t = 1/μ, where μ = E[T₁]

**Theorem 3.17** (Key Renewal Theorem)
For a directly Riemann integrable function h:
lim_{t→∞} ∫_0^t h(t-s) dm(s) = (1/μ) ∫_0^∞ h(s) ds

## Applications

### Finance

- Option pricing (Black-Scholes model)
- Risk management
- Portfolio optimization

### Physics

- Diffusion processes
- Quantum mechanics
- Statistical mechanics

### Biology

- Population dynamics
- Gene expression
- Neural networks

### Engineering

- Queueing theory
- Reliability theory
- Control systems

## References

1. Karlin, S. & Taylor, H.M. (1975). "A First Course in Stochastic Processes"
2. Ross, S.M. (1996). "Stochastic Processes"
3. Durrett, R. (2019). "Probability: Theory and Examples"
4. Øksendal, B. (2003). "Stochastic Differential Equations"

## Cross-References

- [Basic Probability Theory](../01_Basic_Probability_Theory.md)
- [Statistics and Data Analysis](../02_Statistics_Data_Analysis.md)
- [Financial Mathematics](../08_Financial_Mathematics.md)
- [Queueing Theory](../09_Queueing_Theory.md)
