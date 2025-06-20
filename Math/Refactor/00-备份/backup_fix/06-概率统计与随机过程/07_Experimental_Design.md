# Experimental Design

## Overview

Experimental design provides the mathematical framework for planning and conducting experiments efficiently. It encompasses principles for controlling variability, optimizing information gathering, and ensuring valid statistical inference in scientific research and industrial applications.

## Historical Development

### Early Foundations

- **1920s**: Fisher's work on agricultural experiments
- **1930s**: Development of factorial designs
- **1940s**: Response surface methodology
- **1950s**: Optimal design theory

### Modern Development

- **1960s**: Computer-aided design
- **1970s**: Robust design principles
- **1980s**: Taguchi methods
- **1990s-present**: Adaptive and sequential designs

## Basic Principles

### Experimental Units and Treatments

**Definition 7.1** (Experimental Unit)
The smallest unit to which a treatment can be applied.

**Definition 7.2** (Treatment)
A specific combination of factor levels applied to experimental units.

**Definition 7.3** (Factor)
A variable that may influence the response.

**Definition 7.4** (Level)
A specific value or setting of a factor.

### Design Principles

**Principle 7.1** (Randomization)
Random assignment of treatments to experimental units.

**Principle 7.2** (Replication)
Multiple observations under the same treatment conditions.

**Principle 7.3** (Blocking)
Grouping experimental units to control known sources of variation.

**Principle 7.4** (Factorial Structure)
Systematic investigation of factor combinations.

## Completely Randomized Design

### Model and Analysis

**Definition 7.5** (CRD Model)
Yᵢⱼ = μ + τᵢ + εᵢⱼ

where:

- Yᵢⱼ: response for treatment i, replication j
- μ: overall mean
- τᵢ: treatment effect
- εᵢⱼ: random error

**Assumption 7.1** (Error Assumptions)
εᵢⱼ ~ N(0, σ²) i.i.d.

**Theorem 7.1** (ANOVA Decomposition)
SST = SSTr + SSE

where:

- SST = Σᵢⱼ (Yᵢⱼ - Ȳ..)²
- SSTr = Σᵢ nᵢ(Ȳᵢ. - Ȳ..)²
- SSE = Σᵢⱼ (Yᵢⱼ - Ȳᵢ.)²

**Theorem 7.2** (F-Test)
F = MSTr/MSE ~ F_{t-1, N-t} under H₀: τ₁ = ... = τₜ = 0

## Randomized Block Design

### Model and Analysis

**Definition 7.6** (RBD Model)
Yᵢⱼ = μ + τᵢ + βⱼ + εᵢⱼ

where:

- βⱼ: block effect
- Other terms as in CRD

**Theorem 7.3** (RBD ANOVA)
SST = SSTr + SSB + SSE

where:

- SSB = Σⱼ t(Ȳ.ⱼ - Ȳ..)²

**Theorem 7.4** (RBD F-Test)
F = MSTr/MSE ~ F_{t-1, (t-1)(b-1)} under H₀: τ₁ = ... = τₜ = 0

### Efficiency

**Definition 7.7** (Relative Efficiency)
RE = MSE_RBD / MSE_CRD

**Theorem 7.5** (Efficiency Formula)
RE = (b-1)(t-1) / (bt-1) × (1 + bσ²_β/σ²)

## Latin Square Design

### Design Structure

**Definition 7.8** (Latin Square)
A t × t array where each treatment appears exactly once in each row and column.

**Definition 7.9** (LSD Model)
Yᵢⱼₖ = μ + τᵢ + ρⱼ + γₖ + εᵢⱼₖ

where:

- ρⱼ: row effect
- γₖ: column effect

**Theorem 7.6** (LSD ANOVA)
SST = SSTr + SSR + SSC + SSE

**Theorem 7.7** (LSD F-Test)
F = MSTr/MSE ~ F_{t-1, (t-1)(t-2)} under H₀

## Factorial Designs

### Two-Factor Design

**Definition 7.10** (Two-Factor Model)
Yᵢⱼₖ = μ + αᵢ + βⱼ + (αβ)ᵢⱼ + εᵢⱼₖ

where:

- αᵢ: factor A effect
- βⱼ: factor B effect
- (αβ)ᵢⱼ: interaction effect

**Theorem 7.8** (Two-Factor ANOVA)
SST = SSA + SSB + SSAB + SSE

**Definition 7.11** (Interaction)
Factors A and B interact if the effect of A depends on the level of B.

### Higher-Order Factorials

**Definition 7.12** (Three-Factor Model)
Yᵢⱼₖₗ = μ + αᵢ + βⱼ + γₖ + (αβ)ᵢⱼ + (αγ)ᵢₖ + (βγ)ⱼₖ + (αβγ)ᵢⱼₖ + εᵢⱼₖₗ

**Theorem 7.9** (Three-Factor ANOVA)
SST = SSA + SSB + SSC + SSAB + SSAC + SSBC + SSABC + SSE

## Fractional Factorial Designs

### Confounding

**Definition 7.13** (Confounding)
Two effects are confounded if they cannot be estimated separately.

**Definition 7.14** (Defining Relation)
I = ABC for a 2³⁻¹ design means A = BC.

**Theorem 7.10** (Confounding Pattern)
In a 2ᵏ⁻ᵖ design, effects are confounded in groups of 2ᵖ.

### Resolution

**Definition 7.15** (Resolution)
The resolution of a fractional factorial is the length of the shortest word in the defining relation.

**Example 7.1** (Resolution III)
2³⁻¹ with I = ABC: main effects confounded with two-factor interactions.

**Example 7.2** (Resolution IV)
2⁴⁻¹ with I = ABCD: main effects not confounded with two-factor interactions.

## Response Surface Methodology

### First-Order Model

**Definition 7.16** (First-Order Model)
Y = β₀ + Σᵢ₌₁ᵏ βᵢxᵢ + ε

**Definition 7.17** (Steepest Ascent)
Direction of maximum increase: ∇f = (β₁, ..., βₖ)

**Algorithm 7.1** (Steepest Ascent)

1. Fit first-order model
2. Move in direction of steepest ascent
3. Continue until lack of fit detected

### Second-Order Model

**Definition 7.18** (Second-Order Model)
Y = β₀ + Σᵢ₌₁ᵏ βᵢxᵢ + Σᵢ₌₁ᵏ βᵢᵢxᵢ² + Σᵢ<ⱼ βᵢⱼxᵢxⱼ + ε

**Theorem 7.11** (Stationary Point)
xₛ = -(1/2)B⁻¹b where B is the matrix of second-order coefficients.

**Definition 7.19** (Canonical Form)
Y = Yₛ + Σᵢ₌₁ᵏ λᵢwᵢ²

where λᵢ are eigenvalues of B.

## Optimal Design Theory

### Design Criteria

**Definition 7.20** (D-Optimality)
Maximize |X'X| where X is the design matrix.

**Definition 7.21** (A-Optimality)
Minimize tr((X'X)⁻¹).

**Definition 7.22** (G-Optimality)
Minimize max_x d(x) where d(x) is the prediction variance.

**Theorem 7.12** (Equivalence Theorem)
For linear models, D-optimal and G-optimal designs are equivalent.

### Algorithmic Construction

**Algorithm 7.2** (Exchange Algorithm)

1. Start with initial design
2. For each point in design:
   - Try replacing with candidate point
   - Accept if criterion improves
3. Repeat until convergence

**Algorithm 7.3** (Coordinate Exchange)

1. Start with factorial design
2. For each coordinate:
   - Optimize coordinate values
   - Update design matrix
3. Repeat until convergence

## Robust Design

### Taguchi Methods

**Definition 7.23** (Control Factors)
Factors that can be controlled in production.

**Definition 7.24** (Noise Factors)
Factors that vary in use but cannot be controlled.

**Definition 7.25** (Signal-to-Noise Ratio)
SNR = 10 log₁₀(μ²/σ²) for larger-the-better.

**Theorem 7.13** (Two-Step Optimization)

1. Maximize SNR to reduce variability
2. Adjust mean to target using control factors

### Response Modeling

**Definition 7.26** (Dual Response Model)
μ = f₁(x_c, x_n)
σ² = f₂(x_c, x_n)

**Definition 7.27** (Robust Optimization)
min σ² subject to μ = target

## Sequential Design

### Adaptive Design

**Definition 7.28** (Adaptive Design)
Design that uses information from previous runs to guide future experiments.

**Algorithm 7.4** (Sequential Design)

1. Start with initial design
2. Fit model to current data
3. Choose next design point to maximize information
4. Conduct experiment
5. Repeat until stopping criterion met

### Bayesian Design

**Definition 7.29** (Bayesian Design Criterion)
U(ξ) = E[U(ξ, θ)] where expectation is over prior π(θ).

**Definition 7.30** (Expected Information Gain)
U(ξ) = E[I(θ; y)] where I is mutual information.

**Theorem 7.14** (Bayesian D-Optimality)
For normal linear models, Bayesian D-optimal design maximizes E[log|X'X + V₀⁻¹|].

## Computer Experiments

### Gaussian Process Models

**Definition 7.31** (Gaussian Process)
Y(x) ~ GP(μ(x), k(x,x'))

**Definition 7.32** (Kriging Predictor)
Ŷ(x) = μ + k'(x)K⁻¹(y - 1μ)

**Definition 7.33** (Mean Squared Prediction Error)
MSPE(x) = k(x,x) - k'(x)K⁻¹k(x)

### Space-Filling Designs

**Definition 7.34** (Latin Hypercube)
Design where each factor has exactly one observation in each of n intervals.

**Definition 7.35** (Maximin Distance)
maximize minᵢ<ⱼ d(xᵢ, xⱼ)

**Definition 7.36** (Minimax Distance)
minimize max_x minᵢ d(x, xᵢ)

## Applications

### Industrial Applications

- Process optimization
- Quality improvement
- Product design

### Scientific Research

- Agricultural experiments
- Clinical trials
- Engineering studies

### Computer Experiments

- Simulation optimization
- Model calibration
- Sensitivity analysis

## References

1. Montgomery, D.C. (2017). "Design and Analysis of Experiments"
2. Box, G.E.P. et al. (2005). "Statistics for Experimenters"
3. Myers, R.H. et al. (2016). "Response Surface Methodology"
4. Santner, T.J. et al. (2018). "The Design and Analysis of Computer Experiments"

## Cross-References

- [Statistics and Data Analysis](../02_Statistics_Data_Analysis.md)
- [Optimization Theory](../14_Optimization_Theory.md)
- [Quality Control](../16_Quality_Control.md)
- [Simulation Methods](../17_Simulation_Methods.md)
