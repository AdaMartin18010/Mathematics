# Statistics and Data Analysis

## Overview

Statistics and data analysis provide the mathematical framework for extracting meaningful information from data, making inferences about populations, and supporting decision-making under uncertainty. This field bridges probability theory with practical applications in science, engineering, and social sciences.

## Historical Development

### Early Statistics

- **17th century**: Graunt's demographic studies
- **18th century**: Laplace's statistical methods
- **19th century**: Gauss's method of least squares
- **Early 20th century**: Fisher's contributions to experimental design

### Modern Era

- **Mid-20th century**: Development of computational statistics
- **Late 20th century**: Emergence of data science
- **21st century**: Big data and machine learning integration

## Descriptive Statistics

### Measures of Central Tendency

**Definition 2.1** (Sample Mean)
For data x₁, x₂, ..., xₙ:
x̄ = (1/n)Σᵢ₌₁ⁿ xᵢ

**Definition 2.2** (Sample Median)
The middle value when data is ordered, or average of two middle values.

**Definition 2.3** (Sample Mode)
The most frequently occurring value in the dataset.

### Measures of Dispersion

**Definition 2.4** (Sample Variance)
s² = (1/(n-1))Σᵢ₌₁ⁿ (xᵢ - x̄)²

**Definition 2.5** (Sample Standard Deviation)
s = √s²

**Definition 2.6** (Range)
R = max(xᵢ) - min(xᵢ)

**Definition 2.7** (Interquartile Range)
IQR = Q₃ - Q₁, where Q₁ and Q₃ are first and third quartiles

### Shape Measures

**Definition 2.8** (Skewness)
γ₁ = (1/n)Σᵢ₌₁ⁿ ((xᵢ - x̄)/s)³

**Definition 2.9** (Kurtosis)
γ₂ = (1/n)Σᵢ₌₁ⁿ ((xᵢ - x̄)/s)⁴ - 3

## Statistical Inference

### Point Estimation

**Definition 2.10** (Estimator)
A statistic used to estimate a population parameter.

**Definition 2.11** (Unbiased Estimator)
An estimator θ̂ is unbiased if E[θ̂] = θ.

**Definition 2.12** (Consistent Estimator)
An estimator θ̂ is consistent if θ̂ → θ in probability.

**Definition 2.13** (Efficient Estimator)
An estimator with minimum variance among all unbiased estimators.

### Maximum Likelihood Estimation

**Definition 2.14** (Likelihood Function)
L(θ|x) = f(x|θ) for continuous data or P(x|θ) for discrete data.

**Definition 2.15** (Maximum Likelihood Estimator)
θ̂_MLE = argmax L(θ|x)

**Theorem 2.1** (Invariance Property)
If θ̂ is the MLE of θ, then g(θ̂) is the MLE of g(θ).

### Method of Moments

**Definition 2.16** (Method of Moments Estimator)
Equate sample moments to population moments and solve for parameters.

**Example 2.1** (Normal Distribution)
For X ~ N(μ,σ²):

- μ̂ = x̄
- σ̂² = (1/n)Σᵢ₌₁ⁿ (xᵢ - x̄)²

## Interval Estimation

### Confidence Intervals

**Definition 2.17** (Confidence Interval)
A random interval [L(X), U(X)] such that P(θ ∈ [L(X), U(X)]) = 1-α.

**Theorem 2.2** (Normal Mean CI)
For X ~ N(μ,σ²) with known σ²:
[x̄ - z_{α/2}σ/√n, x̄ + z_{α/2}σ/√n]

**Theorem 2.3** (Normal Mean CI - Unknown Variance)
For X ~ N(μ,σ²) with unknown σ²:
[x̄ - t_{α/2,n-1}s/√n, x̄ + t_{α/2,n-1}s/√n]

### Bootstrap Methods

**Definition 2.18** (Bootstrap Sample)
A random sample with replacement from the original data.

**Algorithm 2.1** (Bootstrap Confidence Interval)

1. Generate B bootstrap samples
2. Calculate θ̂* for each bootstrap sample
3. Use percentiles of θ̂* to construct CI

## Hypothesis Testing

### Framework

**Definition 2.19** (Null Hypothesis)
H₀: θ ∈ Θ₀

**Definition 2.20** (Alternative Hypothesis)
H₁: θ ∈ Θ₁

**Definition 2.21** (Test Statistic)
A function T(X) used to make the decision.

**Definition 2.22** (Rejection Region)
R = {x: T(x) ∈ C}, where C is the critical region.

### Error Types

**Definition 2.23** (Type I Error)
Rejecting H₀ when it is true: P(Type I) = α

**Definition 2.24** (Type II Error)
Failing to reject H₀ when H₁ is true: P(Type II) = β

**Definition 2.25** (Power)
Power = 1 - β = P(Reject H₀|H₁ is true)

### Common Tests

**Theorem 2.4** (Z-Test for Mean)
For H₀: μ = μ₀ vs H₁: μ ≠ μ₀ with known σ²:
Z = (x̄ - μ₀)/(σ/√n) ~ N(0,1) under H₀

**Theorem 2.5** (T-Test for Mean)
For H₀: μ = μ₀ vs H₁: μ ≠ μ₀ with unknown σ²:
T = (x̄ - μ₀)/(s/√n) ~ t_{n-1} under H₀

**Theorem 2.6** (Chi-Square Test for Variance)
For H₀: σ² = σ₀² vs H₁: σ² ≠ σ₀²:
χ² = (n-1)s²/σ₀² ~ χ²_{n-1} under H₀

**Theorem 2.7** (F-Test for Two Variances)
For H₀: σ₁² = σ₂² vs H₁: σ₁² ≠ σ₂²:
F = s₁²/s₂² ~ F_{n₁-1,n₂-1} under H₀

## Regression Analysis

### Linear Regression

**Definition 2.26** (Simple Linear Regression Model)
Y = β₀ + β₁X + ε, where ε ~ N(0,σ²)

**Definition 2.27** (Least Squares Estimators)
β̂₀ = ȳ - β̂₁x̄
β̂₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²

**Theorem 2.8** (Gauss-Markov Theorem)
OLS estimators are BLUE (Best Linear Unbiased Estimators).

### Multiple Regression

**Definition 2.28** (Multiple Linear Regression Model)
Y = β₀ + β₁X₁ + ... + βₚXₚ + ε

**Definition 2.29** (Matrix Form)
Y = Xβ + ε, where Y ∈ ℝⁿ, X ∈ ℝⁿˣᵖ, β ∈ ℝᵖ

**Theorem 2.9** (OLS Solution)
β̂ = (X'X)⁻¹X'Y

### Model Diagnostics

**Definition 2.30** (Residuals)
eᵢ = yᵢ - ŷᵢ

**Definition 2.31** (R-Squared)
R² = 1 - SSE/SST = SSR/SST

**Definition 2.32** (Adjusted R-Squared)
R²_adj = 1 - (1-R²)(n-1)/(n-p-1)

## Analysis of Variance (ANOVA)

### One-Way ANOVA

**Definition 2.33** (One-Way ANOVA Model)
Yᵢⱼ = μ + αᵢ + εᵢⱼ, where Σαᵢ = 0

**Theorem 2.10** (F-Statistic)
F = MSB/MSE ~ F_{k-1,n-k} under H₀: α₁ = ... = αₖ = 0

### Two-Way ANOVA

**Definition 2.34** (Two-Way ANOVA Model)
Yᵢⱼₖ = μ + αᵢ + βⱼ + (αβ)ᵢⱼ + εᵢⱼₖ

## Nonparametric Methods

### Rank-Based Tests

**Definition 2.35** (Wilcoxon Rank-Sum Test)
Nonparametric alternative to two-sample t-test.

**Definition 2.36** (Kruskal-Wallis Test)
Nonparametric alternative to one-way ANOVA.

### Goodness-of-Fit Tests

**Definition 2.37** (Chi-Square Goodness-of-Fit)
Test whether data follows a specified distribution.

**Definition 2.38** (Kolmogorov-Smirnov Test)
Test based on maximum difference between empirical and theoretical CDFs.

## Bayesian Statistics

### Bayesian Framework

**Definition 2.39** (Prior Distribution)
π(θ): Prior belief about parameter θ.

**Definition 2.40** (Posterior Distribution)
π(θ|x) ∝ f(x|θ)π(θ)

**Theorem 2.11** (Bayes' Theorem for Parameters)
π(θ|x) = f(x|θ)π(θ) / ∫ f(x|θ)π(θ) dθ

### Conjugate Priors

**Definition 2.41** (Conjugate Prior)
A prior that results in a posterior from the same family.

**Example 2.2** (Normal-Normal Conjugate)

- Prior: μ ~ N(μ₀, σ₀²)
- Likelihood: X|μ ~ N(μ, σ²)
- Posterior: μ|X ~ N(μ₁, σ₁²)

## Time Series Analysis

### Stationarity

**Definition 2.42** (Weak Stationarity)
A process is weakly stationary if:

1. E[X_t] = μ (constant)
2. Var(X_t) = σ² (constant)
3. Cov(X_t, X_{t+h}) = γ(h) (depends only on lag h)

### ARIMA Models

**Definition 2.43** (AR(p) Model)
X_t = φ₁X_{t-1} + ... + φₚX_{t-p} + ε_t

**Definition 2.44** (MA(q) Model)
X_t = ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}

**Definition 2.45** (ARIMA(p,d,q) Model)
(1-φ₁B-...-φₚB^p)(1-B)^d X_t = (1+θ₁B+...+θ_qB^q)ε_t

## Modern Data Analysis

### Machine Learning Integration

**Definition 2.46** (Supervised Learning)
Learning from labeled training data.

**Definition 2.47** (Unsupervised Learning)
Learning patterns in unlabeled data.

### Big Data Challenges

**Definition 2.48** (Scalability)
Ability to handle large datasets efficiently.

**Definition 2.49** (Dimensionality)
Curse of dimensionality in high-dimensional spaces.

## Applications

### Scientific Research

- Experimental design
- Data analysis
- Hypothesis testing

### Business and Economics

- Market research
- Quality control
- Financial modeling

### Social Sciences

- Survey analysis
- Demographic studies
- Policy evaluation

## References

1. Casella, G. & Berger, R.L. (2002). "Statistical Inference"
2. Rice, J.A. (2007). "Mathematical Statistics and Data Analysis"
3. Wasserman, L. (2004). "All of Statistics"
4. Gelman, A. et al. (2013). "Bayesian Data Analysis"

## Cross-References

- [Basic Probability Theory](../01_Basic_Probability_Theory.md)
- [Stochastic Processes](../03_Stochastic_Processes.md)
- [Machine Learning Theory](../06_Machine_Learning_Theory.md)
- [Experimental Design](../07_Experimental_Design.md)
