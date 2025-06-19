# Real Analysis and Measure Theory

## Overview

Real analysis and measure theory provide the rigorous foundation for calculus and analysis. They establish the mathematical framework for understanding continuity, differentiation, integration, and convergence in the context of real numbers and more general measure spaces.

## Historical Development

### Early Foundations

- **17th century**: Newton and Leibniz's intuitive calculus
- **19th century**: Cauchy's ε-δ formalism
- **Late 19th century**: Weierstrass's rigorous foundation
- **Early 20th century**: Lebesgue's measure theory

### Modern Development

- **1920s-30s**: Development of abstract measure theory
- **1940s-50s**: Functional analysis and integration theory
- **1960s-70s**: Geometric measure theory
- **1980s-present**: Harmonic analysis and geometric analysis

## Real Number System

### Axioms of Real Numbers

**Axiom 1.1** (Field Axioms)
The real numbers ℝ form a field with operations + and ×.

**Axiom 1.2** (Order Axioms)
ℝ is an ordered field with order relation <.

**Axiom 1.3** (Completeness Axiom)
Every non-empty subset of ℝ that is bounded above has a least upper bound.

**Definition 1.1** (Supremum)
sup A = least upper bound of A

**Definition 1.2** (Infimum)
inf A = greatest lower bound of A

### Sequences and Series

**Definition 1.3** (Convergence)
A sequence {aₙ} converges to L if for every ε > 0, there exists N such that |aₙ - L| < ε for all n ≥ N.

**Definition 1.4** (Cauchy Sequence)
A sequence {aₙ} is Cauchy if for every ε > 0, there exists N such that |aₘ - aₙ| < ε for all m, n ≥ N.

**Theorem 1.1** (Completeness of ℝ)
Every Cauchy sequence in ℝ converges.

**Definition 1.5** (Series)
Σₙ₌₁^∞ aₙ = limₙ→∞ Σₖ₌₁ⁿ aₖ

**Definition 1.6** (Absolute Convergence)
Σₙ₌₁^∞ |aₙ| converges.

**Theorem 1.2** (Absolute Convergence Implies Convergence)
If Σ|aₙ| converges, then Σaₙ converges.

## Continuity and Limits

### Limits of Functions

**Definition 1.7** (Limit)
lim_{x→a} f(x) = L if for every ε > 0, there exists δ > 0 such that |f(x) - L| < ε whenever 0 < |x - a| < δ.

**Definition 1.8** (One-Sided Limits)
lim_{x→a⁺} f(x) = L (right limit)
lim_{x→a⁻} f(x) = L (left limit)

**Theorem 1.3** (Limit Laws)
If lim_{x→a} f(x) = L and lim_{x→a} g(x) = M, then:

1. lim_{x→a} (f(x) + g(x)) = L + M
2. lim_{x→a} (f(x)g(x)) = LM
3. lim_{x→a} (f(x)/g(x)) = L/M (if M ≠ 0)

### Continuity

**Definition 1.9** (Continuity)
A function f is continuous at a if lim_{x→a} f(x) = f(a).

**Definition 1.10** (Uniform Continuity)
A function f is uniformly continuous on A if for every ε > 0, there exists δ > 0 such that |f(x) - f(y)| < ε whenever |x - y| < δ and x, y ∈ A.

**Theorem 1.4** (Intermediate Value Theorem)
If f is continuous on [a,b] and f(a) < c < f(b), then there exists x ∈ (a,b) such that f(x) = c.

**Theorem 1.5** (Extreme Value Theorem)
If f is continuous on [a,b], then f attains its maximum and minimum values on [a,b].

## Differentiation

### Derivative Definition

**Definition 1.11** (Derivative)
f'(a) = lim_{h→0} (f(a+h) - f(a))/h

**Definition 1.12** (Differentiability)
A function f is differentiable at a if f'(a) exists.

**Theorem 1.6** (Differentiability Implies Continuity)
If f is differentiable at a, then f is continuous at a.

### Differentiation Rules

**Theorem 1.7** (Basic Rules)

1. (f + g)'(x) = f'(x) + g'(x)
2. (fg)'(x) = f'(x)g(x) + f(x)g'(x)
3. (f/g)'(x) = (f'(x)g(x) - f(x)g'(x))/g(x)²
4. (f∘g)'(x) = f'(g(x))g'(x) (Chain Rule)

**Theorem 1.8** (Mean Value Theorem)
If f is continuous on [a,b] and differentiable on (a,b), then there exists c ∈ (a,b) such that f'(c) = (f(b) - f(a))/(b - a).

**Theorem 1.9** (Rolle's Theorem)
If f is continuous on [a,b], differentiable on (a,b), and f(a) = f(b), then there exists c ∈ (a,b) such that f'(c) = 0.

### Taylor Series

**Definition 1.13** (Taylor Polynomial)
Pₙ(x) = Σₖ₌₀ⁿ (f⁽ᵏ⁾(a)/k!)(x-a)ᵏ

**Theorem 1.10** (Taylor's Theorem)
f(x) = Pₙ(x) + Rₙ(x) where Rₙ(x) = (f⁽ⁿ⁺¹⁾(c)/(n+1)!)(x-a)ⁿ⁺¹ for some c between a and x.

## Riemann Integration

### Definition and Properties

**Definition 1.14** (Partition)
A partition P of [a,b] is a finite set {x₀, x₁, ..., xₙ} with a = x₀ < x₁ < ... < xₙ = b.

**Definition 1.15** (Upper and Lower Sums)
U(f,P) = Σᵢ₌₁ⁿ MᵢΔxᵢ
L(f,P) = Σᵢ₌₁ⁿ mᵢΔxᵢ

where Mᵢ = sup{f(x) : x ∈ [xᵢ₋₁, xᵢ]} and mᵢ = inf{f(x) : x ∈ [xᵢ₋₁, xᵢ]}.

**Definition 1.16** (Riemann Integrable)
f is Riemann integrable if sup L(f,P) = inf U(f,P).

**Definition 1.17** (Riemann Integral)
∫ₐᵇ f(x) dx = sup L(f,P) = inf U(f,P)

**Theorem 1.11** (Riemann Integrability Criterion)
f is Riemann integrable if and only if for every ε > 0, there exists a partition P such that U(f,P) - L(f,P) < ε.

### Fundamental Theorem of Calculus

**Theorem 1.12** (First Fundamental Theorem)
If f is continuous on [a,b] and F(x) = ∫ₐˣ f(t) dt, then F'(x) = f(x).

**Theorem 1.13** (Second Fundamental Theorem)
If F is differentiable on [a,b] and F' is Riemann integrable, then ∫ₐᵇ F'(x) dx = F(b) - F(a).

## Measure Theory

### σ-Algebras and Measures

**Definition 1.18** (σ-Algebra)
A collection ℱ of subsets of X is a σ-algebra if:

1. ∅ ∈ ℱ
2. If A ∈ ℱ, then Aᶜ ∈ ℱ
3. If {Aₙ} ⊆ ℱ, then ∪ₙ Aₙ ∈ ℱ

**Definition 1.19** (Measure)
A function μ: ℱ → [0,∞] is a measure if:

1. μ(∅) = 0
2. μ(∪ₙ Aₙ) = Σₙ μ(Aₙ) for disjoint sets

**Definition 1.20** (Measure Space)
A triple (X, ℱ, μ) where X is a set, ℱ is a σ-algebra, and μ is a measure.

### Lebesgue Measure

**Definition 1.21** (Outer Measure)
μ*(E) = inf{Σₙ l(Iₙ) : E ⊆ ∪ₙ Iₙ, Iₙ intervals}

**Definition 1.22** (Lebesgue Measurable)
E is Lebesgue measurable if μ*(A) = μ*(A∩E) + μ*(A∩Eᶜ) for all A ⊆ ℝ.

**Definition 1.23** (Lebesgue Measure)
μ(E) = μ*(E) for measurable E.

**Theorem 1.14** (Properties of Lebesgue Measure)

1. All open sets are measurable
2. All closed sets are measurable
3. Countable unions and intersections of measurable sets are measurable
4. Translation invariance: μ(E + x) = μ(E)

### Measurable Functions

**Definition 1.24** (Measurable Function)
f: X → ℝ is measurable if f⁻¹((a,∞)) ∈ ℱ for all a ∈ ℝ.

**Theorem 1.15** (Properties of Measurable Functions)

1. Continuous functions are measurable
2. Pointwise limits of measurable functions are measurable
3. Sums, products, and compositions of measurable functions are measurable

## Lebesgue Integration

### Definition and Properties

**Definition 1.25** (Simple Function)
φ = Σᵢ₌₁ⁿ aᵢχ_{Aᵢ} where Aᵢ ∈ ℱ and aᵢ ∈ ℝ.

**Definition 1.26** (Integral of Simple Function)
∫ φ dμ = Σᵢ₌₁ⁿ aᵢμ(Aᵢ)

**Definition 1.27** (Lebesgue Integral)
∫ f dμ = sup{∫ φ dμ : 0 ≤ φ ≤ f, φ simple}

**Theorem 1.16** (Monotone Convergence Theorem)
If 0 ≤ f₁ ≤ f₂ ≤ ... and fₙ → f pointwise, then ∫ fₙ dμ → ∫ f dμ.

**Theorem 1.17** (Dominated Convergence Theorem)
If |fₙ| ≤ g, g integrable, and fₙ → f pointwise, then ∫ fₙ dμ → ∫ f dμ.

### L^p Spaces

**Definition 1.28** (L^p Norm)
||f||_p = (∫ |f|^p dμ)^(1/p)

**Definition 1.29** (L^p Space)
L^p(X,μ) = {f measurable : ||f||_p < ∞}

**Theorem 1.18** (Hölder's Inequality)
||fg||_1 ≤ ||f||_p ||g||_q where 1/p + 1/q = 1.

**Theorem 1.19** (Minkowski's Inequality)
||f + g||_p ≤ ||f||_p + ||g||_p

## Function Spaces

### Continuous Functions

**Definition 1.30** (C(X))
Space of continuous functions on X.

**Definition 1.31** (C₀(X))
Space of continuous functions vanishing at infinity.

**Theorem 1.20** (Stone-Weierstrass Theorem)
If A is a subalgebra of C(X) that separates points and contains constants, then A is dense in C(X).

### Sobolev Spaces

**Definition 1.32** (Weak Derivative)
g is the weak derivative of f if ∫ fφ' dx = -∫ gφ dx for all φ ∈ C₀^∞.

**Definition 1.33** (Sobolev Space)
W^{k,p}(Ω) = {f ∈ L^p(Ω) : D^αf ∈ L^p(Ω) for |α| ≤ k}

**Theorem 1.21** (Sobolev Embedding)
W^{k,p}(ℝⁿ) ⊆ C^{k-[n/p]-1}(ℝⁿ) if k > n/p.

### Hölder Spaces

**Definition 1.34** (Hölder Continuity)
|f(x) - f(y)| ≤ C|x - y|^α for some C, α > 0.

**Definition 1.35** (Hölder Space)
C^{k,α}(Ω) = {f ∈ C^k(Ω) : D^αf is Hölder continuous}

## Convergence Concepts

### Pointwise and Uniform Convergence

**Definition 1.36** (Pointwise Convergence)
fₙ → f pointwise if fₙ(x) → f(x) for each x.

**Definition 1.37** (Uniform Convergence)
fₙ → f uniformly if sup_x |fₙ(x) - f(x)| → 0.

**Theorem 1.22** (Uniform Convergence Preserves Continuity)
If fₙ are continuous and fₙ → f uniformly, then f is continuous.

### Convergence in Measure

**Definition 1.38** (Convergence in Measure)
fₙ → f in measure if μ({x : |fₙ(x) - f(x)| > ε}) → 0 for all ε > 0.

**Theorem 1.23** (Convergence in L^p)
If fₙ → f in L^p, then fₙ → f in measure.

### Almost Everywhere Convergence

**Definition 1.39** (Almost Everywhere)
A property holds almost everywhere (a.e.) if it holds except on a set of measure zero.

**Definition 1.40** (Almost Everywhere Convergence)
fₙ → f a.e. if fₙ(x) → f(x) for almost every x.

**Theorem 1.24** (Egorov's Theorem)
If fₙ → f a.e. on a set of finite measure, then fₙ → f almost uniformly.

## Applications

### Fourier Analysis

- L^p theory of Fourier transforms
- Convergence of Fourier series
- Harmonic analysis

### Partial Differential Equations

- Sobolev spaces and regularity
- Weak solutions
- Energy methods

### Probability Theory

- Measure-theoretic foundations
- Random variables and expectations
- Convergence theorems

### Functional Analysis

- Banach and Hilbert spaces
- Operator theory
- Spectral theory

## References

1. Rudin, W. (1976). "Principles of Mathematical Analysis"
2. Folland, G.B. (1999). "Real Analysis: Modern Techniques and Their Applications"
3. Royden, H.L. (1988). "Real Analysis"
4. Wheeden, R.L. & Zygmund, A. (1977). "Measure and Integral"

## Cross-References

- [Complex Analysis and Function Theory](../02_Complex_Analysis_Function_Theory.md)
- [Functional Analysis and Operator Theory](../03_Functional_Analysis_Operator_Theory.md)
- [Harmonic Analysis and Fourier Theory](../04_Harmonic_Analysis_Fourier_Theory.md)
- [Differential Equations and Dynamical Systems](../05_Differential_Equations_Dynamical_Systems.md)
