# Information Theory

## Overview

Information theory provides the mathematical foundation for quantifying information, understanding communication systems, and analyzing data compression and transmission. It establishes fundamental limits on data processing and communication, with applications spanning computer science, statistics, physics, and engineering.

## Historical Development

### Foundation

- **1948**: Shannon's "A Mathematical Theory of Communication"
- **1949**: Shannon's work on cryptography
- **1950s**: Development of coding theory
- **1960s**: Rate-distortion theory

### Modern Development

- **1970s-80s**: Algorithmic information theory
- **1990s**: Network information theory
- **2000s**: Quantum information theory
- **2010s-present**: Machine learning applications

## Entropy and Information

### Shannon Entropy

**Definition 4.1** (Shannon Entropy)
For a discrete random variable X with probability mass function p(x):
H(X) = -Σ_x p(x) log p(x)

**Properties:**

1. H(X) ≥ 0
2. H(X) = 0 if and only if X is deterministic
3. H(X) ≤ log |X| (maximum when uniform)

**Theorem 4.1** (Entropy Bounds)
0 ≤ H(X) ≤ log |X|

**Example 4.1** (Binary Entropy)
For X ~ Bernoulli(p):
H(X) = -p log p - (1-p) log(1-p) = H(p)

### Joint and Conditional Entropy

**Definition 4.2** (Joint Entropy)
H(X,Y) = -Σ_{x,y} p(x,y) log p(x,y)

**Definition 4.3** (Conditional Entropy)
H(Y|X) = -Σ_{x,y} p(x,y) log p(y|x)

**Theorem 4.2** (Chain Rule)
H(X,Y) = H(X) + H(Y|X)

**Theorem 4.3** (Conditioning Reduces Entropy)
H(Y|X) ≤ H(Y)

### Relative Entropy and Mutual Information

**Definition 4.4** (Relative Entropy/Kullback-Leibler Divergence)
D(p||q) = Σ_x p(x) log(p(x)/q(x))

**Properties:**

1. D(p||q) ≥ 0
2. D(p||q) = 0 if and only if p = q
3. Not symmetric: D(p||q) ≠ D(q||p)

**Definition 4.5** (Mutual Information)
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)

**Properties:**

1. I(X;Y) ≥ 0
2. I(X;Y) = 0 if and only if X and Y are independent
3. I(X;Y) = I(Y;X)

**Theorem 4.4** (Data Processing Inequality)
If X → Y → Z forms a Markov chain, then I(X;Y) ≥ I(X;Z)

## Source Coding

### Kraft Inequality

**Definition 4.6** (Prefix Code)
A code where no codeword is a prefix of another.

**Theorem 4.5** (Kraft Inequality)
For a prefix code with codeword lengths l₁, l₂, ..., lₙ:
Σᵢ 2^(-lᵢ) ≤ 1

**Theorem 4.6** (Kraft Inequality Converse)
If lengths l₁, l₂, ..., lₙ satisfy Kraft inequality, then there exists a prefix code with these lengths.

### Shannon's Source Coding Theorem

**Theorem 4.7** (Source Coding Theorem)
For a discrete memoryless source with entropy H(X):

- Any prefix code has average length L ≥ H(X)
- There exists a prefix code with L < H(X) + 1

**Definition 4.7** (Efficiency)
η = H(X)/L

### Huffman Coding

**Algorithm 4.1** (Huffman Coding)

1. Create leaf nodes for each symbol with their probabilities
2. While more than one node remains:
   - Find two nodes with lowest probabilities
   - Create parent node with sum of probabilities
   - Assign 0 to left branch, 1 to right branch
3. Read codewords from root to leaves

**Theorem 4.8** (Huffman Optimality)
Huffman coding produces an optimal prefix code.

## Channel Coding

### Channel Models

**Definition 4.8** (Discrete Memoryless Channel)
A channel with input X, output Y, and transition probabilities p(y|x).

**Definition 4.9** (Binary Symmetric Channel)
Input and output alphabets {0,1}, p(1|0) = p(0|1) = p.

**Definition 4.10** (Binary Erasure Channel)
Input {0,1}, output {0,1,?}, p(?|0) = p(?|1) = ε.

### Channel Capacity

**Definition 4.11** (Channel Capacity)
C = max_{p(x)} I(X;Y)

**Theorem 4.9** (Shannon's Channel Coding Theorem)
For a discrete memoryless channel with capacity C:

- If R < C, there exists a code with rate R and arbitrarily small error probability
- If R > C, any code with rate R has error probability bounded away from zero

**Example 4.2** (BSC Capacity)
For binary symmetric channel with crossover probability p:
C = 1 - H(p)

**Example 4.3** (BEC Capacity)
For binary erasure channel with erasure probability ε:
C = 1 - ε

### Error-Correcting Codes

**Definition 4.12** (Block Code)
A code that maps k information bits to n coded bits.

**Definition 4.13** (Code Rate)
R = k/n

**Definition 4.14** (Hamming Distance)
d(x,y) = number of positions where x and y differ.

**Definition 4.15** (Minimum Distance)
d_min = min_{x≠y} d(x,y)

**Theorem 4.10** (Error Detection)
A code can detect up to d_min - 1 errors.

**Theorem 4.11** (Error Correction)
A code can correct up to ⌊(d_min - 1)/2⌋ errors.

## Rate-Distortion Theory

### Distortion Measures

**Definition 4.16** (Distortion Function)
d(x,ŷ): Measure of distortion between source x and reconstruction ŷ.

**Definition 4.17** (Average Distortion)
D = E[d(X,Ŷ)]

### Rate-Distortion Function

**Definition 4.18** (Rate-Distortion Function)
R(D) = min_{p(ŷ|x): E[d(X,Ŷ)]≤D} I(X;Ŷ)

**Theorem 4.12** (Rate-Distortion Theorem)
For distortion D:

- If R > R(D), there exists a code with rate R and distortion ≤ D
- If R < R(D), any code with rate R has distortion > D

**Example 4.4** (Gaussian Source)
For X ~ N(0,σ²) with squared error distortion:
R(D) = (1/2) log(σ²/D) for 0 ≤ D ≤ σ²

## Network Information Theory

### Multiple Access Channel

**Definition 4.19** (Multiple Access Channel)
Channel with two inputs X₁, X₂ and one output Y.

**Theorem 4.13** (MAC Capacity Region)
The capacity region is the convex hull of all (R₁,R₂) satisfying:
R₁ ≤ I(X₁;Y|X₂)
R₂ ≤ I(X₂;Y|X₁)
R₁ + R₂ ≤ I(X₁,X₂;Y)

### Broadcast Channel

**Definition 4.20** (Broadcast Channel)
Channel with one input X and two outputs Y₁, Y₂.

**Theorem 4.14** (Degraded Broadcast Channel)
For X → Y₁ → Y₂:
C₁ = max_{p(x)} I(X;Y₁)
C₂ = max_{p(x)} I(X;Y₂|Y₁)

## Algorithmic Information Theory

### Kolmogorov Complexity

**Definition 4.21** (Kolmogorov Complexity)
K(x) = min{|p| : U(p) = x}, where U is a universal Turing machine.

**Properties:**

1. K(x) ≤ |x| + c for some constant c
2. K(x) is not computable
3. K(x) is universal (up to additive constant)

**Theorem 4.15** (Incompressibility)
For any n, most strings of length n have K(x) ≥ n - log n.

### Algorithmic Mutual Information

**Definition 4.22** (Algorithmic Mutual Information)
I(x:y) = K(x) + K(y) - K(x,y)

**Theorem 4.16** (Symmetry of Information)
I(x:y) = I(y:x) + O(log min(K(x), K(y)))

## Quantum Information Theory

### Quantum Entropy

**Definition 4.23** (Von Neumann Entropy)
S(ρ) = -Tr(ρ log ρ)

**Properties:**

1. S(ρ) ≥ 0
2. S(ρ) = 0 if and only if ρ is pure
3. S(ρ) ≤ log d for d-dimensional system

### Quantum Mutual Information

**Definition 4.24** (Quantum Mutual Information)
I(A;B) = S(A) + S(B) - S(AB)

**Theorem 4.17** (Strong Subadditivity)
S(ABC) + S(B) ≤ S(AB) + S(BC)

## Applications

### Data Compression

- Lossless compression (ZIP, PNG)
- Lossy compression (JPEG, MP3)
- Video compression (H.264, H.265)

### Communication Systems

- Error-correcting codes
- Channel coding
- Network protocols

### Machine Learning

- Feature selection
- Model complexity
- Information bottleneck

### Cryptography

- Perfect secrecy
- Key distribution
- Random number generation

## References

1. Cover, T.M. & Thomas, J.A. (2006). "Elements of Information Theory"
2. Shannon, C.E. (1948). "A Mathematical Theory of Communication"
3. MacKay, D.J.C. (2003). "Information Theory, Inference, and Learning Algorithms"
4. Li, M. & Vitányi, P. (2008). "An Introduction to Kolmogorov Complexity and Its Applications"

## Cross-References

- [Basic Probability Theory](../01_Basic_Probability_Theory.md)
- [Statistics and Data Analysis](../02_Statistics_Data_Analysis.md)
- [Coding Theory](../10_Coding_Theory.md)
- [Quantum Information Theory](../11_Quantum_Information_Theory.md)
