# 调和分析 | 傅里叶变换·小波·L^p空间（条目与练习）

---

## 1. 学习导引 | Cognitive Primer

- 先修：实分析、复分析、线性代数
- 主线：傅里叶级数→傅里叶变换→L^p空间→小波变换→应用

---

## 2. 傅里叶级数与变换 | Fourier Series & Transform

- 傅里叶级数：f(x) = a₀/2 + Σ(a_n cos nx + b_n sin nx)
- 傅里叶变换：F(ω) = ∫ f(x)e^{-iωx} dx
- 逆变换：f(x) = (1/2π) ∫ F(ω)e^{iωx} dω
- 卷积定理：F(f*g) = F(f)F(g)

---

## 3. L^p空间 | L^p Spaces

- L^p范数：‖f‖_p = (∫ |f|^p)^{1/p}
- 赫尔德不等式：|∫ fg| ≤ ‖f‖_p ‖g‖_q，1/p + 1/q = 1
- 闵可夫斯基不等式：‖f+g‖_p ≤ ‖f‖_p + ‖g‖_p
- 对偶性：L^p* = L^q（1 < p < ∞）

---

## 4. 小波变换 | Wavelet Transform

- 连续小波：W_f(a,b) = ∫ f(t)ψ_{a,b}(t) dt
- 离散小波：ψ_{j,k}(t) = 2^{j/2} ψ(2^j t - k)
- 多分辨率分析：嵌套子空间 V_j ⊂ V_{j+1}
- 正交小波基：Haar、Daubechies

---

## 5. 典例 | Worked Examples

1) 计算：f(x) = x 在 [-π,π] 的傅里叶级数
2) 证明：高斯函数 e^{-x²/2} 的傅里叶变换仍为高斯
3) 构造：Haar 小波基的前几个函数

---

## 6. 练习（8题） | Exercises (8)

1) 计算：f(x) = |x| 在 [-π,π] 的傅里叶系数
2) 证明：傅里叶变换的平移性质 F(f(x-a)) = e^{-iωa}F(ω)
3) 证明：赫尔德不等式在 p=1, q=∞ 的情况
4) 计算：L^2[0,1] 中函数 f(x)=x 的范数
5) 证明：L^p 空间的完备性（概念性）
6) 讨论：傅里叶变换与卷积的关系
7) 构造：简单信号的小波分解（Haar 例）
8) 连接：傅里叶变换与微分算子的关系

---

## 7. 认知提示 | Tips

- 傅里叶变换将时域信号转换到频域
- 小波提供时频局部化，适合非平稳信号

---

## 8. 参考 | References

- `https://en.wikipedia.org/wiki/Fourier_transform`
- `https://en.wikipedia.org/wiki/Wavelet`
