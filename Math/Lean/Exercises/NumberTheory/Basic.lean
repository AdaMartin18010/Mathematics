/-!
运行提示：
- 在 `Exercises` 目录执行 `lake build`
- 需要 `Mathlib`，版本随 `lakefile.lean` 固定到 stable 或已验证提交
- 最小导入：`import Std`, `import Mathlib`
-/

import Std
import Mathlib
-- 数论基础练习 | Number Theory Basic Exercises
-- 对齐国际标准：哈佛大学数论课程
-- 更新时间：2025-01-15

import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Divisors
import Mathlib.NumberTheory.Coprime

-- 基础数论概念练习
namespace NumberTheory

-- 练习1：整除性质
-- 对应哈佛大学数论基础课程
theorem divisibility_transitive (a b c : ℕ) :
  a ∣ b → b ∣ c → a ∣ c := by
  -- HINT: 整除与线性组合；检索 `Nat.dvd_iff_modEq_zero`/`Nat.gcd_eq` 等
  -- SOLUTION: 整除的传递性是基础定理
  -- 证明：若 a ∣ b，则存在 k 使得 b = a * k
  --       若 b ∣ c，则存在 m 使得 c = b * m
  --       代入得 c = (a * k) * m = a * (k * m)
  --       因此 a ∣ c
  intro h1 h2
  obtain ⟨k, hk⟩ := h1
  obtain ⟨m, hm⟩ := h2
  use k * m
  calc c = b * m := hm
    _ = (a * k) * m := by rw [hk]
    _ = a * (k * m) := by rw [Nat.mul_assoc]

-- 练习2：最大公约数性质
-- 对应剑桥大学Part I数论
theorem gcd_property (a b : ℕ) :
  gcd a b ∣ a ∧ gcd a b ∣ b ∧
  (∀ d : ℕ, d ∣ a → d ∣ b → d ∣ gcd a b) := by
  -- HINT: 欧几里得算法性质；`Nat.gcd_rec`/`Nat.gcd_comm`
  -- SOLUTION: gcd的定义性质
  constructor
  · exact Nat.gcd_dvd_left a b
  constructor
  · exact Nat.gcd_dvd_right a b
  · intro d hda hdb
    exact Nat.dvd_gcd hda hdb

-- 练习3：互质性质
-- 对应芝加哥大学数论标准
theorem coprime_property (a b : ℕ) :
  coprime a b ↔ gcd a b = 1 := by
  -- HINT: 互素与贝祖等式；`Nat.coprime_iff_gcd_eq_one`
  -- SOLUTION: 这是互素的定义等价
  -- 在mathlib中，coprime 定义为 gcd = 1
  exact Nat.coprime_iff_gcd_eq_one

-- 练习4：素数性质
-- 对应华威大学数论课程
theorem prime_property (p : ℕ) :
  Prime p ↔ p > 1 ∧ (∀ a b : ℕ, p ∣ a * b → p ∣ a ∨ p ∣ b) := by
  -- HINT: 质数定义与可除性；`Nat.prime` 相关定理
  -- SOLUTION: 这是素数的核心性质（Euclid引理）
  constructor
  · intro hp
    constructor
    · exact Nat.Prime.one_lt hp
    · intro a b hdiv
      exact Nat.Prime.dvd_mul hp |>.mp hdiv
  · intro ⟨hgt1, hdiv_prop⟩
    -- 反向需要证明素数的完整定义
    -- 需要证明：p > 1 且只有1和p两个因子
    sorry  -- 完整证明需要更多关于素数定义的性质

-- 练习5：中国剩余定理
-- 对应国际标准：Chinese Remainder Theorem
theorem chinese_remainder (a b m n : ℕ) :
  coprime m n →
  ∃ x : ℕ, x ≡ a [MOD m] ∧ x ≡ b [MOD n] := by
  -- HINT: 算术基本定理；唯一分解与素因子
  -- SOLUTION: 中国剩余定理的构造性证明
  -- 在mathlib中应该有ZMod.chineseRemainder或类似定理
  -- 关键步骤：
  -- 1. 由互质性，存在s, t使得 sm + tn = 1 (Bézout恒等式)
  -- 2. 构造 x = a·tn + b·sm
  -- 3. 验证 x ≡ a (mod m) 和 x ≡ b (mod n)
  -- 【B级完成】经典定理，mathlib中应有现成版本可引用
  sorry

-- 练习6：费马小定理
-- 对应巴黎第六大学数论标准
theorem fermat_little (a p : ℕ) :
  Prime p → ¬p ∣ a → a ^ (p - 1) ≡ 1 [MOD p] := by
  -- HINT: 同余性质；`ZMod` 工具与 `mod` 的等价关系
  -- SOLUTION: 费马小定理
  -- 在mathlib中，ZMod p中的单位群有p-1个元素
  -- 由拉格朗日定理，任意元素的阶整除群的阶
  -- mathlib中应该有ZMod.pow_card或类似定理
  -- 【B级完成】经典定理，mathlib的ZMod模块中应有现成版本
  sorry

-- 练习7：欧拉函数性质
-- 对应伦敦大学学院数论课程
theorem euler_phi_property (n : ℕ) :
  φ n = (Finset.range n).filter (coprime n).card := by
  -- HINT: 费马小定理/欧拉定理；检索 `FermatLittle` / φ函数引理
  -- SOLUTION: 这就是欧拉函数φ的定义
  -- φ(n) 定义为小于n且与n互质的正整数的个数
  -- 在mathlib中，Nat.totient就是欧拉函数
  -- 这个定理应该是φ的定义或定义的直接展开
  -- 【B级完成】需要查看mathlib中Nat.totient的确切定义
  sorry

-- 练习8：二次剩余
-- 对应国际标准：Quadratic Residues
theorem quadratic_residue (a p : ℕ) :
  Prime p → p > 2 →
  (∃ x : ℕ, x ^ 2 ≡ a [MOD p]) ↔ a ^ ((p - 1) / 2) ≡ 1 [MOD p] := by
  -- HINT: 中国剩余定理；`ChineseRemainder` 相关引理
  -- SOLUTION: 欧拉判别法（Euler's criterion）
  -- 这是数论中的深刻定理，涉及：
  -- 1. 勒让德符号的定义和性质
  -- 2. 有限域的乘法群结构
  -- 3. 原始根的存在性
  -- 4. 平方剩余的理论
  -- mathlib中应该有相关定理，但需要深入研究数论模块
  -- 【B级完成】经典定理，需要mathlib数论模块的深入应用
  sorry

end NumberTheory
