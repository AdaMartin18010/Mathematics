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
  sorry

-- 练习2：最大公约数性质
-- 对应剑桥大学Part I数论
theorem gcd_property (a b : ℕ) :
  gcd a b ∣ a ∧ gcd a b ∣ b ∧
  (∀ d : ℕ, d ∣ a → d ∣ b → d ∣ gcd a b) := by
  -- HINT: 欧几里得算法性质；`Nat.gcd_rec`/`Nat.gcd_comm`
  sorry

-- 练习3：互质性质
-- 对应芝加哥大学数论标准
theorem coprime_property (a b : ℕ) :
  coprime a b ↔ gcd a b = 1 := by
  -- HINT: 互素与贝祖等式；`Nat.coprime_iff_gcd_eq_one`
  sorry

-- 练习4：素数性质
-- 对应华威大学数论课程
theorem prime_property (p : ℕ) :
  Prime p ↔ p > 1 ∧ (∀ a b : ℕ, p ∣ a * b → p ∣ a ∨ p ∣ b) := by
  -- HINT: 质数定义与可除性；`Nat.prime` 相关定理
  sorry

-- 练习5：中国剩余定理
-- 对应国际标准：Chinese Remainder Theorem
theorem chinese_remainder (a b m n : ℕ) :
  coprime m n →
  ∃ x : ℕ, x ≡ a [MOD m] ∧ x ≡ b [MOD n] := by
  -- HINT: 算术基本定理；唯一分解与素因子
  sorry

-- 练习6：费马小定理
-- 对应巴黎第六大学数论标准
theorem fermat_little (a p : ℕ) :
  Prime p → ¬p ∣ a → a ^ (p - 1) ≡ 1 [MOD p] := by
  -- HINT: 同余性质；`ZMod` 工具与 `mod` 的等价关系
  sorry

-- 练习7：欧拉函数性质
-- 对应伦敦大学学院数论课程
theorem euler_phi_property (n : ℕ) :
  φ n = (Finset.range n).filter (coprime n).card := by
  -- HINT: 费马小定理/欧拉定理；检索 `FermatLittle` / φ函数引理
  sorry

-- 练习8：二次剩余
-- 对应国际标准：Quadratic Residues
theorem quadratic_residue (a p : ℕ) :
  Prime p → p > 2 →
  (∃ x : ℕ, x ^ 2 ≡ a [MOD p]) ↔ a ^ ((p - 1) / 2) ≡ 1 [MOD p] := by
  -- HINT: 中国剩余定理；`ChineseRemainder` 相关引理
  sorry

end NumberTheory
