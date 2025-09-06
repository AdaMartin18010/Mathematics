import Std
import Mathlib

namespace Exercises.Basics

-- 基础加法交换律练习
theorem add_comm_ex (a b : Nat) : a + b = b + a := by
  exact Nat.add_comm a b

-- 基础乘法交换律练习
theorem mul_comm_ex (a b : Nat) : a * b = b * a := by
  exact Nat.mul_comm a b

-- 基础加法结合律练习
theorem add_assoc_ex (a b c : Nat) : (a + b) + c = a + (b + c) := by
  exact Nat.add_assoc a b c

-- 基础乘法结合律练习
theorem mul_assoc_ex (a b c : Nat) : (a * b) * c = a * (b * c) := by
  exact Nat.mul_assoc a b c

-- 基础分配律练习
theorem mul_add_ex (a b c : Nat) : a * (b + c) = a * b + a * c := by
  exact Nat.mul_add a b c

end Exercises.Basics
