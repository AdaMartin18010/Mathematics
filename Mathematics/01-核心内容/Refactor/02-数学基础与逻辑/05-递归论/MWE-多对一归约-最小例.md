# MWE｜多对一（many-one）归约（最小例）

## 1. 定义

- 语言 A 多对一归约到语言 B（记 A ≤_m B），若存在可计算函数 f，使得 x∈A ⇔ f(x)∈B。
- 性质：若 A ≤_m B 且 B 可判定，则 A 可判定；故若 A 不可判定且 A ≤_m B，则 B 不可判定。 [Rogers 1967]

## 2. 从 HALT 到 K 的归约骨架

- 记 HALT = {⟨M,w⟩ | 机 M 在 w 上停机}，K = {⟨M,w⟩ | M 接受 w}（或把 K 定义为 M 在 ⟨M⟩ 上接受的集合，视课本变体）。
- 归约思想：给定 ⟨M,w⟩，构造机器 N 与输入 x，使得
  - 若 M 在 w 上停机，则 N 接受 x；
  - 若 M 在 w 上不停机，则 N 不接受 x。
- 具体构造（骨架）：令 N 在输入 x 上先模拟 M(w)；若停机则转入一个必接受分支（例如输出 1 并停机），否则循环。定义 f(⟨M,w⟩)=⟨N,x⟩。
- 则 ⟨M,w⟩∈HALT ⇔ f(⟨M,w⟩)∈K，且 f 可计算，故 HALT ≤_m K。由 HALT 不可判定，得 K 不可判定。 [Davis 1958; Rogers 1967; Soare 1987]

## 3. 参考

- Davis, Computability and Unsolvability.
- Rogers, Theory of Recursive Functions and Effective Computability.
- Soare, Recursively Enumerable Sets and Degrees.
