# MWE｜AI与自动证明（Lean/Coq 最小证明）

## Lean

```lean
theorem imp_self (P : Prop) : P → P := fun h => h
```

## Coq

```coq
Theorem imp_self : forall P:Prop, P -> P.
Proof. intros P H. exact H. Qed.
```

## Coq 极简证明（P -> P）

```coq
(* 需要 Coq 标准库，命令行 coqc 可直接处理 *)
Section Minimal.
Variable P : Prop.
Theorem idP : P -> P.
Proof.
  intro p. exact p.
Qed.
End Minimal.
```

- 要点：最小可验证证明，展示形式系统的“语法→可检验证明”。
- 参考：Lean/Coq 文档
