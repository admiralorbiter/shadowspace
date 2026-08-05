# Empirical Audit Reliability Research Protocol & Claim Registry

## Primary Research Question
> **Under what conditions is a counterfactual bias conclusion reliable across evaluators, contexts, prompts, profiles, and sampling seeds?**

---

## Machine-Readable Preregistered Protocol
- **Primary Epsilon ($\epsilon$)**: `0.01`
- **Equivalence Bound ($\delta$)**: `0.02`
- **Multiple Testing Control**: BCa bootstrap intervals ($B=1,000$) for primary outcomes; Benjamini-Hochberg FDR for secondary exploratory contrasts.
- **Human Data Lock**: Automated evaluator metrics remain locked until human rating annotations are finalized.

---

## Claim Gates
1. **Independence Gate**: Evaluator must not be a derived transformation of another panel instrument.
2. **Provenance Gate**: Checkpoint revision, commit SHA, and training data hash must be fail-closed pinned.
3. **Scale Gate**: Population-level empirical claims require $N \ge 1,000$ paired comparisons.
4. **Estimand Consistency Gate**: Cross-domain comparisons must use matched mathematical estimands.
