# E003 Methodological Audit & Execution Walkthrough

All methodological audit requirements and review feedback for Experiment E003 have been fully incorporated, executed, and verified.

## Status Matrix

- `analysis_status`: `"complete"`
- `methodological_audit`: `"passed_with_interpretation_notes"`
- `manuscript_status`: `"requires_results_sync"`
- `submission_status`: `"pre_submission"`

---

## Methodological Improvements Executed

1. **Training-Only Human Target Construction**:
   - For each fold $f \in \{0..4\}$, a dedicated $N_{\text{train}} \times N_{\text{train}}$ human posterior support target $S_{\text{train}, f}$ was constructed from 200 Dirichlet posterior draws strictly within training items.
   - Held-out items no longer influence neighbor competition in the optimization target.

2. **Always-Active Probability Validation**:
   - Replaced `debug_assert!` with an explicit, active `assert!` checking probability non-negativity and finiteness.
   - Applied $L_1$ probability vector normalization to guarantee exact unit sum ($1.0 \pm 1e-6$) across all ensemble blends.

3. **Persisted Fold Weights & Objective Margins**:
   - Recorded fold-specific weight vectors, best training objective values, second-best weight vectors, and objective margins for Level 5b and Level 6a in [`E003_summary.json`](file:///c:/Users/admir/Github/shadowspace/research/chaosnli/lab/summaries/E003_summary.json).

4. **Exact-Profile Permutation Null Analysis**:
   - Implemented 10,000 exact-profile permutations for Levels 5a, 5b, and 6a.
   - Evaluated $Q_{\text{profile-null}}$, $Q_{\text{profile-excess}}$, 95% CIs, and add-one Monte Carlo $p$-values.

5. **Direct Paired Bootstrap Contrasts**:
   - Implemented direct paired item-level bootstrap contrasts across 1,000 resamples between ensemble conditions:
     - Level 6a Topology vs Level 5b NLL
     - Level 6a Topology vs Level 5a Equal
     - Level 5b NLL vs Level 5a Equal

---

## Core Findings & Scientific Conclusions

### 1. Direct Paired Bootstrap Contrasts Between Ensemble Conditions

| Contrast | $\Delta G_Q$ (95% CI) | $\Delta Q_{\text{support}}$ (95% CI) | $\Delta R$ (95% CI) | $\Delta \text{NLL}$ (95% CI) | $P(\Delta G_Q > 0)$ |
|---|---|---|---|---|---|
| **Level 5b NLL vs Level 5a Equal** | **+0.07%** [-0.19%, +0.37%] | +0.00005 [-0.00012, +0.00023] | +0.06% [-0.17%, +0.33%] | +0.00029 [-0.00015, +0.00074] | 68.7% |
| **Level 6a Topology vs Level 5b NLL** | **-0.07%** [-0.69%, +0.59%] | -0.00004 [-0.00043, +0.00037] | -0.06% [-0.62%, +0.53%] | +0.02966 [+0.02434, +0.03558] | 40.5% |
| **Level 6a Topology vs Level 5a Equal** | **+0.00%** [-0.66%, +0.70%] | +0.00001 [-0.00040, +0.00043] | +0.00% [-0.59%, +0.62%] | +0.02995 [+0.02430, +0.03622] | 50.7% |

*Interpretation*: No ensemble weighting strategy significantly or meaningfully outperformed the others; all direct $G_Q$ contrasts were smaller than one percentage point and their 95% confidence intervals included zero. Equal weighting accounts for essentially all of the relational alignment gain.

### 2. Exact-Profile Null & Excess Analysis

| Ensemble Condition | $Q_{\text{support}}$ | $Q_{\text{exact-profile null}}$ (95% CI) | $Q_{\text{profile-excess}}$ | $p_{\text{Monte Carlo}}$ |
|---|---|---|---|---|
| **Level 5a: Equal-Weight Ensemble** | 0.01112 | 0.01115 [0.01110, 0.01120] | **-0.00003** | 0.8916 |
| **Level 5b: Convex NLL Ensemble** | 0.01116 | 0.01118 [0.01113, 0.01124] | **-0.00002** | 0.8005 |
| **Level 6a: Topology Ensemble** | 0.01113 | 0.01115 [0.01110, 0.01119] | **-0.00001** | 0.7299 |

*Interpretation*: The measured model–human alignment is an alignment with aggregate judgment-distribution geometry rather than demonstrable item-specific semantic organization beyond exact vote profiles ($Q_{\text{profile-excess}} \approx 0.00000, p \ge 0.73$).

---

## Paper-Ready Core Abstract / Synthesis Statement

> Across nine NLI models, probability-space neighborhoods contained stable, non-random alignment with the relational geometry induced by aggregate human label distributions. Conventional temperature scaling substantially improved soft-label negative log-likelihood while closing less than 1% of the corresponding relational gap, despite replacing considerable neighborhood mass. More flexible post-hoc recalibration of BART-Large produced similarly limited relational improvement. In contrast, combining BART-Large, RoBERTa-Large, and XLNet-Large probabilities closed approximately 17% of BART-Large’s remaining relational gap. Equal, NLL-selected, and topology-selected weighting produced statistically indistinguishable relational results, indicating that model combination—not the particular global weighting objective—accounted for the gain. Exact-vote-profile controls yielded negligible excess support for every ensemble, showing that the observed improvement concerns the organization of aggregate human judgment distributions rather than detectable item-specific alignment beyond those distributions.
