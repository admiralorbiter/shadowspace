# Study 2B: Exploratory External Test Using VariErr NLI

- **Document type:** empirical report
- **Status:** exploratory and statistically inconclusive ($p=0.2045$)
- **Scope:** Level-1 opinion-profile homogeneity in 500 matched VariErr NLI items

---

## 1. Overview and Match Statistics

We perform an exploratory external test of the two-level opinion architecture using 500 matched items from VariErr NLI (Weber-Genzel et al., ACL 2024), containing 7,732 human validity judgments over re-annotated MNLI items. The test evaluates whether items sharing identical Level-1 opinion profiles exhibit lower validity dispersion than items sampled across different profiles.

- **Matched Items**: 500 / 500 VariErr items matched to ChaosNLI-M items (`source_pair_id`).
- **Multi-Item Profiles**: 52 multi-item profiles (group sizes: min 2, median 3, max 16; total 188 matched items in multi-item groups).
- **Variance Estimation**: Sample standard deviations use Bessel's correction ($n-1$ denominator, `ddof=1`); multi-profile means apply equal weighting across profiles.

---

## 2. Descriptive and Permutation Null Results

**Table 1: Profile Homogeneity Statistics on Matched VariErr Items**

| Metric | Value | Description |
|---|---|---|
| Matched Items | 500 | Items present in both VariErr and ChaosNLI-M |
| Multi-Item Profiles ($\lvert g\rvert > 1$) | 52 | Distinct ChaosNLI vote profiles containing $\ge 2$ VariErr items |
| Overall Item-Level Validity SD | 0.1413 | Total sample SD of explanation validity ratios ($n=500$) |
| Overall Item-Level Validity Variance | 0.0200 | Total sample variance ($0.1413^2 = 0.0200$) |
| Mean Within-Profile Validity SD | 0.1060 | Average sample SD across 52 multi-item profiles |
| Mean Within-Profile Validity Variance | 0.0180 | Average sample variance across 52 multi-item profiles |
| **Null-Relative SD Reduction** | **7.8%** | Observed within-profile SD ($0.1060$) vs. null mean ($0.1150$) |
| Descriptive SD Reduction vs. Overall | 25.0% | Within-profile SD ($0.1060$) vs. overall SD ($0.1413$). *Note: partly reflects small-sample downward SD bias.* |
| **500,000-Permutation Null Mean Within-Profile SD** | **0.1150** | Profile-size preserved label permutations ($N_{\text{perm}} = 500,000$, native Rust 16-core, $190.5\text{ ms}$) |
| **Permutation $p$-value** | $p = 0.2045$ | **Inconclusive** — observed within-profile SD ($0.1060$) vs. null mean ($0.1150$), $p = 0.2045$ ($102,248 / 500,000$) |

---

## 3. Scientific Takeaway

Matched items sharing an exact ChaosNLI vote profile exhibit a **7.8% reduction in validity standard deviation** compared to the profile-size preserved null mean ($0.1060$ vs. $0.1150$).

**Statistical Status**: A **500,000-permutation** profile-size preserved null (executed natively in Rust across 16 CPU cores in $190.5\text{ ms}$) yields a mean within-profile SD of $0.1150$ ($p = 0.2045$, $102,248 / 500,000$ resamples). The observed within-profile SD ($0.1060$) is not statistically distinguishable from the permutation null.

**Scientific Takeaway**: Under the selected profile-dispersion statistic, we found no evidence that exact vote-profile identity predicts explanation-validity composition ($p = 0.2045$). This is consistent with the view that aggregate Level-1 vote distributions are insufficient for recovering rationale structure, but does not establish a zero effect. Larger sample sizes and finer-grained text annotations are needed for a confirmatory test.
