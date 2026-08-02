# Study 2A: Exploratory Text-Space and Heuristic Taxonomy Analysis

*Exploratory Analysis of Level-2 Item Heterogeneity within Collective Opinion Profiles*

---

## 1. Overview

This document reports exploratory analyses on Level-2 item heterogeneity within identical Level-1 human collective opinion profiles. It evaluates surface text-based heuristic taxonomies, sentence-embedding text distance, lexicographic tie resolution, and operational case-routing candidate edge classification.

> **Status Note**: This analysis is exploratory and uses surface-text heuristics. For confirmatory external validation using expert-annotated human validity judgments, see [`STUDY2B_VARIERR_EXTERNAL_VALIDATION.md`](STUDY2B_VARIERR_EXTERNAL_VALIDATION.md).

---

## 2. Level-1 Profile Graph

Among 3,113 ChaosNLI items, **1,604 unique opinion profiles** exist. Of 684 multi-item profiles (covering 2,193 items), **337 profiles (49.3%)** contain items from both SNLI and MNLI.

## 3. Profile-Level Model Dispersion

Mean model dispersion across 684 multi-item profiles is **0.2793 Hellinger distance** (ranging from BART-Large $0.2185$ to BERT-Base $0.3415$).

Correlations of profile dispersion with profile features:
- Entropy $H(p)$: $r = +0.1418$
- Profile frequency $|g|$: $r = -0.1001$
- Dominant class $\max(p)$: $r = -0.0519$

Model dispersion is only weakly associated with these three profile-level summaries, indicating that identifying its linguistic drivers requires direct feature analysis.

## 4. Operational Case-Routing Edge Ledger

We evaluate 307,662 candidate directed edges (where either $w_{\text{human}} > 0$ or $c_{\text{model}} > 0$) using 25th/75th percentile quantile thresholds for operational case-routing:

| Operational Category | Candidate Edges | Percentage | Review Action |
|---|---|---|---|
| Unclassified / Intermediate | 156,999 | 51.0% | Background candidate pool |
| Model Artifact Candidate | 69,838 | 22.7% | High model consensus, low human & text support |
| Semantic Similarity Divergence | 67,455 | 21.9% | High model consensus & text similarity, low human support |
| Human Relation Missed by Models | 6,835 | 2.2% | High human support & text similarity, low model consensus |
| Same Opinion, Distinct Language | 5,743 | 1.9% | High human support, low model & text support |
| Broadly Shared Relation | 792 | 0.3% | Consensus reference edge |

*Operational Note*: These category percentages represent operational routing labels under a specific thresholding scheme, not natural population prevalence rates.

## 5. Automated Proxy-Taxonomy Benchmark

**Table 1: Heuristic Proxy-Taxonomy Tie Resolution ($k=10$)**

| Strategy | MAP@10 | 95% Monte Carlo CI | Delta MAP@10 (vs Random) | Monte Carlo $p$-value |
|---|---|---|---|---|
| 500-Pass Random Tie Baseline | 0.52967 | [0.52714, 0.53217] | — | — |
| Lexicographic $(d_H, d_{\text{text}})$ | **0.53502** | — | **+0.00535** | **$p \le 0.002$** |
| Pure Text Space | 0.59650 | — | +0.06683 | $p \le 0.002$ |

*Observation*: Pure Text achieves highest retrieval against text-derived heuristic categories, but discards opinion topology ($Q_{NX}^{\text{soft}} = 0.0041$). Lexicographic tie-breaking achieves a modest $+0.00535$ MAP@10 improvement over random tie resolution ($p \le 0.002$, minimum one-sided Monte Carlo bound for $N_{\text{perm}} = 500$) while preserving exact opinion rank order.
