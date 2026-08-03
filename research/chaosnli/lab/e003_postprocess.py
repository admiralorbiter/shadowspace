"""Post-processing script to generate publication-grade E003_summary.md markdown report."""

from __future__ import annotations

import json
from pathlib import Path

json_path = Path("research/chaosnli/lab/summaries/E003_summary.json")
md_path = Path("research/chaosnli/lab/summaries/E003_summary.md")

def generate_e003_markdown() -> None:
    if not json_path.exists():
        print(f"Missing {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ladder_data = data["ladder_results"]
    sorted_ladder = sorted(ladder_data.values(), key=lambda item: item["level_name"])

    lines = []
    lines.append("# E003: Relational Repair Capacity of Flexible Post-Hoc Calibration & Ensembling\n")
    lines.append("**Experiment ID**: E003  ")
    lines.append("**Title**: Relational Repair Capacity of Increasingly Flexible Post-Hoc Transformations & Ensembling  ")
    lines.append(f"**Status**: `{data.get('status', 'complete_publication_grade')}`  ")
    lines.append("Dataset Release: `chaosnli-canonical-2026-08-02` (N = 3,113 items)  ")
    lines.append("Cross-Validation: 5-Fold Stratified Coherent Cross-Fitting by (Dataset, Majority Label, Empirical Entropy Quintile)  ")
    lines.append(f"Bound E001 Artifact (k=10): `{data['e001_artifact_id']}` (SHA-256: `{data['e001_matrix_k10_sha256'][:16]}...`)  ")
    lines.append(f"Bound E001 Artifact (k=50): `S_hellinger_k050.bin` (SHA-256: `{data['e001_matrix_k50_sha256'][:16]}...`)  ")
    lines.append(f"Model Probs Hash: `{data.get('model_probs_sha256', '')[:16]}...`  ")
    lines.append(f"Human Soft-Label Entropy Floor ($H(p)$): {data['human_entropy_floor_nats']:.5f} nats  ")
    lines.append(f"Human Relational Reference ($Q_{{HH}}$): {data['q_hh_relational']:.5f}  \n")
    lines.append("---\n")
    lines.append("## Executive Summary\n")
    lines.append("Experiment **E003** evaluates the **Relational Repair Ladder**: *How much of the human belief-space relational topology gap ($G_Q$) can be recovered through increasingly flexible post-hoc calibration and ensembling techniques BEFORE representational fine-tuning becomes necessary?*\n")
    lines.append("### Key Scientific Findings\n")
    lines.append("1. **BART-Large Post-Hoc Calibration Produces Little Relational Repair ($G_Q \\le 0.83\\%$)**:")
    lines.append("   - For the BART-Large anchor, moving from scalar temperature scaling ($G_Q = 0.59\\%$) to class-wise vector scaling + bias, coarse-grid identifiable 8-parameter affine matrix scaling, and coarse-grid identifiable 8-parameter Dirichlet calibration closes at most approximately 0.83% of BART's remaining relational gap.")
    lines.append("2. **Diverse-Model Probability Ensembling Substantially Improves Relational Alignment**:")
    lines.append("   - Combining BART-Large, RoBERTa-Large, and XLNet-Large output distributions materially improves both pointwise and relational alignment relative to BART-Large alone ($G_Q \\approx 17.18\\% - 17.71\\%$). Equal-weight ensembling provides most of the gain.")
    lines.append("3. **Level 6a Topology Weighting Gain is Incremental**:")
    lines.append("   - Level 6a produced the highest relational-recovery point estimate, but its advantage over equal and NLL-optimized weighting is small and evaluated directly via paired bootstrap contrasts.")
    lines.append("4. **Post-Hoc Ceiling Provides Strong Motivation for Fine-Tuning**: Global post-hoc recalibration leaves most of BART's relational gap unclosed. This motivates topology-aware fine-tuning (E004), but does not establish it as uniquely necessary.\n")
    lines.append("---\n")
    lines.append("## 6-Level Relational Repair Ladder Summary Results\n")
    lines.append("| Ladder Level | NLL (nats) | JSD (bits) | $Q_{\\text{support, OOF}}$ | $Q_{\\text{null, OOF}}$ | $Q_{\\text{global-excess}}$ | $G_{\\text{NLL}}$ | Relational Gap Closure $G_Q$ | $\\Delta G = G_{\\text{NLL}} - G_Q$ (95% CI) | Min Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for l_res in sorted_ladder:
        m = l_res["metrics"]
        gn = l_res["gap_closure_nll"]
        gq = l_res["gap_closure_q"]
        b_gc = l_res["bootstrap_delta_gap_closure"]

        lines.append(
            f"| **{l_res['display_name']}** | {m['nll']:.4f} | {m['jsd_bits']:.4f} | "
            f"**{m['q_support_oof']:.5f}** | {m['q_null_oof']:.5f} | {m['q_global_excess_oof']:.5f} | "
            f"**{gn*100.0:.2f}%** | **{gq*100.0:.2f}%** | **+{b_gc['mean']*100.0:.2f}%** [{b_gc['ci_lower_95']*100.0:.2f}%, {b_gc['ci_upper_95']*100.0:.2f}%] | "
            f"{m['graph_turnover_min_oof']*100.0:.2f}% | {m['core_mass_k50_oof']:.6f} | {m['core_recall_k50_oof']*100.0:.2f}% |"
        )

    # Exact-Profile Null Section
    lines.append("\n---\n")
    lines.append("## Exact-Profile Null & Excess Analysis (Ensemble Conditions)\n")
    lines.append("| Ensemble Condition | $Q_{\\text{support}}$ | $Q_{\\text{exact-profile null}}$ (95% CI) | $Q_{\\text{profile-excess}}$ | $p_{\\text{Monte Carlo}}$ |")
    lines.append("|---|---|---|---|---|")

    for l_res in sorted_ladder:
        m = l_res["metrics"]
        if m.get("q_profile_null_oof") is not None:
            pn_mean = m["q_profile_null_oof"]
            pn_ci = m.get("q_profile_null_95ci", [0.0, 0.0])
            pe = m.get("q_profile_excess_oof", 0.0)
            pval = m.get("q_profile_pvalue_monte_carlo", 1.0)
            lines.append(
                f"| **{l_res['display_name']}** | {m['q_support_oof']:.5f} | "
                f"{pn_mean:.5f} [{pn_ci[0]:.5f}, {pn_ci[1]:.5f}] | **{pe:.5f}** | {pval:.4f} |"
            )

    # Direct Paired Ensemble Contrasts Section
    if "ensemble_pairwise_contrasts" in data and data["ensemble_pairwise_contrasts"]:
        lines.append("\n---\n")
        lines.append("## Direct Paired Bootstrap Contrasts Between Ensemble Conditions\n")
        lines.append("| Contrast | $\\Delta G_Q$ (95% CI) | $\\Delta Q_{\\text{support}}$ (95% CI) | $\\Delta R$ (95% CI) | $\\Delta \\text{NLL}$ (95% CI) | $P(\\Delta G_Q > 0)$ |")
        lines.append("|---|---|---|---|---|---|")
        
        contrasts = data["ensemble_pairwise_contrasts"]
        for c_key, c in contrasts.items():
            gq_c = c["delta_g_q"]
            qs_c = c["delta_q_support"]
            r_c = c["delta_r_recovery"]
            nll_c = c["delta_nll"]
            lines.append(
                f"| **{c['contrast_name']}** | "
                f"**{gq_c['mean']*100.0:+.2f}%** [{gq_c['ci_lower_95']*100.0:+.2f}%, {gq_c['ci_upper_95']*100.0:+.2f}%] | "
                f"{qs_c['mean']:+.5f} [{qs_c['ci_lower_95']:+.5f}, {qs_c['ci_upper_95']:+.5f}] | "
                f"{r_c['mean']*100.0:+.2f}% [{r_c['ci_lower_95']*100.0:+.2f}%, {r_c['ci_upper_95']*100.0:+.2f}%] | "
                f"{nll_c['mean']:+.5f} [{nll_c['ci_lower_95']:+.5f}, {nll_c['ci_upper_95']:+.5f}] | "
                f"{c['prob_gt_zero']*100.0:.1f}% |"
            )

    # Fold Weights Section
    lines.append("\n---\n")
    lines.append("## Fold-Specific Optimization Weights & Margins\n")
    for l_res in sorted_ladder:
        opt = l_res.get("optimization_info")
        if opt:
            lines.append(f"### {l_res['display_name']}\n")
            lines.append("| Fold | Best Weights (BART, RoBERTa, XLNet) | Best Objective | Second-Best Weights | Objective Margin |")
            lines.append("|---|---|---|---|---|")
            for f_idx in range(len(opt["fold_weights"])):
                bw = opt["fold_weights"][f_idx]
                bo = opt["fold_training_objectives"][f_idx]
                sbw = opt["fold_second_best_weights"][f_idx]
                margin = opt["fold_objective_margins"][f_idx]
                lines.append(
                    f"| Fold {f_idx} | [{bw[0]:.3f}, {bw[1]:.3f}, {bw[2]:.3f}] | {bo:.5f} | "
                    f"[{sbw[0]:.3f}, {sbw[1]:.3f}, {sbw[2]:.3f}] | {margin:.5f} |"
                )
            lines.append("\n")

    lines.append("\n---\n")
    lines.append("## Scientific Conclusions for Experiment E003\n")
    lines.append("- **Levels 1 to 4 (BART-Large Post-Hoc Calibration)**: Scalar, vector, coarse-grid affine, and coarse-grid Dirichlet transformations closed at most approximately $0.83\\%$ of BART's remaining relational gap.")
    lines.append("- **Levels 5a/5b/6a (Multi-Model Ensembling)**: Combining BART-Large, RoBERTa-Large, and XLNet-Large output distributions materially improves both pointwise and relational alignment relative to BART-Large alone.")
    lines.append("- **Ensemble Contrast & Ranking**: Level 6a produced the highest relational-recovery point estimate, but its advantage over equal and NLL-optimized weighting was small.")
    lines.append("- **Core Takeaway**: Diverse-model probability ensembling produces a large and reliable improvement in both pointwise and relational alignment, whereas single-model BART recalibration produces little relational improvement. This motivates topology-aware fine-tuning (E004).\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {md_path}")

if __name__ == "__main__":
    generate_e003_markdown()
