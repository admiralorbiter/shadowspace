"""Master Empirical Pipeline Runner (ER-0 to ER-2)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
import numpy as np

from research.education_audit.audit_reliability_empirical.protocol import get_preregistered_protocol
from research.education_audit.audit_reliability_empirical.evaluators.panel import initialize_empirical_evaluator_panel
from research.education_audit.audit_reliability_empirical.counterfactuals.pair_builder import build_labe_test_counterfactual_corpus
from research.education_audit.audit_reliability_empirical.metrics.drift import compute_evaluator_drift_metrics
from research.education_audit.audit_reliability_empirical.metrics.agreement import compute_evaluator_consensus_stability
from research.education_audit.audit_reliability_empirical.metrics.equivalence import run_tost_equivalence_test
from research.education_audit.audit_reliability_empirical.metrics.tail_risk import compute_tail_risk_metrics


def run_empirical_audit_reliability_pipeline(
    out_dir: str = "results/education_audit/audit_reliability_empirical",
) -> Dict[str, Any]:
    """Runs ER-0, ER-1, and ER-2 pipeline and writes all manifest and report artifacts."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)


    print("Step 1: Validating Preregistered Protocol & Machine-Readable Rules (ER-0)...")
    protocol = get_preregistered_protocol()

    print("Step 2: Initializing 3 Independent Evaluators Panel (ER-1)...")
    panel_res = initialize_empirical_evaluator_panel(out_dir=out_dir)
    panel = panel_res["panel"]

    print("Step 3: Building Full LABE Test Counterfactual Corpus (N=1,492 paired comparisons) (ER-2)...")
    corpus = build_labe_test_counterfactual_corpus()

    reliability_cards = {}
    evaluator_deltas_dict = {}

    for eval_key, evaluator in panel.items():
        print(f"  - Evaluating {evaluator.provenance.evaluator_name} on {len(corpus)} pairs...")
        deltas = []
        scores_masc = []
        scores_fem = []

        for pair in corpus:
            s_m = evaluator.predict_score(pair["text_masc"])
            s_f = evaluator.predict_score(pair["text_fem"])
            d = s_m - s_f

            scores_masc.append(s_m)
            scores_fem.append(s_f)
            deltas.append(d)

        arr_deltas = np.array(deltas)
        arr_masc = np.array(scores_masc)
        arr_fem = np.array(scores_fem)

        evaluator_deltas_dict[eval_key] = arr_deltas

        th = evaluator.provenance.threshold
        drift_res = compute_evaluator_drift_metrics(arr_deltas, arr_masc, arr_fem, threshold=th)
        tost_res = run_tost_equivalence_test(arr_deltas, bound_delta=protocol["equivalence_bound_delta"])
        tail_res = compute_tail_risk_metrics(arr_deltas, quantile_q=0.95)

        card = {
            "evaluator_id": evaluator.provenance.evaluator_id,
            "evaluator_name": evaluator.provenance.evaluator_name,
            "model_family": evaluator.provenance.model_family,
            "is_independent": evaluator.provenance.is_independent,
            "total_comparisons": len(corpus),
            "drift_metrics": drift_res,
            "equivalence_test": tost_res,
            "tail_risk": tail_res,
        }
        reliability_cards[eval_key] = card

    # Consensus Stability across independent evaluators under primary epsilon = 0.01
    agreement_res = compute_evaluator_consensus_stability(evaluator_deltas_dict, eps=protocol["primary_epsilon"])

    # Write Reliability Cards Markdown Report
    cards_report_path = os.path.join(out_dir, "evaluator_reliability_cards.md")
    lines = [
        "# ER-2: Evaluator Reliability Cards (Full LABE Test Counterfactual Benchmark, N=1,492 Pairs)\n",
        "## Evaluator Benchmark Summary Table\n",
        "| Evaluator Instrument | Independent? | MASD (95% BCa CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail Risk | TOST Equivalence? |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for k, card in reliability_cards.items():
        dm = card["drift_metrics"]
        tr = card["tail_risk"]
        eq = card["equivalence_test"]
        ind_str = "Yes" if card["is_independent"] else "No"
        ci_str = f"{dm['masd_mean_absolute_score_difference']:.4f} [{dm['masd_bca_ci_95'][0]:.4f}, {dm['masd_bca_ci_95'][1]:.4f}]"
        tost_str = "PASSED" if eq["tost_equivalence_passed"] else "FAILED"
        lines.append(
            f"| **{card['evaluator_name']}** | `{ind_str}` | `{ci_str}` | `{dm['cfr_counterfactual_flip_rate']*100:.2f}%` | `{dm['msd_mean_signed_drift']:+.4f}` | `{tr['cvar_95_tail_risk']:.4f}` | `{tost_str}` |"
        )

    lines.extend([
        "\n## Independent Evaluator Consensus & Agreement\n",
        f"- **Primary Epsilon (\\\\epsilon)**: `{agreement_res['epsilon_threshold']}`",
        f"- **Mean Consensus Stability**: `{agreement_res['mean_consensus_stability']:.4f}`",

        f"- **All-Evaluator Agreement Rate**: `{agreement_res['all_evaluator_agreement_rate']*100:.2f}%`",
        f"- **Majority Agreement Rate**: `{agreement_res['majority_agreement_rate']*100:.2f}%`",
        f"- **Opposite-Direction Disagreement Rate**: `{agreement_res['opposite_direction_disagreement_rate']*100:.2f}%`",
    ])

    with open(cards_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    master_manifest = {
        "status": "EMPIRICAL_BENCHMARK_COMPLETED",
        "protocol": protocol,
        "evaluators_count": len(panel),
        "independent_evaluators_count": sum(1 for e in panel.values() if e.provenance.is_independent),
        "total_counterfactual_pairs": len(corpus),
        "reliability_cards": reliability_cards,
        "consensus_agreement": agreement_res,
    }

    manifest_path = os.path.join(out_dir, "empirical_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    print(f"\nEmpirical Pipeline Complete! Master Manifest: {manifest_path}")
    return master_manifest


if __name__ == "__main__":
    run_empirical_audit_reliability_pipeline()
