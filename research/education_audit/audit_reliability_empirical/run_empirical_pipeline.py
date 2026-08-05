"""Master Empirical Pipeline Runner for ER-1R, ER-2R, and ER-2S Repair Milestone."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
import numpy as np

from research.education_audit.audit_reliability_empirical.protocol import get_preregistered_protocol
from research.education_audit.audit_reliability_empirical.evaluators.panel import initialize_empirical_evaluator_panel
from research.education_audit.audit_reliability_empirical.counterfactuals.pair_builder import build_labe_test_counterfactual_corpus
from research.education_audit.audit_reliability_empirical.metrics.drift import compute_cluster_aware_drift_metrics
from research.education_audit.audit_reliability_empirical.metrics.agreement import compute_substantive_evaluator_consensus
from research.education_audit.audit_reliability_empirical.metrics.equivalence import run_tost_equivalence_test
from research.education_audit.audit_reliability_empirical.metrics.tail_risk import compute_tail_risk_metrics


def run_empirical_audit_reliability_pipeline(
    out_dir: str = "results/education_audit/audit_reliability_empirical",
) -> Dict[str, Any]:
    """Runs ER-1R, ER-2R, and ER-2S repair pipeline and generates master manifest and reliability cards."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Step 1: Validating Preregistered Protocol & Claim Gates (ER-0)...")
    protocol = get_preregistered_protocol()

    print("Step 2: Initializing Frozen 3-Evaluator Independent Panel (ER-1R)...")
    panel_res = initialize_empirical_evaluator_panel(out_dir=out_dir)
    panel = panel_res["panel"]

    print("Step 3: Building Separated Natural Substitutions & Controlled Injection Corpora (ER-2R)...")
    corpora_res = build_labe_test_counterfactual_corpus()
    natural_corpus = corpora_res["natural_corpus"]
    injection_corpus = corpora_res["injection_corpus"]
    rejected_log = corpora_res["rejected_log"]

    # Write rejection log
    with open(os.path.join(out_dir, "pair_rejection_log.json"), "w", encoding="utf-8") as f:
        json.dump(rejected_log, f, indent=2)

    reliability_cards = {}
    substantive_deltas_dict = {}

    for corpus_name, corpus_data in [("natural_substitutions", natural_corpus), ("controlled_injection", injection_corpus)]:
        print(f"\nEvaluating Benchmark Corpus: {corpus_name} ({len(corpus_data)} pairs)...")
        corpus_cards = {}
        corpus_eval_deltas = {}

        for eval_key, evaluator in panel.items():
            print(f"  - Instrument: {evaluator.provenance.evaluator_name}")
            drift_res = compute_cluster_aware_drift_metrics(corpus_data, evaluator)
            arr_deltas = drift_res["raw_deltas"]
            corpus_eval_deltas[eval_key] = arr_deltas

            tost_res = run_tost_equivalence_test(arr_deltas, evaluator_type=eval_key)
            tail_res = compute_tail_risk_metrics(arr_deltas, quantile_q=0.95)

            card = {
                "evaluator_id": evaluator.provenance.evaluator_id,
                "evaluator_name": evaluator.provenance.evaluator_name,
                "model_family": evaluator.provenance.model_family,
                "checkpoint_sha256": evaluator.provenance.checkpoint_sha256,
                "is_independent": evaluator.provenance.is_independent,
                "corpus_name": corpus_name,
                "drift_metrics": {k: v for k, v in drift_res.items() if k != "raw_deltas"},
                "equivalence_test": tost_res,
                "tail_risk": tail_res,
            }
            corpus_cards[eval_key] = card

        # Consensus stability (excluding exact lexicon control)
        substantive_consensus = compute_substantive_evaluator_consensus(corpus_eval_deltas, eps=protocol["primary_epsilon"])

        reliability_cards[corpus_name] = {
            "evaluator_cards": corpus_cards,
            "substantive_consensus": substantive_consensus,
        }

    cards_report_path = os.path.join(out_dir, "evaluator_reliability_cards.md")
    lines = [
        "# ER-2R & ER-2S: Evaluator Reliability Cards (Cluster-Aware Inference, N=373 Sentence Clusters)\n",
    ]

    for corpus_name, c_info in reliability_cards.items():
        lines.append(f"## Benchmark Corpus: `{corpus_name}`\n")
        lines.append("| Evaluator Instrument | Independent? | MASD (95% Cluster BCa CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail | TOST Mean Equivalence |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for k, card in c_info["evaluator_cards"].items():
            dm = card["drift_metrics"]
            tr = card["tail_risk"]
            eq = card["equivalence_test"]
            ind_str = "Yes" if card["is_independent"] else "No"
            ci_str = f"{dm['masd_mean_absolute_score_difference']:.4f} [{dm['masd_cluster_bca_ci_95'][0]:.4f}, {dm['masd_cluster_bca_ci_95'][1]:.4f}]"
            lines.append(
                f"| **{card['evaluator_name']}** | `{ind_str}` | `{ci_str}` | `{dm['cfr_counterfactual_flip_rate']*100:.2f}%` | `{dm['msd_mean_signed_drift']:+.4f}` | `{tr['cvar_95_tail_risk']:.4f}` | `{eq['status_label']}` |"
            )

        sc = c_info["substantive_consensus"]
        lines.extend([
            "\n### Substantive 2-Evaluator Agreement (Excludes Deterministic Negative Control)\n",
            f"- **Substantive Evaluators**: `{', '.join(sc['substantive_evaluators'])}`",
            f"- **Mean Substantive Consensus Stability**: `{sc['mean_substantive_consensus_stability']:.4f}`",
            f"- **Exact Category Agreement Rate**: `{sc['exact_category_agreement_rate']*100:.2f}%`",
            f"- **Opposite-Direction Disagreement Rate**: `{sc['opposite_direction_disagreement_rate']*100:.2f}%`\n",
        ])

    with open(cards_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


    master_manifest = {
        "status": "EMPIRICAL_BENCHMARK_REPAIR_COMPLETED",
        "protocol": protocol,
        "evaluators_count": len(panel),
        "independent_evaluators_count": sum(1 for e in panel.values() if e.provenance.is_independent),
        "natural_pairs_count": len(natural_corpus),
        "injection_pairs_count": len(injection_corpus),
        "reliability_cards": reliability_cards,
    }

    manifest_path = os.path.join(out_dir, "empirical_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    print(f"\nEmpirical Repair Pipeline Complete! Master Manifest: {manifest_path}")
    return master_manifest


if __name__ == "__main__":
    run_empirical_audit_reliability_pipeline()
