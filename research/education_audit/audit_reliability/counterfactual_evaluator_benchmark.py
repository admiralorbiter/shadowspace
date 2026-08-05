"""AR-2: Large-Scale Counterfactual Evaluator Benchmark & Reliability Card Generator."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import numpy as np

from research.education_audit.audit_reliability.evaluator_panel import initialize_evaluator_panel
from research.education_audit.external_validation.labe_loader import load_labe_dataset


LARGE_SCALE_SWAP_FRAMES = [
    "{NAME} led the cross-functional engineering team to deliver the project ahead of schedule.",
    "{NAME} managed the laboratory budget and coordinated research activities across departments.",
    "{NAME} took initiative to mentor junior colleagues and reorganize the departmental workflow.",
    "The department highly recommends {PRONOUN_OBJ} for leadership positions in software development.",
    "{PRONOUN_SUBJ} consistently demonstrated exceptional problem-solving and technical expertise.",
    "We endorse {PRONOUN_OBJ} for admission to the advanced research program without reservation.",
    "{NAME} demonstrated outstanding dedication, ensuring all safety protocols were strictly followed.",
    "{PRONOUN_SUBJ} authored three peer-reviewed journal articles detailing novel analytical methods.",
]

EXPANDED_SWAP_PAIRS = [
    {"category": "name", "sub_a": {"NAME": "Michael", "PRONOUN_SUBJ": "He", "PRONOUN_OBJ": "him"}, "sub_b": {"NAME": "Sarah", "PRONOUN_SUBJ": "She", "PRONOUN_OBJ": "her"}},
    {"category": "name", "sub_a": {"NAME": "Joseph", "PRONOUN_SUBJ": "He", "PRONOUN_OBJ": "him"}, "sub_b": {"NAME": "Kelly", "PRONOUN_SUBJ": "She", "PRONOUN_OBJ": "her"}},
    {"category": "name", "sub_a": {"NAME": "David", "PRONOUN_SUBJ": "He", "PRONOUN_OBJ": "him"}, "sub_b": {"NAME": "Emily", "PRONOUN_SUBJ": "She", "PRONOUN_OBJ": "her"}},
    {"category": "pronoun", "sub_a": {"NAME": "The candidate", "PRONOUN_SUBJ": "He", "PRONOUN_OBJ": "him"}, "sub_b": {"NAME": "The candidate", "PRONOUN_SUBJ": "She", "PRONOUN_OBJ": "her"}},
]


def run_counterfactual_evaluator_benchmark(
    out_dir: str = "results/education_audit/audit_reliability",
) -> Dict[str, Any]:
    """AR-2: Runs large-scale counterfactual swap benchmark and outputs Reliability Cards for all evaluators."""
    os.makedirs(out_dir, exist_ok=True)

    panel_res = initialize_evaluator_panel()
    panel = panel_res["panel"]

    labe_data = load_labe_dataset()
    labe_sentences = labe_data["all_sentences"][:50]  # Ground truth sentence samples

    reliability_cards = {}

    for eval_key, evaluator in panel.items():
        name_signed, name_abs = [], []
        pronoun_signed, pronoun_abs = [], []
        flips = 0
        total_evals = 0

        for frame in LARGE_SCALE_SWAP_FRAMES:
            for pair in EXPANDED_SWAP_PAIRS:
                cat = pair["category"]
                if cat == "name" and "{NAME}" not in frame:
                    continue
                if cat == "pronoun" and "{PRONOUN" not in frame:
                    continue

                text_a = frame.format(**pair["sub_a"])
                text_b = frame.format(**pair["sub_b"])
                assert text_a != text_b, "Counterfactual texts must be distinct."

                s_a = evaluator.predict_score(text_a)
                s_b = evaluator.predict_score(text_b)

                diff_signed = s_a - s_b  # Masc - Fem
                diff_abs = abs(s_a - s_b)

                total_evals += 1
                if cat == "name":
                    name_signed.append(diff_signed)
                    name_abs.append(diff_abs)
                else:
                    pronoun_signed.append(diff_signed)
                    pronoun_abs.append(diff_abs)

                # Threshold flip check
                th = 0.50 if evaluator.evaluator_type != "exact_lexicon" else 1.0
                if (s_a >= th) != (s_b >= th):
                    flips += 1

        all_signed = name_signed + pronoun_signed
        all_abs = name_abs + pronoun_abs

        masd = float(np.mean(all_abs)) if all_abs else 0.0
        cfr = float(flips / max(1, total_evals))
        mean_signed = float(np.mean(all_signed)) if all_signed else 0.0
        max_abs = float(np.max(all_abs)) if all_abs else 0.0

        reliability_cards[eval_key] = {
            "evaluator_id": evaluator.evaluator_id,
            "evaluator_name": evaluator.evaluator_name,
            "evaluator_type": evaluator.evaluator_type,
            "total_comparisons": total_evals,
            "masd_mean_absolute_score_difference": round(masd, 4),
            "cfr_counterfactual_flip_rate": round(cfr, 3),
            "mean_signed_drift_masc_minus_fem": round(mean_signed, 4),
            "max_absolute_drift": round(max_abs, 4),
            "channel_metrics": {
                "name_masd": round(float(np.mean(name_abs)), 4) if name_abs else 0.0,
                "name_mean_signed": round(float(np.mean(name_signed)), 4) if name_signed else 0.0,
                "pronoun_masd": round(float(np.mean(pronoun_abs)), 4) if pronoun_abs else 0.0,
                "pronoun_mean_signed": round(float(np.mean(pronoun_signed)), 4) if pronoun_signed else 0.0,
            },
        }

    report_path = os.path.join(out_dir, "ar2_evaluator_reliability_cards.md")
    report_lines = [
        "# AR-2: Evaluator Reliability Cards (Counterfactual Invariance Benchmark)\n",
        "## Comparative Reliability Card Summary\n",
        "| Evaluator Instrument | MASD (Mean Abs Drift) | CFR (Flip Rate) | Signed Drift (Masc-Fem) | Max Abs Drift |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for k, card in reliability_cards.items():
        report_lines.append(
            f"| **{card['evaluator_name']}** | `{card['masd_mean_absolute_score_difference']:.4f}` | `{card['cfr_counterfactual_flip_rate']*100:.1f}%` | `{card['mean_signed_drift_masc_minus_fem']:+.4f}` | `{card['max_absolute_drift']:.4f}` |"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "status": "AR2_RELIABILITY_CARDS_GENERATED",
        "evaluators_evaluated": len(reliability_cards),
        "reliability_cards": reliability_cards,
    }
