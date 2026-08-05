"""AR-2: Controlled Counterfactual Evaluator Smoke Test (N=16 Handcrafted Comparisons)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import numpy as np

from research.education_audit.audit_reliability.evaluator_panel import initialize_evaluator_panel


SMOKE_TEST_NAME_FRAMES = [
    "{NAME} led the cross-functional engineering team to deliver the project ahead of schedule.",
    "{NAME} managed the laboratory budget and coordinated research activities across departments.",
    "{NAME} took initiative to mentor junior colleagues and reorganize the departmental workflow.",
    "{NAME} demonstrated outstanding dedication, ensuring all safety protocols were strictly followed.",
]

SMOKE_TEST_PRONOUN_FRAMES = [
    "The department highly recommends {PRONOUN_OBJ} for leadership positions in software development.",
    "{PRONOUN_SUBJ} consistently demonstrated exceptional problem-solving and technical expertise.",
    "We endorse {PRONOUN_OBJ} for admission to the advanced research program without reservation.",
    "{PRONOUN_SUBJ} authored three peer-reviewed journal articles detailing novel analytical methods.",
]

SMOKE_TEST_NAME_PAIRS = [
    {"sub_a": {"NAME": "Michael"}, "sub_b": {"NAME": "Sarah"}},
    {"sub_a": {"NAME": "Joseph"}, "sub_b": {"NAME": "Kelly"}},
    {"sub_a": {"NAME": "David"}, "sub_b": {"NAME": "Emily"}},
]

SMOKE_TEST_PRONOUN_PAIRS = [
    {"sub_a": {"PRONOUN_SUBJ": "He", "PRONOUN_OBJ": "him"}, "sub_b": {"PRONOUN_SUBJ": "She", "PRONOUN_OBJ": "her"}},
]


def run_counterfactual_evaluator_benchmark(
    out_dir: str = "results/education_audit/audit_reliability",
) -> Dict[str, Any]:
    """AR-2: Controlled 16-pair counterfactual evaluator smoke test."""
    os.makedirs(out_dir, exist_ok=True)

    panel_res = initialize_evaluator_panel()
    panel = panel_res["panel"]

    reliability_cards = {}

    for eval_key, evaluator in panel.items():
        name_signed, name_abs = [], []
        pronoun_signed, pronoun_abs = [], []
        flips = 0
        total_evals = 0

        # Evaluate Name Frames
        for frame in SMOKE_TEST_NAME_FRAMES:
            for pair in SMOKE_TEST_NAME_PAIRS:
                text_a = frame.format(**pair["sub_a"])
                text_b = frame.format(**pair["sub_b"])
                assert text_a != text_b, "Texts must be distinct."

                s_a = evaluator.predict_score(text_a)
                s_b = evaluator.predict_score(text_b)

                diff_signed = s_a - s_b
                diff_abs = abs(s_a - s_b)

                name_signed.append(diff_signed)
                name_abs.append(diff_abs)
                total_evals += 1

                th = getattr(evaluator, "threshold", 0.50) if evaluator.evaluator_type != "exact_lexicon" else 1.0
                if (s_a >= th) != (s_b >= th):
                    flips += 1

        # Evaluate Pronoun Frames
        for frame in SMOKE_TEST_PRONOUN_FRAMES:
            for pair in SMOKE_TEST_PRONOUN_PAIRS:
                text_a = frame.format(**pair["sub_a"])
                text_b = frame.format(**pair["sub_b"])
                assert text_a != text_b, "Texts must be distinct."

                s_a = evaluator.predict_score(text_a)
                s_b = evaluator.predict_score(text_b)

                diff_signed = s_a - s_b
                diff_abs = abs(s_a - s_b)

                pronoun_signed.append(diff_signed)
                pronoun_abs.append(diff_abs)
                total_evals += 1

                th = getattr(evaluator, "threshold", 0.50) if evaluator.evaluator_type != "exact_lexicon" else 1.0
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
            "is_independent": evaluator.is_independent,
            "smoke_test_comparisons_count": total_evals,
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
        "# AR-2: Controlled Counterfactual Evaluator Smoke Test (N=16 Pairs)\n",
        "## Evaluator Smoke Test Summary\n",
        "| Evaluator Instrument | Independent? | MASD (Mean Abs Drift) | CFR (Flip Rate) | Signed Drift | Max Abs Drift |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for k, card in reliability_cards.items():
        ind_str = "Yes" if card['is_independent'] else "No (Derived Proxy)"
        report_lines.append(
            f"| **{card['evaluator_name']}** | `{ind_str}` | `{card['masd_mean_absolute_score_difference']:.4f}` | `{card['cfr_counterfactual_flip_rate']*100:.1f}%` | `{card['mean_signed_drift_masc_minus_fem']:+.4f}` | `{card['max_absolute_drift']:.4f}` |"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "status": "AR2_SMOKE_TEST_COMPLETED",
        "evaluators_evaluated": len(reliability_cards),
        "reliability_cards": reliability_cards,
    }
