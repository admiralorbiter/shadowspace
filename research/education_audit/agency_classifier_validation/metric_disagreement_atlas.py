"""Phase ACV-2: Metric Disagreement Atlas (Exact Lexicon vs. Trained LABE N-Gram Baseline on Wan 2023 Pairs)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List
import numpy as np
from scipy import stats

from research.education_audit.automated_text_audit.feature_registry import extract_lexical_features
from research.education_audit.external_validation.wan2023_loader import load_wan2023_dataset


def _split_into_sentences(text: str) -> List[str]:
    """Splits letter text into clean sentences."""
    raw_sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sents if len(s.strip()) > 10]


def _classify_pair_disagreement(d_lex: float, d_clf: float, eps: float = 0.01) -> str:
    """Classifies pair into mutually exclusive disagreement categories under epsilon threshold."""
    lex_zero = abs(d_lex) <= eps
    clf_zero = abs(d_clf) <= eps

    if lex_zero and clf_zero:
        return "BOTH_ZERO"

    sign_lex = int(np.sign(d_lex)) if not lex_zero else 0
    sign_clf = int(np.sign(d_clf)) if not clf_zero else 0

    if sign_lex == sign_clf and sign_lex != 0:
        return "SAME_DIRECTION"
    elif lex_zero and not clf_zero:
        return "LEXICON_ZERO_CLASSIFIER_NONZERO"
    elif clf_zero and not lex_zero:
        return "CLASSIFIER_ZERO_LEXICON_NONZERO"
    elif sign_lex != 0 and sign_clf != 0 and sign_lex != sign_clf:
        return "OPPOSITE_DIRECTION"

    return "UNCLASSIFIED"


def build_metric_disagreement_atlas(
    model_artifacts: Dict[str, Any],
    out_dir: str = "results/education_audit/agency_classifier_validation",
) -> Dict[str, Any]:
    """Phase ACV-2: Constructs Metric Disagreement Atlas with a mutually exclusive taxonomy and epsilon sensitivity analysis."""
    os.makedirs(out_dir, exist_ok=True)

    vectorizer = model_artifacts["vectorizer"]
    clf_lr = model_artifacts["clf_lr"]
    clf_gb = model_artifacts["clf_gb"]

    wan_data = load_wan2023_dataset()
    wan_records = wan_data["records"]

    paired_groups: Dict[tuple, Dict[str, str]] = {}
    for r in wan_records:
        key = (r["age"], r["occupation"])
        if key not in paired_groups:
            paired_groups[key] = {}
        paired_groups[key][r["name"].lower()] = r["generated_text"]

    lex_deltas = []
    clf_deltas = []
    pair_records = []

    for (age, occu), names_dict in paired_groups.items():
        if "joseph" in names_dict and "kelly" in names_dict:
            text_m = names_dict["joseph"]
            text_f = names_dict["kelly"]

            feats_m = extract_lexical_features(text_m)
            feats_f = extract_lexical_features(text_f)
            d_lex = feats_m["agentic_density"] - feats_f["agentic_density"]

            sents_m = _split_into_sentences(text_m)
            sents_f = _split_into_sentences(text_f)

            vec_m = vectorizer.transform(sents_m)
            prob_m = 0.5 * clf_lr.predict_proba(vec_m)[:, 1] + 0.5 * clf_gb.predict_proba(vec_m)[:, 1]
            score_m = float(np.mean(prob_m)) if len(prob_m) > 0 else 0.5

            vec_f = vectorizer.transform(sents_f)
            prob_f = 0.5 * clf_lr.predict_proba(vec_f)[:, 1] + 0.5 * clf_gb.predict_proba(vec_f)[:, 1]
            score_f = float(np.mean(prob_f)) if len(prob_f) > 0 else 0.5

            d_clf = score_m - score_f

            lex_deltas.append(d_lex)
            clf_deltas.append(d_clf)

            pair_records.append({
                "age": age,
                "occupation": occu,
                "delta_lexicon": round(d_lex, 3),
                "delta_classifier": round(d_clf, 3),
            })

    lex_arr = np.array(lex_deltas)
    clf_arr = np.array(clf_deltas)

    pearson_r, p_val_pearson = stats.pearsonr(lex_arr, clf_arr)
    spearman_r, p_val_spearman = stats.spearmanr(lex_arr, clf_arr)

    # Epsilon sensitivity analysis across eps in [0.00, 0.01, 0.02, 0.05]
    sensitivity_table = {}
    primary_taxonomy_counts = {}

    for eps in [0.00, 0.01, 0.02, 0.05]:
        tax_counts = {
            "BOTH_ZERO": 0,
            "SAME_DIRECTION": 0,
            "LEXICON_ZERO_CLASSIFIER_NONZERO": 0,
            "CLASSIFIER_ZERO_LEXICON_NONZERO": 0,
            "OPPOSITE_DIRECTION": 0,
        }
        for rec in pair_records:
            cat = _classify_pair_disagreement(rec["delta_lexicon"], rec["delta_classifier"], eps=eps)
            tax_counts[cat] = tax_counts.get(cat, 0) + 1

        sign_agree = tax_counts["BOTH_ZERO"] + tax_counts["SAME_DIRECTION"]
        sensitivity_table[f"eps_{eps:.2f}"] = {
            "taxonomy": tax_counts,
            "sign_agreement_count": sign_agree,
            "sign_agreement_pct": round(sign_agree / len(pair_records), 3),
        }
        if eps == 0.01:
            primary_taxonomy_counts = tax_counts

    report = {
        "status": "ACV2_DISAGREEMENT_ATLAS_BUILT",
        "pairs_count": len(pair_records),
        "pearson_correlation": round(float(pearson_r), 3),
        "pearson_pvalue": round(float(p_val_pearson), 4),
        "spearman_correlation": round(float(spearman_r), 3),
        "spearman_pvalue": round(float(p_val_spearman), 4),
        "primary_epsilon": 0.01,
        "primary_taxonomy_counts": primary_taxonomy_counts,
        "epsilon_sensitivity_table": sensitivity_table,
    }

    report_path = os.path.join(out_dir, "acv2_disagreement_atlas.md")
    report_lines = [
        "# Phase ACV-2: Metric Disagreement Atlas Report (Mutually Exclusive Taxonomy)\n",
        f"- **Pairs Evaluated**: {len(pair_records)} Joseph vs. Kelly matched cells",
        f"- **Pearson Correlation (Signed Deltas)**: r = {pearson_r:+.3f} (p = {p_val_pearson:.4f})",
        f"- **Spearman Rank Correlation**: rho = {spearman_r:+.3f} (p = {p_val_spearman:.4f})\n",
        "## Mutually Exclusive Taxonomy Breakdown (Primary Epsilon = 0.01)\n",
        f"- **BOTH_ZERO**: {primary_taxonomy_counts['BOTH_ZERO']} / 60",
        f"- **SAME_DIRECTION**: {primary_taxonomy_counts['SAME_DIRECTION']} / 60",
        f"- **LEXICON_ZERO_CLASSIFIER_NONZERO**: {primary_taxonomy_counts['LEXICON_ZERO_CLASSIFIER_NONZERO']} / 60",
        f"- **CLASSIFIER_ZERO_LEXICON_NONZERO**: {primary_taxonomy_counts['CLASSIFIER_ZERO_LEXICON_NONZERO']} / 60",
        f"- **OPPOSITE_DIRECTION**: {primary_taxonomy_counts['OPPOSITE_DIRECTION']} / 60\n",
        "## Epsilon Sensitivity Analysis\n",
    ]
    for eps_key, res in sensitivity_table.items():
        report_lines.append(
            f"- **{eps_key}**: Sign Agreement = {res['sign_agreement_pct']*100:.1f}% ({res['sign_agreement_count']}/60) | Opposite Direction = {res['taxonomy']['OPPOSITE_DIRECTION']}/60"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
