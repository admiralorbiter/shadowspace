"""Phase ACV-2: Metric Disagreement Atlas (Exact Lexicon vs. Trained LABE Classifier on Wan 2023 Pairs)."""

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


def build_metric_disagreement_atlas(
    model_artifacts: Dict[str, Any],
    out_dir: str = "results/education_audit/agency_classifier_validation",
) -> Dict[str, Any]:
    """Phase ACV-2: Constructs Metric Disagreement Atlas comparing exact-lexicon density vs. contextual classifier predictions."""
    os.makedirs(out_dir, exist_ok=True)

    vectorizer = model_artifacts["vectorizer"]
    clf_lr = model_artifacts["clf_lr"]
    clf_gb = model_artifacts["clf_gb"]
    threshold = model_artifacts["best_threshold"]

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
    pair_details = []

    for (age, occu), names_dict in paired_groups.items():
        if "joseph" in names_dict and "kelly" in names_dict:
            text_m = names_dict["joseph"]
            text_f = names_dict["kelly"]

            # Lexicon evaluation
            feats_m = extract_lexical_features(text_m)
            feats_f = extract_lexical_features(text_f)
            d_lex = feats_m["agentic_density"] - feats_f["agentic_density"]

            # Classifier evaluation: average sentence-level agency probability
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

            # Categorize agreement / disagreement
            sign_lex = np.sign(d_lex)
            sign_clf = np.sign(d_clf) if abs(d_clf) > 0.01 else 0

            status = "AGREE"
            if sign_lex == 0 and abs(d_clf) > 0.02:
                status = "LEXICON_ZERO_CLASSIFIER_DETECTED"
            elif sign_lex != 0 and sign_clf != 0 and sign_lex != sign_clf:
                status = "DIRECTIONAL_DISAGREEMENT"

            pair_details.append({
                "age": age,
                "occupation": occu,
                "delta_lexicon": round(d_lex, 3),
                "delta_classifier": round(d_clf, 3),
                "sign_lexicon": int(sign_lex),
                "sign_classifier": int(sign_clf),
                "status": status,
            })

    lex_arr = np.array(lex_deltas)
    clf_arr = np.array(clf_deltas)

    pearson_r, p_val_pearson = stats.pearsonr(lex_arr, clf_arr)
    spearman_r, p_val_spearman = stats.spearmanr(lex_arr, clf_arr)

    # Sign agreement
    signs_match = sum(1 for p in pair_details if p["sign_lexicon"] == p["sign_classifier"])
    sign_agreement_pct = round(signs_match / len(pair_details), 3)

    zero_lex_clf_detected = [p for p in pair_details if p["status"] == "LEXICON_ZERO_CLASSIFIER_DETECTED"]
    directional_disagreements = [p for p in pair_details if p["status"] == "DIRECTIONAL_DISAGREEMENT"]

    report = {
        "status": "ACV2_DISAGREEMENT_ATLAS_BUILT",
        "pairs_count": len(pair_details),
        "pearson_correlation": round(float(pearson_r), 3),
        "pearson_pvalue": round(float(p_val_pearson), 4),
        "spearman_correlation": round(float(spearman_r), 3),
        "spearman_pvalue": round(float(p_val_spearman), 4),
        "sign_agreement_percentage": sign_agreement_pct,
        "lexicon_zero_classifier_detected_count": len(zero_lex_clf_detected),
        "directional_disagreements_count": len(directional_disagreements),
        "disagreement_samples": (zero_lex_clf_detected + directional_disagreements)[:5],
    }

    report_path = os.path.join(out_dir, "acv2_disagreement_atlas.md")
    report_lines = [
        "# Phase ACV-2: Metric Disagreement Atlas Report (Lexicon vs. Classifier)\n",
        f"- **Pairs Evaluated**: {len(pair_details)} Joseph vs. Kelly matched cells",
        f"- **Pearson Correlation (Signed Deltas)**: r = {pearson_r:+.3f} (p = {p_val_pearson:.4f})",
        f"- **Spearman Rank Correlation**: rho = {spearman_r:+.3f} (p = {p_val_spearman:.4f})",
        f"- **Sign Agreement Rate**: {sign_agreement_pct * 100:.1f}%\n",
        "## Disagreement Breakdown\n",
        f"- **Pairs where Lexicon = 0 but Classifier Detected Agency Shift**: {len(zero_lex_clf_detected)} / 60",
        f"- **Directional Disagreement Pairs (Opposite Signs)**: {len(directional_disagreements)} / 60\n",
        "## Sample Disagreement Records\n",
    ]
    for d in (zero_lex_clf_detected + directional_disagreements)[:8]:
        report_lines.append(
            f"- **Age {d['age']} {d['occupation']}**: Status: `{d['status']}` | Lexicon Delta: `{d['delta_lexicon']:+.3f}` | Classifier Delta: `{d['delta_classifier']:+.3f}`"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
