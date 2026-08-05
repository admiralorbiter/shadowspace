"""AR-3: Cross-Context Score Agreement Contrast Module."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import numpy as np
from scipy import stats

from research.education_audit.audit_reliability.evaluator_panel import initialize_evaluator_panel
from research.education_audit.external_validation.wan2023_loader import load_wan2023_dataset
from research.education_audit.external_validation.labe_loader import load_labe_dataset


def run_cross_domain_transfer_benchmark(
    out_dir: str = "results/education_audit/audit_reliability",
) -> Dict[str, Any]:
    """AR-3: Cross-context score agreement contrast between single LABE sentences and Wan recommendation letters."""
    os.makedirs(out_dir, exist_ok=True)

    panel_res = initialize_evaluator_panel()
    panel = panel_res["panel"]

    lex_eval = panel["exact_lexicon"]
    ngram_eval = panel["sparse_ngram_ensemble"]

    wan_data = load_wan2023_dataset()
    wan_records = wan_data["records"]

    paired_groups: Dict[tuple, Dict[str, str]] = {}
    for r in wan_records:
        key = (r["age"], r["occupation"])
        if key not in paired_groups:
            paired_groups[key] = {}
        paired_groups[key][r["name"].lower()] = r["generated_text"]

    lex_wan_deltas = []
    ngram_wan_deltas = []

    for (age, occu), names_dict in paired_groups.items():
        if "joseph" in names_dict and "kelly" in names_dict:
            text_m = names_dict["joseph"]
            text_f = names_dict["kelly"]

            d_lex = lex_eval.predict_score(text_m) - lex_eval.predict_score(text_f)
            d_ngram = ngram_eval.predict_score(text_m) - ngram_eval.predict_score(text_f)

            lex_wan_deltas.append(d_lex)
            ngram_wan_deltas.append(d_ngram)

    r_wan, p_wan = stats.pearsonr(lex_wan_deltas, ngram_wan_deltas)
    rho_wan, p_rho_wan = stats.spearmanr(lex_wan_deltas, ngram_wan_deltas)

    # In-domain LABE sentences cross-evaluator comparison
    labe_data = load_labe_dataset()
    labe_sentences = labe_data["all_sentences"][:100]

    lex_labe_scores = [lex_eval.predict_score(s["text"]) for s in labe_sentences]
    ngram_labe_scores = [ngram_eval.predict_score(s["text"]) for s in labe_sentences]

    r_labe, p_labe = stats.pearsonr(lex_labe_scores, ngram_labe_scores)
    rho_labe, p_rho_labe = stats.spearmanr(lex_labe_scores, ngram_labe_scores)

    report = {
        "status": "AR3_CROSS_CONTEXT_CONTRAST_COMPLETED",
        "labe_sentences_score_correlation": {
            "sample_count": 100,
            "pearson_r": round(float(r_labe), 3),
            "spearman_rho": round(float(rho_labe), 3),
        },
        "wan_letters_delta_correlation": {
            "sample_count": 60,
            "pearson_r": round(float(r_wan), 3),
            "spearman_rho": round(float(rho_wan), 3),
        },
        "correlation_contrast_delta_r": round(float(r_labe - r_wan), 3),
    }

    report_path = os.path.join(out_dir, "ar3_cross_domain_transfer.md")
    report_lines = [
        "# AR-3: Cross-Context Score Agreement Contrast Report\n",
        f"- **LABE Sentences Score Correlation (N=100)**: Pearson r = `{r_labe:+.3f}`, Spearman rho = `{rho_labe:+.3f}`",
        f"- **Wan Letter Delta Correlation (N=60)**: Pearson r = `{r_wan:+.3f}`, Spearman rho = `{rho_wan:+.3f}`",
        f"- **Correlation Contrast Delta (r_sentences - r_letter_deltas)**: `{r_labe - r_wan:+.3f}`\n",
        "## Descriptive Observation\n",
        f"Lexicon-to-n-gram correlation on isolated LABE sentences was **{r_labe:+.3f}**, whereas correlation between counterfactual deltas on full Wan letters was **{r_wan:+.3f}** (contrast delta = `{r_labe - r_wan:+.3f}`).",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
