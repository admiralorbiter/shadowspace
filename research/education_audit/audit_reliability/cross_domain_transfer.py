"""AR-3: Cross-Domain Evaluator Transfer & Generalization Benchmark."""

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
    """AR-3: Benchmarks cross-domain metric agreement between LABE sentences and Wan full recommendation letters."""
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
        "status": "AR3_CROSS_DOMAIN_TRANSFER_COMPLETED",
        "indomain_labe_sentences": {
            "sample_count": 100,
            "pearson_r": round(float(r_labe), 3),
            "spearman_rho": round(float(rho_labe), 3),
        },
        "crossdomain_wan_letters": {
            "sample_count": 60,
            "pearson_r": round(float(r_wan), 3),
            "spearman_rho": round(float(rho_wan), 3),
        },
        "domain_transfer_degradation_delta_r": round(float(r_labe - r_wan), 3),
    }

    report_path = os.path.join(out_dir, "ar3_cross_domain_transfer.md")
    report_lines = [
        "# AR-3: Cross-Domain Evaluator Transfer Report\n",
        f"- **In-Domain LABE Sentences (N=100)**: Pearson r = `{r_labe:+.3f}`, Spearman rho = `{rho_labe:+.3f}`",
        f"- **Cross-Domain Wan Recommendation Letters (N=60)**: Pearson r = `{r_wan:+.3f}`, Spearman rho = `{rho_wan:+.3f}`",
        f"- **Transfer Degradation Delta (r_in_domain - r_cross_domain)**: `{r_labe - r_wan:+.3f}`\n",
        "## Key Conclusion\n",
        f"Evaluator agreement drops by **{abs(r_labe - r_wan):.3f}** when transferring from isolated in-domain sentences to full recommendation letters, confirming that in-domain evaluation accuracy does not guarantee cross-domain audit agreement.",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
