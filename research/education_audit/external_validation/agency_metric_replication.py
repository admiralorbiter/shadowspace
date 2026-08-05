"""Milestone EV-1: Real External Benchmark Agency Metric Replication & Disagreement Atlas.

Runs exact-lexicon feature extraction and agency scoring against:
1. Real Wan et al. EMNLP 2023 published ChatGPT reference letters (Joseph vs. Kelly).
2. Real LABE 2023 Language Agency Classification (LAC) ground-truth labeled sentences.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import numpy as np
from scipy import stats

from research.education_audit.automated_text_audit.feature_registry import extract_lexical_features
from research.education_audit.external_validation.wan2023_loader import load_wan2023_dataset
from research.education_audit.external_validation.labe_loader import load_labe_dataset


def run_agency_metric_replication(out_dir: str = "results/education_audit/external_validation") -> Dict[str, Any]:
    """Runs Milestone EV-1 external metric replication analysis on REAL published datasets."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Ingest Real Wan et al. 2023 Dataset
    wan_data = load_wan2023_dataset()
    wan_records = wan_data["records"]

    # Pair Joseph (male) and Kelly (female) across matching age & occupation
    paired_groups: Dict[tuple, Dict[str, str]] = {}
    for r in wan_records:
        key = (r["age"], r["occupation"])
        if key not in paired_groups:
            paired_groups[key] = {}
        paired_groups[key][r["name"].lower()] = r["generated_text"]

    lex_deltas = []
    wan_pairs_evaluated = 0
    disagreements = []

    for (age, occu), names_dict in paired_groups.items():
        if "joseph" in names_dict and "kelly" in names_dict:
            wan_pairs_evaluated += 1
            text_m = names_dict["joseph"]
            text_f = names_dict["kelly"]

            feats_m = extract_lexical_features(text_m)
            feats_f = extract_lexical_features(text_f)

            delta_lex = feats_m["agentic_density"] - feats_f["agentic_density"]
            delta_lead = feats_m["leadership_density"] - feats_f["leadership_density"]
            delta_com = feats_m["communal_density"] - feats_f["communal_density"]

            lex_deltas.append(delta_lex)

            if (delta_lex > 0 and delta_com > 0) or (delta_lex < 0 and delta_com < 0):
                disagreements.append({
                    "age": age,
                    "occupation": occu,
                    "delta_agency": delta_lex,
                    "delta_leadership": delta_lead,
                    "delta_communal": delta_com,
                    "joseph_preview": text_m[:120] + "...",
                    "kelly_preview": text_f[:120] + "...",
                })

    mean_delta_agency = float(np.mean(lex_deltas)) if lex_deltas else 0.0

    # 2. Ingest Real LABE 2023 LAC Dataset & Calculate Lexicon Precision/Recall
    labe_data = load_labe_dataset()
    lac_sentences = labe_data["sentences"]

    tp, fp, tn, fn = 0, 0, 0, 0
    for s in lac_sentences:
        feats = extract_lexical_features(s["text"])
        # Lexicon classifies as agentic if agentic_count > 0 or leadership_count > 0
        pred_agentic = (feats["agentic_count"] + feats["leadership_count"]) > 0
        true_agentic = (s["label_int"] == 1)

        if pred_agentic and true_agentic:
            tp += 1
        elif pred_agentic and not true_agentic:
            fp += 1
        elif not pred_agentic and not true_agentic:
            tn += 1
        else:
            fn += 1

    precision = round(tp / max(1, tp + fp), 3)
    recall = round(tp / max(1, tp + fn), 3)
    f1_score = round(2 * precision * recall / max(0.001, precision + recall), 3)

    report = {
        "status": "EV1_REPL_COMPLETED_REAL_DATA",
        "wan2023_sha256_hash": wan_data["sha256_hash"],
        "labe_train_sha256_hash": labe_data["train_sha256_hash"],
        "wan_pairs_evaluated": wan_pairs_evaluated,
        "mean_wan_agency_delta_masc_minus_fem": round(mean_delta_agency, 3),
        "labe_lac_sentences_evaluated": len(lac_sentences),
        "lexicon_precision_on_labe_lac": precision,
        "lexicon_recall_on_labe_lac": recall,
        "lexicon_f1_score_on_labe_lac": f1_score,
        "disagreements_count": len(disagreements),
        "disagreement_samples": disagreements[:5],
    }

    report_path = os.path.join(out_dir, "replication_report.md")
    report_lines = [
        "# Milestone EV-1: Real External Benchmark Agency Metric Replication Report\n",
        f"- **Wan 2023 Source Hash**: `{wan_data['sha256_hash']}`",
        f"- **LABE LAC Source Hash**: `{labe_data['train_sha256_hash']}`",
        f"- **Real Wan 2023 ChatGPT Pairs Evaluated**: {wan_pairs_evaluated}",
        f"- **Mean Wan Agency Delta (Joseph - Kelly)**: {mean_delta_agency:+.3f} per 100 words",
        f"- **Real LABE LAC Sentences Evaluated**: {len(lac_sentences)}",
        f"- **Lexicon Precision on LABE LAC**: {precision * 100:.1f}%",
        f"- **Lexicon Recall on LABE LAC**: {recall * 100:.1f}%",
        f"- **Lexicon F1 Score on LABE LAC**: {f1_score:.3f}\n",
        "## Empirical Metric Findings on Real Published Datasets\n",
        "1. **Real ChatGPT Letters (Wan et al. 2023)**: Evaluated exact-dictionary agency deltas across published ChatGPT letters for matched Joseph and Kelly candidates.",
        "2. **Real Sentence Agency Classification (LABE 2023)**: Benchmark exact-keyword agency density against 1,093+ human-labeled ground-truth sentences from the NAACL 2024 LABE benchmark.",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
