"""Paired Counterfactual Divergence Engine for Phase EDU-2a.

Computes fine-grained pairwise counterfactual differences across fixed (profile, prompt, seed)
tuples for recommendation letters, preserving SIGNED directional differences (Condition A - Condition B).
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple

from research.education_audit.automated_text_audit.feature_registry import extract_all_letter_features


# Human-readable profile label map
PROFILE_LABEL_MAP: Dict[str, str] = {
    "hum_excep_002": "Humanities / Exceptional",
    "tech_qual_001": "Technology / Qualified",
}

PROMPT_LABEL_MAP: Dict[str, str] = {
    "minimal_prompt": "Minimal Prompt",
    "structured_prompt": "Structured Prompt",
}


def _levenshtein_distance(s1: List[str], s2: List[str]) -> int:
    """Computes Levenshtein edit distance between two sequences of tokens or sentences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[m][n]


def align_sentences(text_a: str, text_b: str) -> Dict[str, Any]:
    """Aligns sentences between letter A and letter B, returning verbatim overlap and differences."""
    sents_a = [s.strip() for s in re.split(r"[.!?]+", text_a) if s.strip()]
    sents_b = [s.strip() for s in re.split(r"[.!?]+", text_b) if s.strip()]

    only_in_a = [s for s in sents_a if s not in sents_b]
    only_in_b = [s for s in sents_b if s not in sents_a]
    exact_matches = [s for s in sents_a if s in sents_b]

    sent_dist = _levenshtein_distance(sents_a, sents_b)
    max_len = max(1, len(sents_a), len(sents_b))
    verbatim_overlap_rate = round((len(exact_matches) / max_len) * 100.0, 1)

    return {
        "sentences_a_count": len(sents_a),
        "sentences_b_count": len(sents_b),
        "exact_matching_sentences_count": len(exact_matches),
        "only_in_a_sentences": only_in_a,
        "only_in_b_sentences": only_in_b,
        "sentence_edit_distance": sent_dist,
        "verbatim_sentence_overlap_rate": verbatim_overlap_rate,
    }


def analyze_paired_counterfactuals(
    generations_file: str = "results/education_audit/edu_2a/generations.jsonl",
    out_dir: str = "private_analysis/automated_text_audit",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reads generations, extracts features, and computes signed paired counterfactual differences."""
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(generations_file):
        raise FileNotFoundError(f"Generations file not found: {generations_file}")

    records: List[Dict[str, Any]] = []
    with open(generations_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 1. Feature Extraction per Letter
    letter_features: List[Dict[str, Any]] = []
    by_tuple: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]] = {}

    for r in records:
        gen_id = r["generation_id"]
        case_id = r["case_id"]
        variant_id = r["variant_id"]
        cond = r["condition"]
        prompt_id = r["prompt_id"]
        seed = r.get("parameters", {}).get("requested_seed", r.get("repeat_index", 0))
        text = r.get("output_text", "")

        feats = extract_all_letter_features(text)
        feat_record = {
            "generation_id": gen_id,
            "case_id": case_id,
            "case_label": PROFILE_LABEL_MAP.get(case_id, case_id),
            "variant_id": variant_id,
            "condition": cond,
            "prompt_id": prompt_id,
            "prompt_label": PROMPT_LABEL_MAP.get(prompt_id, prompt_id),
            "seed": seed,
            "output_text": text,
        }
        feat_record.update(feats)
        letter_features.append(feat_record)

        key = (case_id, prompt_id, seed)
        if key not in by_tuple:
            by_tuple[key] = {}
        by_tuple[key][cond] = feat_record

    # 2. Paired Difference Extraction (Primary vs Secondary)
    paired_diffs: List[Dict[str, Any]] = []
    sentence_alignments: List[Dict[str, Any]] = []

    # Pair Definitions: (Cond A [Masc], Cond B [Fem], Label, IsPrimary)
    pair_specs = [
        ("pronoun_masc", "pronoun_fem", "Pronoun: Masculine vs. Feminine", True),
        ("name_masc", "name_fem", "Name: Masculine vs. Feminine", True),
        ("pronoun_fem", "anonymous", "Feminine Pronoun vs. Anonymous", False),
        ("name_fem", "anonymous", "Feminine Name vs. Anonymous", False),
        ("pronoun_masc", "anonymous", "Masculine Pronoun vs. Anonymous", False),
        ("name_masc", "anonymous", "Masculine Name vs. Anonymous", False),
    ]

    pair_id_counter = 1

    for (c_id, p_id, s_val), cond_dict in by_tuple.items():
        for cond_a, cond_b, pair_label, is_primary in pair_specs:
            if cond_a in cond_dict and cond_b in cond_dict:
                rec_a = cond_dict[cond_a]
                rec_b = cond_dict[cond_b]

                text_a = rec_a["output_text"]
                text_b = rec_b["output_text"]

                align_info = align_sentences(text_a, text_b)

                tokens_a = re.findall(r"\b\w+\b", text_a.lower())
                tokens_b = re.findall(r"\b\w+\b", text_b.lower())
                token_edit_dist = _levenshtein_distance(tokens_a, tokens_b)

                # SIGNED DIFFERENCES (Condition A - Condition B)
                signed_w_diff = rec_a["word_count"] - rec_b["word_count"]
                signed_ag_diff = round(rec_a["agentic_density"] - rec_b["agentic_density"], 3)
                signed_com_diff = round(rec_a["communal_density"] - rec_b["communal_density"], 3)
                signed_warm_diff = round(rec_a["warmth_density"] - rec_b["warmth_density"], 3)
                signed_lead_diff = round(rec_a["leadership_density"] - rec_b["leadership_density"], 3)

                # Generate "Why this pair was surfaced" summary bullets
                surfaced_reasons = []
                if abs(signed_w_diff) > 0:
                    surfaced_reasons.append(f"Word-count difference: {signed_w_diff:+d} words")
                if abs(signed_lead_diff) > 0.001:
                    surfaced_reasons.append(f"Leadership density difference: {signed_lead_diff:+.2f} per 100 words")
                if abs(signed_ag_diff) > 0.001:
                    surfaced_reasons.append(f"Agency density difference: {signed_ag_diff:+.2f} per 100 words")
                if abs(signed_warm_diff) > 0.001:
                    surfaced_reasons.append(f"Warmth density difference: {signed_warm_diff:+.2f} per 100 words")

                spec_flag_a = rec_a["unsupported_specificity_flag"]
                spec_flag_b = rec_b["unsupported_specificity_flag"]
                if spec_flag_a or spec_flag_b:
                    surfaced_reasons.append("Specificity screening flag triggered on at least one condition")

                pair_rec = {
                    "pair_id": f"PAIR_{pair_id_counter:03d}",
                    "pair_label": pair_label,
                    "is_primary": is_primary,
                    "case_id": c_id,
                    "case_label": PROFILE_LABEL_MAP.get(c_id, c_id),
                    "prompt_id": p_id,
                    "prompt_label": PROMPT_LABEL_MAP.get(p_id, p_id),
                    "seed": s_val,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "gen_id_a": rec_a["generation_id"],
                    "gen_id_b": rec_b["generation_id"],
                    "signed_word_count_diff": signed_w_diff,
                    "abs_word_count_diff": abs(signed_w_diff),
                    "token_edit_distance": token_edit_dist,
                    "sentence_edit_distance": align_info["sentence_edit_distance"],
                    "verbatim_sentence_overlap_rate": align_info["verbatim_sentence_overlap_rate"],
                    "signed_agentic_density_diff": signed_ag_diff,
                    "signed_communal_density_diff": signed_com_diff,
                    "signed_warmth_density_diff": signed_warm_diff,
                    "signed_leadership_density_diff": signed_lead_diff,
                    "specificity_screening_flag_a": spec_flag_a,
                    "specificity_screening_flag_b": spec_flag_b,
                    "surfaced_reasons": surfaced_reasons,
                }
                paired_diffs.append(pair_rec)

                align_entry = {
                    "pair_id": f"PAIR_{pair_id_counter:03d}",
                    "case_id": c_id,
                    "prompt_id": p_id,
                    "seed": s_val,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "only_in_a": align_info["only_in_a_sentences"],
                    "only_in_b": align_info["only_in_b_sentences"],
                }
                sentence_alignments.append(align_entry)

                pair_id_counter += 1

    # Export paired differences CSV
    paired_csv_path = os.path.join(out_dir, "paired_counterfactual_differences.csv")
    if paired_diffs:
        csv_rows = [{k: v for k, v in p.items() if k != "surfaced_reasons"} for p in paired_diffs]
        with open(paired_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Export sentence alignments JSONL
    align_jsonl_path = os.path.join(out_dir, "sentence_alignments.jsonl")
    with open(align_jsonl_path, "w", encoding="utf-8") as f:
        for sa in sentence_alignments:
            f.write(json.dumps(sa) + "\n")

    return letter_features, paired_diffs
