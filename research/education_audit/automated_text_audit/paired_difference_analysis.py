"""Paired Counterfactual Divergence Engine for Phase EDU-2a.

Computes fine-grained pairwise counterfactual differences across fixed (profile, prompt, seed)
tuples for the 60 frozen Gemma recommendation letters.

Safeguard: Exports outputs to private_analysis/automated_text_audit/ to prevent anchoring
human reviewers prior to manual review closure.
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple

from research.education_audit.automated_text_audit.feature_registry import extract_all_letter_features


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
    """Aligns sentences between letter A and letter B, returning additions, omissions, and similarity."""
    sents_a = [s.strip() for s in re.split(r"[.!?]+", text_a) if s.strip()]
    sents_b = [s.strip() for s in re.split(r"[.!?]+", text_b) if s.strip()]

    only_in_a = [s for s in sents_a if s not in sents_b]
    only_in_b = [s for s in sents_b if s not in sents_a]
    exact_matches = [s for s in sents_a if s in sents_b]

    sent_dist = _levenshtein_distance(sents_a, sents_b)
    max_len = max(1, max(len(sents_a), len(sents_b)))
    alignment_similarity = round(1.0 - (sent_dist / max_len), 3)

    return {
        "sentences_a_count": len(sents_a),
        "sentences_b_count": len(sents_b),
        "exact_matching_sentences_count": len(exact_matches),
        "only_in_a_sentences": only_in_a,
        "only_in_b_sentences": only_in_b,
        "sentence_edit_distance": sent_dist,
        "alignment_similarity": alignment_similarity,
    }


def analyze_paired_counterfactuals(
    generations_file: str = "results/education_audit/edu_2a/generations.jsonl",
    out_dir: str = "private_analysis/automated_text_audit",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reads 60 frozen Gemma generations, extracts features, and computes paired differences."""
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
            "variant_id": variant_id,
            "condition": cond,
            "prompt_id": prompt_id,
            "seed": seed,
            "output_text": text,
        }
        feat_record.update(feats)
        letter_features.append(feat_record)

        key = (case_id, prompt_id, seed)
        if key not in by_tuple:
            by_tuple[key] = {}
        by_tuple[key][cond] = feat_record

    # Export letter features CSV & Parquet metadata
    feat_csv_path = os.path.join(out_dir, "letter_features.csv")
    if letter_features:
        # Exclude complex list types for CSV
        csv_rows = []
        for lf in letter_features:
            row = {}
            for k, v in lf.items():
                if k != "output_text" and isinstance(v, (int, float, str, bool)):
                    row[k] = v
            csv_rows.append(row)

        with open(feat_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # 2. Paired Difference Extraction
    paired_diffs: List[Dict[str, Any]] = []
    sentence_alignments: List[Dict[str, Any]] = []

    pair_specs = [
        ("pronoun_masc", "pronoun_fem", "gender_pronoun_pair"),
        ("name_masc", "name_fem", "gender_name_pair"),
        ("pronoun_fem", "anonymous", "fem_pronoun_vs_anon_pair"),
        ("name_fem", "anonymous", "fem_name_vs_anon_pair"),
        ("pronoun_masc", "anonymous", "masc_pronoun_vs_anon_pair"),
        ("name_masc", "anonymous", "masc_name_vs_anon_pair"),
    ]

    pair_id_counter = 1

    for (c_id, p_id, s_val), cond_dict in by_tuple.items():
        for cond_a, cond_b, pair_label in pair_specs:
            if cond_a in cond_dict and cond_b in cond_dict:
                rec_a = cond_dict[cond_a]
                rec_b = cond_dict[cond_b]

                text_a = rec_a["output_text"]
                text_b = rec_b["output_text"]

                align_info = align_sentences(text_a, text_b)

                tokens_a = re.findall(r"\b\w+\b", text_a.lower())
                tokens_b = re.findall(r"\b\w+\b", text_b.lower())
                token_edit_dist = _levenshtein_distance(tokens_a, tokens_b)

                w_diff = abs(rec_a["word_count"] - rec_b["word_count"])
                ag_diff = round(abs(rec_a["agentic_density"] - rec_b["agentic_density"]), 3)
                com_diff = round(abs(rec_a["communal_density"] - rec_b["communal_density"]), 3)
                warm_diff = round(abs(rec_a["warmth_density"] - rec_b["warmth_density"]), 3)
                lead_diff = round(abs(rec_a["leadership_density"] - rec_b["leadership_density"]), 3)

                pair_rec = {
                    "pair_id": f"PAIR_{pair_id_counter:03d}",
                    "pair_label": pair_label,
                    "case_id": c_id,
                    "prompt_id": p_id,
                    "seed": s_val,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "gen_id_a": rec_a["generation_id"],
                    "gen_id_b": rec_b["generation_id"],
                    "word_count_diff": w_diff,
                    "token_edit_distance": token_edit_dist,
                    "sentence_edit_distance": align_info["sentence_edit_distance"],
                    "alignment_similarity": align_info["alignment_similarity"],
                    "agentic_density_diff": ag_diff,
                    "communal_density_diff": com_diff,
                    "warmth_density_diff": warm_diff,
                    "leadership_density_diff": lead_diff,
                    "unsupported_spec_flag_a": rec_a["unsupported_specificity_flag"],
                    "unsupported_spec_flag_b": rec_b["unsupported_specificity_flag"],
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
        with open(paired_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired_diffs[0].keys()))
            writer.writeheader()
            writer.writerows(paired_diffs)

    # Export sentence alignments JSONL
    align_jsonl_path = os.path.join(out_dir, "sentence_alignments.jsonl")
    with open(align_jsonl_path, "w", encoding="utf-8") as f:
        for sa in sentence_alignments:
            f.write(json.dumps(sa) + "\n")

    return letter_features, paired_diffs
