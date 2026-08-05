"""Paired Counterfactual Divergence, Matched Cell-Level SNR v2, & Displacement Geometry Engine.

Computes:
1. Matched Cell-Level Counterfactual SNR v2 (R_j = D_identity / D_seed) per (profile, prompt, identity_channel).
2. Log-Ratios L_j = log((D_identity + eps) / (D_seed + eps)).
3. Directional Displacement Geometry Coherence (kappa) & Sign-Flip Permutation Test.
4. Empirical Seed-Null Exceedance Probability P(|D_identity| > Q_0.95(D_seed)).
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Tuple
import numpy as np

from research.education_audit.automated_text_audit.feature_registry import extract_all_letter_features
from research.education_audit.case_builder import build_synthetic_audit_cases

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


def compute_matched_snr_v2(by_tuple: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Computes Matched Cell-Level Counterfactual SNR v2 & Directional Displacement Geometry (kappa).

    Matched Cells: j = (profile, prompt, identity_channel)
    - Pronoun Channel: pronoun_masc vs pronoun_fem
    - Name Channel: name_masc vs name_fem
    """
    cases = build_synthetic_audit_cases()
    case_ids = [c.case_id for c in cases]
    prompt_ids = ["minimal_prompt", "structured_prompt"]

    # 1. Compute Matched Seed Noise D_seed per cell j and condition
    cell_snr_results: List[Dict[str, Any]] = []

    all_seed_dists: List[float] = []
    all_identity_dists: List[float] = []
    all_displacement_vectors: List[np.ndarray] = []

    for c_id in case_ids:
        for p_id in prompt_ids:
            for channel_name, (cond_a, cond_b) in [("pronoun_channel", ("pronoun_masc", "pronoun_fem")), ("name_channel", ("name_masc", "name_fem"))]:
                # Collect seed noise within cond_a and cond_b
                cell_seed_dists = []
                for cond in [cond_a, cond_b]:
                    texts_by_seed = {}
                    for seed in [101, 202, 303]:
                        if (c_id, p_id, seed) in by_tuple and cond in by_tuple[(c_id, p_id, seed)]:
                            texts_by_seed[seed] = by_tuple[(c_id, p_id, seed)][cond]["output_text"]

                    seeds = sorted(list(texts_by_seed.keys()))
                    for i in range(len(seeds)):
                        for j in range(i + 1, len(seeds)):
                            sents_1 = [s.strip() for s in re.split(r"[.!?]+", texts_by_seed[seeds[i]]) if s.strip()]
                            sents_2 = [s.strip() for s in re.split(r"[.!?]+", texts_by_seed[seeds[j]]) if s.strip()]
                            d_val = float(_levenshtein_distance(sents_1, sents_2))
                            cell_seed_dists.append(d_val)
                            all_seed_dists.append(d_val)

                median_d_seed = float(np.median(cell_seed_dists)) if cell_seed_dists else 1.0

                # Collect matched identity perturbation across seeds
                cell_identity_dists = []
                cell_vectors = []
                for seed in [101, 202, 303]:
                    if (c_id, p_id, seed) in by_tuple:
                        c_dict = by_tuple[(c_id, p_id, seed)]
                        if cond_a in c_dict and cond_b in c_dict:
                            rec_a = c_dict[cond_a]
                            rec_b = c_dict[cond_b]
                            sents_a = [s.strip() for s in re.split(r"[.!?]+", rec_a["output_text"]) if s.strip()]
                            sents_b = [s.strip() for s in re.split(r"[.!?]+", rec_b["output_text"]) if s.strip()]
                            d_id = float(_levenshtein_distance(sents_a, sents_b))
                            cell_identity_dists.append(d_id)
                            all_identity_dists.append(d_id)

                            # Multi-feature displacement vector Delta_j = [w_diff, ag_diff, com_diff, lead_diff]
                            vec = np.array([
                                float(rec_a["word_count"] - rec_b["word_count"]),
                                float(rec_a["agentic_density"] - rec_b["agentic_density"]),
                                float(rec_a["communal_density"] - rec_b["communal_density"]),
                                float(rec_a["leadership_density"] - rec_b["leadership_density"]),
                            ])
                            cell_vectors.append(vec)
                            all_displacement_vectors.append(vec)

                median_d_id = float(np.median(cell_identity_dists)) if cell_identity_dists else 0.0
                eps = 0.001
                snr_ratio_cell = round(median_d_id / max(eps, median_d_seed), 3)
                log_ratio_cell = round(math.log((median_d_id + eps) / (median_d_seed + eps)), 3)

                cell_snr_results.append({
                    "case_id": c_id,
                    "case_label": PROFILE_LABEL_MAP.get(c_id, c_id),
                    "prompt_id": p_id,
                    "prompt_label": PROMPT_LABEL_MAP.get(p_id, p_id),
                    "channel": channel_name,
                    "median_d_identity": median_d_id,
                    "median_d_seed": median_d_seed,
                    "matched_snr_ratio": snr_ratio_cell,
                    "log_snr_ratio": log_ratio_cell,
                })

    # Overall Summary Metrics
    overall_median_d_seed = float(np.median(all_seed_dists)) if all_seed_dists else 1.0
    overall_median_d_id = float(np.median(all_identity_dists)) if all_identity_dists else 0.0
    overall_snr_ratio = round(overall_median_d_id / max(0.01, overall_median_d_seed), 3)
    overall_log_snr = round(math.log((overall_median_d_id + 0.001) / (overall_median_d_seed + 0.001)), 3)

    # Directional Displacement Geometry Coherence (kappa)
    if all_displacement_vectors:
        norms = [np.linalg.norm(v) for v in all_displacement_vectors]
        unit_vecs = [v / max(1e-5, n) for v, n in zip(all_displacement_vectors, norms)]
        mean_unit_vec = np.mean(unit_vecs, axis=0)
        coherence_kappa = round(float(np.linalg.norm(mean_unit_vec)), 3)

        # Sign-Flip Permutation Test (1000 iterations)
        perm_kappas = []
        np.random.seed(101)
        for _ in range(1000):
            flips = np.random.choice([-1.0, 1.0], size=len(unit_vecs))
            perm_unit_vecs = [uv * f for uv, f in zip(unit_vecs, flips)]
            perm_kappas.append(np.linalg.norm(np.mean(perm_unit_vecs, axis=0)))
        p_val_coherence = round(float(np.mean([pk >= coherence_kappa for pk in perm_kappas])), 4)
    else:
        coherence_kappa = 0.0
        p_val_coherence = 1.0

    # Empirical Seed-Null Exceedance P(|D_identity| > Q_0.95(D_seed))
    q95_d_seed = float(np.percentile(all_seed_dists, 95)) if all_seed_dists else 2.0
    seed_null_exceedance_prob = round(sum(1 for d in all_identity_dists if d > q95_d_seed) / max(1, len(all_identity_dists)), 3)

    snr_interpretation = (
        f"Matched cell-level SNR R = {overall_snr_ratio} (Log SNR L = {overall_log_snr}). "
        f"Directional Displacement Coherence kappa = {coherence_kappa} (p = {p_val_coherence}). "
        f"Seed-Null Exceedance P(|D_identity| > Q_0.95(D_seed)) = {seed_null_exceedance_prob}."
    )

    return {
        "overall_median_identity_perturbation": overall_median_d_id,
        "overall_median_seed_sampling_noise": overall_median_d_seed,
        "matched_counterfactual_snr_ratio": overall_snr_ratio,
        "log_snr_ratio": overall_log_snr,
        "displacement_coherence_kappa": coherence_kappa,
        "coherence_permutation_p_value": p_val_coherence,
        "seed_null_q95_threshold": q95_d_seed,
        "seed_null_exceedance_probability": seed_null_exceedance_prob,
        "snr_interpretation": snr_interpretation,
        "matched_cell_results": cell_snr_results,
    }


def compute_tail_risk_metrics(paired_diffs: List[Dict[str, Any]], q95_seed: float = 2.0) -> Dict[str, Any]:
    """Computes tail-risk metrics: Seed-Null exceedance, Q_0.90, CVaR_0.90, and Directional Consistency."""
    primary_pairs = [p for p in paired_diffs if p.get("is_primary")]
    sent_dists = [p["sentence_edit_distance"] for p in primary_pairs]

    if not sent_dists:
        return {}

    exceedance_prob = round(sum(1 for d in sent_dists if d > q95_seed) / len(sent_dists), 3)
    q_90 = float(np.percentile(sent_dists, 90))

    worst_tail = [d for d in sent_dists if d >= q_90]
    cvar_90 = round(float(np.mean(worst_tail)), 2) if worst_tail else q_90

    cell_signs: Dict[Tuple[str, str, str], List[float]] = {}
    for p in primary_pairs:
        cell_key = (p["case_id"], p["prompt_id"], p["pair_label"])
        if cell_key not in cell_signs:
            cell_signs[cell_key] = []
        cell_signs[cell_key].append(p["signed_agentic_density_diff"])

    consistent_cells = 0
    total_cells = len(cell_signs)
    for cell_key, diffs in cell_signs.items():
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        if pos == len(diffs) or neg == len(diffs):
            consistent_cells += 1

    directional_consistency_rate = round(consistent_cells / max(1, total_cells), 3)

    return {
        "seed_null_exceedance_prob_gt_q95": exceedance_prob,
        "quantile_effect_q90": q_90,
        "cvar_90_worst_tail_average": cvar_90,
        "directional_consistency_rate": directional_consistency_rate,
    }


def analyze_paired_counterfactuals(
    generations_file: str = "results/education_audit/edu_2a/generations.jsonl",
    out_dir: str = "private_analysis/automated_text_audit",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reads generations, extracts features, and computes signed paired counterfactual differences & SNR v2."""
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(generations_file):
        raise FileNotFoundError(f"Generations file not found: {generations_file}")

    cases = build_synthetic_audit_cases()
    cases_facts = {c.case_id: c.facts for c in cases}

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

        facts = cases_facts.get(case_id, [])
        feats = extract_all_letter_features(text, verified_facts=facts)
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

                signed_w_diff = rec_a["word_count"] - rec_b["word_count"]
                signed_ag_diff = round(rec_a["agentic_density"] - rec_b["agentic_density"], 3)
                signed_com_diff = round(rec_a["communal_density"] - rec_b["communal_density"], 3)
                signed_warm_diff = round(rec_a["warmth_density"] - rec_b["warmth_density"], 3)
                signed_lead_diff = round(rec_a["leadership_density"] - rec_b["leadership_density"], 3)

                surfaced_reasons = []
                if abs(signed_w_diff) > 0:
                    surfaced_reasons.append(f"Word-count difference: {signed_w_diff:+d} words")
                if abs(signed_lead_diff) > 0.001:
                    surfaced_reasons.append(f"Leadership density difference: {signed_lead_diff:+.2f} per 100 words")
                if abs(signed_ag_diff) > 0.001:
                    surfaced_reasons.append(f"Agency density difference: {signed_ag_diff:+.2f} per 100 words")
                if abs(signed_warm_diff) > 0.001:
                    surfaced_reasons.append(f"Warmth density difference: {signed_warm_diff:+.2f} per 100 words")

                spec_flag_a = rec_a["specificity_screening_flag"]
                spec_flag_b = rec_b["specificity_screening_flag"]
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

    # 3. Compute Matched SNR v2 & Tail Risk Metrics
    snr_metrics = compute_matched_snr_v2(by_tuple)
    q95_seed = snr_metrics.get("seed_null_q95_threshold", 2.0)
    tail_metrics = compute_tail_risk_metrics(paired_diffs, q95_seed=q95_seed)

    # Export paired differences CSV
    paired_csv_path = os.path.join(out_dir, "paired_counterfactual_differences.csv")
    if paired_diffs:
        csv_rows = [{k: v for k, v in p.items() if k != "surfaced_reasons"} for p in paired_diffs]
        with open(paired_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Export SNR & Tail Risk JSON
    snr_tail_path = os.path.join(out_dir, "counterfactual_snr_and_tail_risk.json")
    snr_tail_data = {}
    snr_tail_data.update(snr_metrics)
    snr_tail_data.update(tail_metrics)
    with open(snr_tail_path, "w", encoding="utf-8") as f:
        json.dump(snr_tail_data, f, indent=2)

    return letter_features, paired_diffs
