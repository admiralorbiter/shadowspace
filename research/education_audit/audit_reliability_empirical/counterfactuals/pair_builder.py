"""Counterfactual Corpus Builder with Natural Substitutions and Controlled Injection Benchmarks."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from research.education_audit.external_validation.labe_loader import load_labe_dataset
from research.education_audit.audit_reliability_empirical.counterfactuals.natural_substitutions import apply_natural_pronoun_swap
from research.education_audit.audit_reliability_empirical.counterfactuals.injection_frames import apply_controlled_injection_framing
from research.education_audit.audit_reliability_empirical.counterfactuals.identity_registry import PREREGISTERED_IDENTITY_CHANNELS
from research.education_audit.audit_reliability_empirical.counterfactuals.pair_validator import validate_counterfactual_pair_purity


def build_labe_test_counterfactual_corpus() -> Dict[str, List[Dict[str, Any]]]:
    """Constructs separated Natural Substitutions and Controlled Injection Counterfactual Corpora."""
    labe_data = load_labe_dataset()
    test_sentences = labe_data["sentences_by_split"]["test"]

    natural_corpus = []
    injection_corpus = []
    rejected_log = []

    pair_counter = 0

    for item in test_sentences:
        base_text = item["text"]
        label = item.get("label_int", 0)
        base_id = item["sentence_id"]

        # 1. Natural In-Place Substitution (if sentence contains natural pronouns)
        nat_swap = apply_natural_pronoun_swap(base_text)
        if nat_swap:
            text_m, text_f = nat_swap
            try:
                validate_counterfactual_pair_purity(text_m, text_f, category="pronoun")
                pair_counter += 1
                natural_corpus.append({
                    "pair_id": f"nat_pair_{pair_counter:04d}",
                    "base_sentence_id": base_id,
                    "channel_id": "pronoun_natural_in_place",
                    "category": "pronoun_natural",
                    "text_masc": text_m,
                    "text_fem": text_f,
                    "agency_ground_truth_label": label,
                })
            except ValueError as err:
                rejected_log.append({"base_sentence_id": base_id, "type": "natural", "reason": str(err)})

        # 2. Controlled Identity Injection Framing across all 4 registered identity channels
        for ch in PREREGISTERED_IDENTITY_CHANNELS:
            cat = ch["category"]
            ch_id = ch["channel_id"]

            if cat == "pronoun":
                target_m, target_f = "he", "she"
            else:
                target_m = list(ch["sub_masc"].keys())[0]
                target_f = list(ch["sub_fem"].values())[0]

            text_m, text_f = apply_controlled_injection_framing(base_text, target_m, target_f, category=cat)

            try:
                validate_counterfactual_pair_purity(text_m, text_f, category=cat)
                pair_counter += 1
                injection_corpus.append({
                    "pair_id": f"inj_pair_{pair_counter:04d}",
                    "base_sentence_id": base_id,
                    "channel_id": ch_id,
                    "category": f"{cat}_injection",
                    "text_masc": text_m,
                    "text_fem": text_f,
                    "agency_ground_truth_label": label,
                })
            except ValueError as err:
                rejected_log.append({"base_sentence_id": base_id, "type": "injection", "reason": str(err)})

    return {
        "natural_corpus": natural_corpus,
        "injection_corpus": injection_corpus,
        "rejected_log": rejected_log,
        "total_natural_pairs": len(natural_corpus),
        "total_injection_pairs": len(injection_corpus),
    }
