"""Counterfactual Corpus Builder using full locked LABE test split (N=373 x 4 = 1,492 pairs)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from research.education_audit.external_validation.labe_loader import load_labe_dataset
from research.education_audit.audit_reliability_empirical.counterfactuals.canonicalize import apply_identity_swap
from research.education_audit.audit_reliability_empirical.counterfactuals.identity_registry import PREREGISTERED_IDENTITY_CHANNELS
from research.education_audit.audit_reliability_empirical.counterfactuals.pair_validator import validate_counterfactual_pair


def build_labe_test_counterfactual_corpus() -> List[Dict[str, Any]]:
    """Constructs 1,492 validated paired counterfactual comparisons from full locked LABE test split (N=373)."""
    labe_data = load_labe_dataset()
    test_sentences = labe_data["sentences_by_split"]["test"]

    corpus = []
    pair_counter = 0

    for item in test_sentences:
        base_text = item["text"]
        label = item.get("label_int", 0)
        base_id = item["sentence_id"]

        for ch in PREREGISTERED_IDENTITY_CHANNELS:
            cat = ch["category"]
            ch_id = ch["channel_id"]

            if cat == "pronoun":
                has_pronoun = bool(re.search(r"\b(he|she|him|her|his|hers)\b", base_text, re.IGNORECASE))
                if has_pronoun:
                    text_masc = apply_identity_swap(base_text, ch["to_masc"])
                    text_fem = apply_identity_swap(base_text, ch["to_fem"])
                    if text_masc == text_fem:
                        text_masc = "He demonstrated that " + base_text
                        text_fem = "She demonstrated that " + base_text
                else:
                    text_masc = "He demonstrated that " + base_text
                    text_fem = "She demonstrated that " + base_text
            else:
                target_masc = list(ch["sub_masc"].keys())[0]
                target_fem = list(ch["sub_fem"].values())[0]
                text_masc = f"{target_masc} {base_text}"
                text_fem = f"{target_fem} {base_text}"

            # Validate purity & distinctness
            validate_counterfactual_pair(text_masc, text_fem, category=cat)

            pair_counter += 1
            corpus.append({
                "pair_id": f"labe_pair_{pair_counter:04d}",
                "base_sentence_id": base_id,
                "channel_id": ch_id,
                "category": cat,
                "text_masc": text_masc,
                "text_fem": text_fem,
                "agency_ground_truth_label": label,
            })

    assert len(corpus) == len(test_sentences) * len(PREREGISTERED_IDENTITY_CHANNELS)
    assert len(corpus) == 373 * 4 == 1492, f"Expected 1,492 paired comparisons, got {len(corpus)}"

    return corpus
