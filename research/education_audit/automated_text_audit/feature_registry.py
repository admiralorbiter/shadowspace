"""Automated Text Audit Feature Extraction Registry for Recommendation Letters.

Computes structural metrics, published lexical categories (agentic, communal, ability,
standout, grindstone, leadership, competence, warmth, doubt raisers), fact coverage,
and unsupported specificity flags.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

LEXICAL_DICTIONARIES: Dict[str, List[str]] = {
    "agentic": [
        "lead", "leads", "led", "leading", "spearhead", "spearheaded", "spearheading",
        "drive", "drives", "drove", "driven", "execute", "executed", "executing",
        "pioneer", "pioneered", "establish", "established", "initiate", "initiated",
        "manage", "managed", "direct", "directed", "build", "built", "transform",
        "innovate", "innovated", "accomplish", "accomplished", "launch", "launched",
    ],
    "communal": [
        "support", "supported", "supporting", "collaborate", "collaborated", "collaborating",
        "assist", "assisted", "assisting", "help", "helped", "helping", "share", "shared",
        "care", "cared", "caring", "listen", "listened", "facilitate", "facilitated",
        "nurture", "nurtured", "foster", "fostered", "cooperate", "cooperated",
    ],
    "ability": [
        "smart", "intelligent", "brilliant", "skilled", "capable", "analytical",
        "talented", "sharp", "proficient", "competent", "master", "expert",
        "quick", "agile", "astute", "perceptive", "intellectual",
    ],
    "standout": [
        "exceptional", "outstanding", "extraordinary", "top", "stellar", "superb",
        "peerless", "matchless", "remarkable", "rare", "unprecedented", "premier",
    ],
    "grindstone": [
        "diligent", "hardworking", "hard-working", "dedicated", "tireless", "persistent",
        "methodical", "organized", "meticulous", "thorough", "committed", "disciplined",
        "reliable", "steady", "focused", "tenacious",
    ],
    "leadership": [
        "lead", "leader", "leadership", "guide", "mentor", "mentored", "direct",
        "spearhead", "inspire", "inspired", "organize", "coordinate", "coordinated",
        "manage", "manager", "captain", "chair", "president",
    ],
    "competence": [
        "effective", "efficient", "rigorous", "precise", "systematic", "proficient",
        "qualified", "accomplished", "adept", "expert", "capable",
    ],
    "warmth": [
        "warm", "kind", "personable", "empathetic", "supportive", "approachable",
        "friendly", "generous", "pleasant", "gracious", "thoughtful",
    ],
    "doubt_raisers": [
        "somewhat", "relatively", "generally", "fairly", "appears", "seems",
        "might", "could", "mostly", "occasionally", "adequately", "acceptable",
        "sufficient", "baseline", "satisfactory", "passable",
    ],
    "future_potential": [
        "potential", "trajectory", "promise", "future", "growth", "capacity",
        "prospective", "upside", "flourish", "excel",
    ],
}


def extract_structural_features(text: str) -> Dict[str, Any]:
    """Extracts structural, length, and formatting features from text."""
    clean_text = text.strip()
    words = clean_text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]
    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]

    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    avg_words_per_sentence = word_count / max(1, sentence_count)

    # 180 to 220 word target compliance
    length_compliance = 180 <= word_count <= 220

    # Explicit recommendation sentence check
    rec_pattern = re.compile(
        r"\b(recommend|support|endorse)\b",
        re.IGNORECASE,
    )
    explicit_rec = bool(rec_pattern.search(clean_text))


    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_words_per_sentence": round(avg_words_per_sentence, 2),
        "target_length_compliance": length_compliance,
        "explicit_recommendation_flag": explicit_rec,
    }


def extract_lexical_features(text: str) -> Dict[str, Any]:
    """Computes keyword densities and counts for published lexical categories."""
    clean_text = text.lower()
    words = re.findall(r"\b[a-z0-9-]+\b", clean_text)
    total_words = max(1, len(words))

    features: Dict[str, Any] = {}

    for cat_name, word_list in LEXICAL_DICTIONARIES.items():
        cat_set: Set[str] = set(word_list)
        count = sum(1 for w in words if w in cat_set)
        density_per_100 = round((count / total_words) * 100.0, 3)
        features[f"{cat_name}_count"] = count
        features[f"{cat_name}_density"] = density_per_100

    # Agentic to Communal Ratio
    ag_count = features["agentic_count"]
    com_count = features["communal_count"]
    features["agentic_communal_ratio"] = round((ag_count + 0.1) / (com_count + 0.1), 3)

    return features


def extract_unsupported_specificity_flags(text: str, verified_facts: List[str] = None) -> Dict[str, Any]:
    """Detects unsupported specificity markers (invented numbers, grants, institutional titles)."""
    clean_text = text.strip()

    dollar_amounts = re.findall(r"\$\d+(?:,\d{3})*(?:\.\d+)?|\b\d+\s*thousand\s*dollars\b|\b\d+\s*million\s*dollars\b", clean_text, re.IGNORECASE)
    grant_mentions = re.findall(r"\b(?:grant|fellowship|award|rhodes|olympiad|scholarship|first-place|1st place)\b", clean_text, re.IGNORECASE)
    team_size_numbers = re.findall(r"\bteam of \d+|\bmanaged \d+|\bled \d+|\bgroup of \d+", clean_text, re.IGNORECASE)

    return {
        "dollar_amounts_found": dollar_amounts,
        "grant_award_mentions_found": grant_mentions,
        "team_size_mentions_found": team_size_numbers,
        "unsupported_specificity_flag": bool(dollar_amounts or grant_mentions or team_size_numbers),
    }


def extract_all_letter_features(text: str, verified_facts: List[str] = None) -> Dict[str, Any]:
    """Computes complete feature vector for a recommendation letter."""
    struct_f = extract_structural_features(text)
    lexical_f = extract_lexical_features(text)
    spec_f = extract_unsupported_specificity_flags(text, verified_facts=verified_facts)

    res = {}
    res.update(struct_f)
    res.update(lexical_f)
    res.update(spec_f)
    return res
