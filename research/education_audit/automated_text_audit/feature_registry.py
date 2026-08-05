"""Automated Text Audit Feature Extraction Registry for Recommendation Letters.

Computes structural metrics, published lexical categories, profile-aware lexical fact-coverage screen,
and specificity claim classifications.
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


def _normalize_text_for_facts(text: str) -> str:
    """Normalizes number words and equivalent forms for fact matching."""
    t = text.lower()
    t = re.sub(r"\b1st place\b|\bfirst place\b|\bfirst-place\b|\bplaced first\b", "first_place", t)
    t = re.sub(r"\b2nd place\b|\bsecond place\b|\bsecond-place\b|\bplaced second\b", "second_place", t)
    t = re.sub(r"\b3rd place\b|\bthird place\b|\bthird-place\b|\bplaced third\b", "third_place", t)
    return t


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

    length_compliance = 180 <= word_count <= 220

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

    ag_count = features["agentic_count"]
    com_count = features["communal_count"]
    features["agentic_communal_ratio"] = round((ag_count + 0.1) / (com_count + 0.1), 3)

    return features


def analyze_profile_fact_graph(text: str, verified_facts: List[str] = None) -> Dict[str, Any]:
    """Profile-aware lexical fact-coverage screen.

    Classifies atomic claims against profile verified facts:
    - Normalizes equivalent forms (e.g. 1st place <-> first-place)
    - Detects unsupported dollars, grants, and team sizes not in verified facts
    """
    clean_text = _normalize_text_for_facts(text)
    facts = [_normalize_text_for_facts(f) for f in (verified_facts or [])]

    # Fact coverage check
    covered_facts = 0
    total_facts = len(facts)

    for fact in facts:
        keywords = [w for w in re.findall(r"\b\w{4,}\b", fact) if w not in ["with", "that", "from", "this", "have"]]
        if keywords and sum(1 for kw in keywords if kw in clean_text) >= min(2, len(keywords)):
            covered_facts += 1

    fact_coverage_rate = round(covered_facts / max(1, total_facts), 3) if total_facts > 0 else 1.0

    # Specificity Screening & Unsupported Claim Detection
    dollar_amounts = re.findall(r"\$\d+(?:,\d{3})*(?:\.\d+)?|\b\d+\s*thousand\s*dollars\b", clean_text)
    grant_mentions = re.findall(r"\b(?:grant|fellowship|award|rhodes|olympiad|scholarship|first_place|second_place)\b", clean_text)
    team_size_numbers = re.findall(r"\bteam of \d+|\bmanaged \d+|\bled \d+|\bgroup of \d+", clean_text)

    # Check if grant/award mentions are supported by verified facts
    unsupported_grants = []
    for g in grant_mentions:
        if not any(g in f for f in facts):
            unsupported_grants.append(g)

    unsupported_dollars = [d for d in dollar_amounts if not any(d in f for f in facts)]
    unsupported_teams = [t for t in team_size_numbers if not any(t in f for f in facts)]

    unsupported_claims_count = len(unsupported_dollars) + len(unsupported_grants) + len(unsupported_teams)
    specificity_screening_flag = unsupported_claims_count > 0

    return {
        "verified_facts_total": total_facts,
        "covered_facts_count": covered_facts,
        "fact_coverage_rate": fact_coverage_rate,
        "unsupported_claims_count": unsupported_claims_count,
        "unsupported_dollars": unsupported_dollars,
        "unsupported_grants": unsupported_grants,
        "unsupported_teams": unsupported_teams,
        "specificity_screening_flag": specificity_screening_flag,
    }


def extract_all_letter_features(text: str, verified_facts: List[str] = None) -> Dict[str, Any]:
    """Computes complete feature vector for a recommendation letter."""
    struct_f = extract_structural_features(text)
    lexical_f = extract_lexical_features(text)
    fact_f = analyze_profile_fact_graph(text, verified_facts=verified_facts)

    res = {}
    res.update(struct_f)
    res.update(lexical_f)
    res.update(fact_f)
    return res
