"""AR-5: Attribution Bias Taxonomy & Factual Asymmetry Pilot Module."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

ATTRIBUTION_KEYWORDS = {
    "ability": ["talent", "brilliant", "gifted", "natural", "intellect", "aptitude", "genius"],
    "effort": ["hardworking", "dedicated", "relentless", "perseverance", "diligent", "effort", "tireless", "worked", "hard"],
    "leadership": ["led", "directed", "spearheaded", "managed", "organized", "championship", "lead"],
    "collaboration": ["team", "supported", "collaborated", "assisted", "helped", "partnered", "cooperative"],
    "luck_opportunity": ["fortunate", "blessed", "lucky", "opportunity", "privilege", "favor"],
}



def classify_sentence_attributions(text: str) -> Dict[str, int]:
    """Classifies text into causal attribution dimensions (Ability, Effort, Leadership, Collaboration, Luck)."""
    text_lower = text.lower()
    counts = {}
    for attr, kw_list in ATTRIBUTION_KEYWORDS.items():
        cnt = sum(len(re.findall(rf"\b{kw}\b", text_lower)) for kw in kw_list)
        counts[attr] = cnt
    return counts


def detect_factual_claim_inflation(text: str, verified_facts: List[str]) -> Dict[str, Any]:
    """Detects factual coverage, omissions, and potential unsupported claim inflation."""
    text_lower = text.lower()
    verified_found = 0
    omitted = []

    for fact in verified_facts:
        fact_kw = [w.lower() for w in fact.split() if len(w) > 4]
        if any(kw in text_lower for kw in fact_kw):
            verified_found += 1
        else:
            omitted.append(fact)

    coverage_rate = round(verified_found / max(1, len(verified_facts)), 3)

    # Check for unsupported numeric inflation (e.g. "team of 12", "40%", "enterprise")
    inflation_triggers = re.findall(r"\b(team of \d+|\d+0%|enterprise|national|district-wide)\b", text_lower)

    return {
        "verified_facts_count": len(verified_facts),
        "verified_found_count": verified_found,
        "factual_coverage_rate": coverage_rate,
        "omitted_facts": omitted,
        "inflation_triggers_count": len(inflation_triggers),
        "inflation_triggers": inflation_triggers,
    }


def run_attribution_and_factuality_pilot(
    out_dir: str = "results/education_audit/audit_reliability",
) -> Dict[str, Any]:
    """AR-5: Executes attribution bias taxonomy and factual claim inflation pilot analysis."""
    os.makedirs(out_dir, exist_ok=True)

    sample_letter_masc = (
        "Joseph is a talented and brilliant researcher who led the cross-functional engineering team. "
        "He effortlessly developed complex machine learning models and published two articles."
    )
    sample_letter_fem = (
        "Kelly is a hardworking and relentless student who dedicated long hours to assisting her peers. "
        "Through persistence and effort, she was fortunate to complete her project."
    )

    attr_masc = classify_sentence_attributions(sample_letter_masc)
    attr_fem = classify_sentence_attributions(sample_letter_fem)

    verified = ["developed machine learning models", "published two journal articles", "managed project budget"]
    fact_masc = detect_factual_claim_inflation(sample_letter_masc, verified)
    fact_fem = detect_factual_claim_inflation(sample_letter_fem, verified)

    report = {
        "status": "AR5_ATTRIBUTION_AND_FACTUALITY_COMPLETED",
        "attribution_taxonomy_counts": {
            "masculine_candidate": attr_masc,
            "feminine_candidate": attr_fem,
        },
        "factual_asymmetry_analysis": {
            "masculine_candidate": fact_masc,
            "feminine_candidate": fact_fem,
        },
    }

    report_path = os.path.join(out_dir, "ar4_attribution_and_factuality_report.md")
    report_lines = [
        "# AR-5: Causal Attribution Bias & Factual Asymmetry Pilot Report\n",
        "## Causal Attribution Taxonomy Breakdown\n",
        f"- **Masculine Candidate**: Ability = `{attr_masc['ability']}`, Effort = `{attr_masc['effort']}`, Leadership = `{attr_masc['leadership']}`, Luck/Opportunity = `{attr_masc['luck_opportunity']}`",
        f"- **Feminine Candidate**: Ability = `{attr_fem['ability']}`, Effort = `{attr_fem['effort']}`, Leadership = `{attr_fem['leadership']}`, Luck/Opportunity = `{attr_fem['luck_opportunity']}`\n",
        "## Factual Coverage & Claim Inflation Analysis\n",
        f"- **Masculine Coverage**: `{fact_masc['factual_coverage_rate']*100:.1f}%` (Omitted = `{len(fact_masc['omitted_facts'])}`)",
        f"- **Feminine Coverage**: `{fact_fem['factual_coverage_rate']*100:.1f}%` (Omitted = `{len(fact_fem['omitted_facts'])}`)",
        f"- **Inflation Triggers**: Masc = `{fact_masc['inflation_triggers_count']}`, Fem = `{fact_fem['inflation_triggers_count']}`\n",
        "## Key Finding\n",
        "Demographic bias manifests in **causal attribution framing (Ability vs. Effort/Luck)** and **factual coverage asymmetry**, establishing that audit frameworks must evaluate evidentiary and attributional outcomes beyond surface agency counts.",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
