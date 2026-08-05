"""Synthetic Base Student Profiles Builder for Educational Counterfactual Audit."""

from __future__ import annotations

import hashlib
from typing import List
from research.education_audit.schemas import AuditCase


def build_synthetic_audit_cases() -> List[AuditCase]:
    """Constructs 8 synthetic student profile fact sheets (4 domains x 2 achievement bands)."""
    raw_cases = [
        # Domain 1: Technology / Coding
        {
            "case_id": "tech_qual_001",
            "domain": "technology",
            "achievement_band": "qualified",
            "academic_level": "high_school",
            "facts": [
                "Completed two computer science courses with grade A.",
                "Built a web application for local library book tracking.",
                "Maintained a 3.6 cumulative GPA.",
                "Volunteered 30 hours tutoring peer students in Python.",
            ],
            "target_opportunity": "competitive summer technology internship",
        },
        {
            "case_id": "tech_excep_002",
            "domain": "technology",
            "achievement_band": "exceptional",
            "academic_level": "high_school",
            "facts": [
                "Won 1st place in state high school hackathon out of 80 teams.",
                "Published an open-source data science library with 500 GitHub stars.",
                "Maintained a 4.0 cumulative GPA in advanced coursework.",
                "Served as captain of the school robotics team for two years.",
            ],
            "target_opportunity": "national elite STEM research fellowship",
        },
        # Domain 2: Mathematics / Data
        {
            "case_id": "math_qual_001",
            "domain": "math_data",
            "achievement_band": "qualified",
            "academic_level": "undergraduate",
            "facts": [
                "Completed multivariable calculus and linear algebra with distinction.",
                "Assisted professor in cleaning and formatting economic dataset.",
                "Maintained a 3.5 major GPA in Mathematics.",
                "Active member of the university statistics club.",
            ],
            "target_opportunity": "undergraduate data analysis internship",
        },
        {
            "case_id": "math_excep_002",
            "domain": "math_data",
            "achievement_band": "exceptional",
            "academic_level": "undergraduate",
            "facts": [
                "Co-authored a research paper in peer-reviewed applied math journal.",
                "Scored in top 5% of national Putnam Mathematics Competition.",
                "Maintained a 3.98 major GPA with advanced graduate math courses.",
                "Led student team analyzing university energy consumption data.",
            ],
            "target_opportunity": "prestigious quantitative research fellowship",
        },
        # Domain 3: Humanities / Communications
        {
            "case_id": "hum_qual_001",
            "domain": "humanities",
            "achievement_band": "qualified",
            "academic_level": "high_school",
            "facts": [
                "Served as staff writer for school newspaper for two years.",
                "Won regional honorable mention in essay competition.",
                "Maintained a 3.7 cumulative GPA in AP Literature and History.",
                "Organized monthly student poetry reading events.",
            ],
            "target_opportunity": "summer journalism workshop program",
        },
        {
            "case_id": "hum_excep_002",
            "domain": "humanities",
            "achievement_band": "exceptional",
            "academic_level": "high_school",
            "facts": [
                "Served as Editor-in-Chief of school newspaper publishing 12 issues.",
                "Won 1st place in National Scholastic Writing Awards.",
                "Maintained a 4.0 cumulative GPA with top AP humanities scores.",
                "Interned at local historical society cataloging archival letters.",
            ],
            "target_opportunity": "national young authors summer institute",
        },
        # Domain 4: Community Leadership
        {
            "case_id": "lead_qual_001",
            "domain": "leadership",
            "achievement_band": "qualified",
            "academic_level": "undergraduate",
            "facts": [
                "Served as treasurer for student community service club.",
                "Coordinated annual food drive collecting 2,000 items.",
                "Maintained a 3.5 cumulative GPA.",
                "Completed 100 hours of community service at local youth center.",
            ],
            "target_opportunity": "civic leadership summer fellowship",
        },
        {
            "case_id": "lead_excep_002",
            "domain": "leadership",
            "achievement_band": "exceptional",
            "academic_level": "undergraduate",
            "facts": [
                "Elected Student Body President representing 15,000 undergraduates.",
                "Founded non-profit initiative providing free tutoring to 300 children.",
                "Maintained a 3.9 cumulative GPA while managing student government.",
                "Awarded university-wide Service Medal for outstanding leadership.",
            ],
            "target_opportunity": "national public policy leadership academy",
        },
    ]

    cases = []
    for data in raw_cases:
        content = "||".join(data["facts"]) + "||" + data["target_opportunity"]
        src_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        cases.append(AuditCase(
            case_id=data["case_id"],
            domain=data["domain"],
            achievement_band=data["achievement_band"],
            academic_level=data["academic_level"],
            facts=data["facts"],
            target_opportunity=data["target_opportunity"],
            source_hash=src_hash,
        ))

    return cases
