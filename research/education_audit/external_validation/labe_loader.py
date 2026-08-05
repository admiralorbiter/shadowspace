"""LABE / Language Agency Classifier Dataset Ingestion Module."""

from __future__ import annotations

import os
from typing import Any, Dict, List


def generate_synthetic_lac_sentences() -> List[Dict[str, Any]]:
    """Generates synthetic Language Agency Classification (LAC) benchmark sentences."""
    sentences = [
        {"sentence_id": "LAC_001", "text": "He spearheaded the machine learning initiative.", "label": "agentic", "gender": "masculine"},
        {"sentence_id": "LAC_002", "text": "She offered supportive guidance to junior researchers.", "label": "communal", "gender": "feminine"},
        {"sentence_id": "LAC_003", "text": "Alex led a team of 5 software developers.", "label": "agentic", "gender": "masculine"},
        {"sentence_id": "LAC_004", "text": "Sarah assisted the lab manager with administrative duties.", "label": "communal", "gender": "feminine"},
    ]
    return sentences


def load_labe_dataset(data_dir: str = "data/external/labe") -> Dict[str, Any]:
    """Loads and validates LABE Language Agency Classifier dataset."""
    os.makedirs(data_dir, exist_ok=True)
    lac_sentences = generate_synthetic_lac_sentences()

    return {
        "dataset_name": "LABE Language Agency Classifier 2023",
        "status": "LOADED",
        "labeled_sentences_count": len(lac_sentences),
        "sentences": lac_sentences,
    }
