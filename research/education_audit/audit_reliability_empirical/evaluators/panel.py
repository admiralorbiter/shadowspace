"""AR-1: Evaluator Panel Initialization & Provenance Validation (Loaded from Frozen Checkpoints)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from research.education_audit.audit_reliability_empirical.provenance import validate_evaluator_dependency_graph
from research.education_audit.audit_reliability_empirical.evaluators.exact_lexicon import ExactLexiconEvaluator
from research.education_audit.audit_reliability_empirical.evaluators.sparse_ngram import SparseNgramEnsembleEvaluator
from research.education_audit.audit_reliability_empirical.evaluators.labe_transformer import LABETransformerAgencyEvaluator


def initialize_empirical_evaluator_panel(
    out_dir: str = "results/education_audit/audit_reliability_empirical",
) -> Dict[str, Any]:
    """Initializes the 3 independent evaluators from frozen checkpoints and validates the provenance dependency graph."""
    os.makedirs(out_dir, exist_ok=True)

    lexicon_eval = ExactLexiconEvaluator()
    ngram_eval = SparseNgramEnsembleEvaluator()
    transformer_eval = LABETransformerAgencyEvaluator()

    panel = {
        "exact_lexicon": lexicon_eval,
        "sparse_ngram_ensemble": ngram_eval,
        "labe_bert_transformer": transformer_eval,
    }

    # Validate provenance graph
    provenance_list = [e.provenance for e in panel.values()]
    validate_evaluator_dependency_graph(provenance_list)

    independent_count = sum(1 for p in provenance_list if p.is_independent)
    assert independent_count == 3, f"Expected 3 independent evaluators, got {independent_count}"

    panel_manifest = {
        "status": "EMPIRICAL_PANEL_INITIALIZED",
        "evaluators_count": len(panel),
        "independent_evaluators_count": independent_count,
        "evaluators": {k: v.provenance.to_dict() for k, v in panel.items()},
    }

    manifest_path = os.path.join(out_dir, "evaluator_panel_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(panel_manifest, f, indent=2)

    return {
        "status": "EMPIRICAL_PANEL_INITIALIZED",
        "panel": panel,
        "panel_manifest": panel_manifest,
    }
