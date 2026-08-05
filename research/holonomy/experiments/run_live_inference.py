"""Standalone Live NLI Inference Runner (Decoupled from Analysis).

Executes batched, prospectively pinned live inference for FacebookAI/roberta-large-mnli
over the 300 duplicate-free controlled orbit dataset and exports rich prediction records.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from research.holonomy.natural_language.controlled_orbit_dataset import build_controlled_orbit_dataset
from research.holonomy.natural_language.model_adapter import HuggingFaceNLIAdapter, LiveNLIConfig


def run_live_inference(
    config: LiveNLIConfig | None = None,
    seed: int = 42,
    out_dir: str = "results/holonomy/e2_a1_2",
) -> str:
    """Executes live NLI inference and exports predictions JSON."""
    if config is None:
        config = LiveNLIConfig(
            model_id="FacebookAI/roberta-large-mnli",
            revision="2a8f12d27941090092df78e4ba6f0928eb5eac98",
            batch_size=16,
            use_mock_fallback=False,  # Hard live execution refusal if offline
        )

    adapter = HuggingFaceNLIAdapter(model_name=config.model_id, config=config)
    success = adapter.load()
    provenance = adapter.get_provenance_metadata()

    # Hard live execution assertions
    assert success is True, f"Live model loading failed: {adapter.load_error}"
    assert provenance["is_loaded"] is True, "Adapter is not loaded!"
    assert provenance["use_mock_fallback"] is False, "Mock fallback must be False!"
    assert provenance["resolved_model_revision"] == config.revision, f"Model revision mismatch: {provenance.get('resolved_model_revision')}"
    assert provenance["resolved_tokenizer_revision"] == config.revision, f"Tokenizer revision mismatch: {provenance.get('resolved_tokenizer_revision')}"

    # Build 300 unique controlled orbits
    ds = build_controlled_orbit_dataset(target_orbit_count=300, seed=seed)
    all_orbits = ds.train_orbits + ds.val_orbits + ds.test_orbits

    pair_list: List[Tuple[str, str]] = []
    vertex_map: List[Tuple[str, str]] = []

    for orb in all_orbits:
        for v_id in ["x0", "x1", "x2", "x3"]:
            v = orb.get_vertex(v_id)
            pair_list.append((v.premise, v.hypothesis))
            vertex_map.append((orb.orbit_id, v_id))

    print(f"Running batched live inference over {len(pair_list)} vertex pairs...")
    batch = adapter.predict_batch(pair_list)

    orbit_preds: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for idx, (orb_id, v_id) in enumerate(vertex_map):
        if orb_id not in orbit_preds:
            orbit_preds[orb_id] = {}
        orbit_preds[orb_id][v_id] = {
            "raw_logits": batch.raw_logits[idx],
            "aligned_logits": batch.aligned_logits[idx],
            "probabilities": batch.probabilities[idx],
            "ilr_coords": batch.ilr_coordinates[idx],
            "token_count": batch.token_counts[idx],
            "truncated": batch.truncated[idx],
        }

    os.makedirs(out_dir, exist_ok=True)
    pred_records = []
    for orb in all_orbits:
        for v_id in ["x0", "x1", "x2", "x3"]:
            v = orb.get_vertex(v_id)
            pdata = orbit_preds[orb.orbit_id][v_id]
            pred_records.append({
                "orbit_id": orb.orbit_id,
                "vertex_id": v_id,
                "premise": v.premise,
                "hypothesis": v.hypothesis,
                "raw_logits": pdata["raw_logits"].tolist(),
                "aligned_logits": pdata["aligned_logits"].tolist(),
                "probabilities": pdata["probabilities"].tolist(),
                "ilr_coords": pdata["ilr_coords"].tolist(),
                "token_count": int(pdata["token_count"]),
                "truncated": bool(pdata["truncated"]),
                "track": orb.metadata.get("track", "unknown"),
                "label_class": orb.metadata.get("label_class", "unknown"),
                "quartet": list(orb.metadata.get("quartet", [])),
                "template_idx": int(orb.metadata.get("template_idx", -1)),
            })

    output_path = os.path.join(out_dir, "predictions_roberta.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pred_records, f, indent=2)

    print(f"Live predictions exported to: {output_path}")
    return output_path


if __name__ == "__main__":
    run_live_inference()
