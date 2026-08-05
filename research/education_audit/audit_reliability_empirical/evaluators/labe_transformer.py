"""LABE Fine-Tuned Independent BERT Transformer Agency Classifier Evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from research.education_audit.audit_reliability_empirical.provenance import EvaluatorProvenance


def compute_file_sha256(filepath: str) -> str:
    """Computes exact 64-character SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class LABETransformerAgencyEvaluator:
    """LABE Fine-Tuned Independent BERT Transformer Classifier Evaluator (Loaded from Frozen Checkpoint)."""

    def __init__(self, checkpoint_dir: str = "models/labe_bert_agency"):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Required frozen LABE BERT checkpoint directory missing: {checkpoint_dir}")

        manifest_path = os.path.join(checkpoint_dir, "manifest.json")
        weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Required weights file missing in {checkpoint_dir}")

        weights_hash = compute_file_sha256(weights_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.50

        # Load fine-tuned tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            num_labels=2,
            output_loading_info=True
        )

        # Fail closed if classifier head weights are missing or uninitialized
        if loading_info.get("missing_keys"):
            raise ValueError(f"BERT Checkpoint loading failed with missing keys: {loading_info['missing_keys']}")

        self.model.to(self.device)
        self.model.eval()

        self.provenance = EvaluatorProvenance(
            evaluator_id="eval_labe_bert_transformer",
            evaluator_name="LABE Fine-Tuned BERT Transformer Classifier",
            model_family="bert_sequence_classification",
            checkpoint_revision="labe_bert_fine_tuned_v1",
            checkpoint_sha256=weights_hash,
            training_data_revision="abcc3ec6032e3b265cbf15c6d8a3da668a2a030675b00f0425b96698c8cd5b56",
            score_scale=[0.0, 1.0],
            threshold=self.threshold,
            threshold_source="labe_validation_split_f1_optimization",
            is_independent=True,
            independent_of=[],
        )

    def _split_sentences(self, text: str) -> List[str]:
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sents if len(s.strip()) > 10]

    def predict_score(self, text: str) -> float:
        sents = self._split_sentences(text)
        if not sents:
            sents = [text]

        scores = []
        with torch.no_grad():
            for s in sents:
                inputs = self.tokenizer(s, return_tensors="pt", truncation=True, max_length=128, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                p_agentic = float(probs[0, 1].item())
                scores.append(p_agentic)

        return float(np.mean(scores)) if scores else 0.50
