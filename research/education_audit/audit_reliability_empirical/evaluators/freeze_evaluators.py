"""Freeze & Serialize LABE Evaluator Checkpoints (N-Gram Ensemble & Fine-Tuned BERT Transformer)."""

from __future__ import annotations

import hashlib
import json
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

from research.education_audit.agency_classifier_validation.labe_classifier_trainer import train_and_evaluate_labe_classifier
from research.education_audit.external_validation.labe_loader import load_labe_dataset


def compute_file_sha256(filepath: str) -> str:
    """Computes exact 64-character SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class LABEDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def freeze_ngram_artifacts(out_dir: str = "models/labe_sparse_ngram") -> dict[str, str]:
    """Trains and serializes the sparse n-gram model artifacts."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _, artifacts = train_and_evaluate_labe_classifier()

    vec_path = os.path.join(out_dir, "vectorizer.joblib")
    lr_path = os.path.join(out_dir, "clf_lr.joblib")
    gb_path = os.path.join(out_dir, "clf_gb.joblib")

    joblib.dump(artifacts["vectorizer"], vec_path)
    joblib.dump(artifacts["clf_lr"], lr_path)
    joblib.dump(artifacts["clf_gb"], gb_path)

    hashes = {
        "vectorizer_sha256": compute_file_sha256(vec_path),
        "clf_lr_sha256": compute_file_sha256(lr_path),
        "clf_gb_sha256": compute_file_sha256(gb_path),
        "best_threshold": float(artifacts["best_threshold"]),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print(f"Sparse N-Gram Artifacts Frozen to {out_dir}")
    return hashes


def freeze_bert_transformer_checkpoint(
    out_dir: str = "models/labe_bert_agency",
    epochs: int = 2,
    lr: float = 2e-5,
) -> dict[str, Any]:
    """Fine-tunes BERT on LABE train split (N=2,979), optimizes threshold on validation (N=372), and records test metrics."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labe_data = load_labe_dataset()
    train_sents = labe_data["sentences_by_split"]["train"]
    val_sents = labe_data["sentences_by_split"]["validation"]
    test_sents = labe_data["sentences_by_split"]["test"]

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_dataset = LABEDataset([s["text"] for s in train_sents], [s["label_int"] for s in train_sents], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    print(f"Fine-tuning {model_name} on {len(train_sents)} LABE train sentences for {epochs} epochs...")
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()

    # Predict probabilities on Validation split (N=372)
    def _get_probs(sentences):
        probs_list = []
        with torch.no_grad():
            for s in sentences:
                inputs = tokenizer(s["text"], return_tensors="pt", truncation=True, max_length=128, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                p = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
                probs_list.append(p)
        return np.array(probs_list)

    val_probs = _get_probs(val_sents)
    val_labels = np.array([s["label_int"] for s in val_sents])

    # Find threshold optimizing F1 on Validation split
    best_th = 0.50
    best_f1 = 0.0
    for th in np.arange(0.20, 0.80, 0.02):
        th = round(float(th), 2)
        preds = (val_probs >= th).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    # Evaluate once on locked Test split (N=373)
    test_probs = _get_probs(test_sents)
    test_labels = np.array([s["label_int"] for s in test_sents])
    test_preds = (test_probs >= best_th).astype(int)

    test_metrics = {
        "precision": round(float(precision_score(test_labels, test_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(test_labels, test_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(test_labels, test_preds, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(test_labels, test_probs)), 4),
        "brier_score": round(float(brier_score_loss(test_labels, test_probs)), 4),
        "log_loss": round(float(log_loss(test_labels, test_probs)), 4),
        "validation_optimal_threshold": best_th,
    }

    # Save fine-tuned model checkpoint & tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    weights_path = os.path.join(out_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(out_dir, "pytorch_model.bin")

    weights_sha256 = compute_file_sha256(weights_path)

    manifest = {
        "model_name": model_name,
        "checkpoint_revision": "labe_bert_fine_tuned_v1",
        "weights_sha256": weights_sha256,
        "config_sha256": compute_file_sha256(os.path.join(out_dir, "config.json")),
        "tokenizer_sha256": compute_file_sha256(os.path.join(out_dir, "tokenizer.json")),
        "best_threshold": best_th,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(out_dir, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "LABE Fine-Tuned BERT Agency Classifier",
            "base_model": model_name,
            "training_split_size": len(train_sents),
            "validation_split_size": len(val_sents),
            "test_split_size": len(test_sents),
            "locked_test_metrics": test_metrics,
            "weights_sha256": weights_sha256,
        }, f, indent=2)

    print(f"LABE Fine-Tuned BERT Checkpoint Frozen to {out_dir}")
    print(f"  - Validation Optimal Threshold: {best_th}")
    print(f"  - Locked Test F1: {test_metrics['f1_score']} | AUROC: {test_metrics['auroc']}")
    return manifest


if __name__ == "__main__":
    freeze_ngram_artifacts()
    freeze_bert_transformer_checkpoint()
