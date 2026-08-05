"""Freeze & Serialize LABE Evaluator Checkpoints (N-Gram Ensemble & BERT Transformer)."""

from __future__ import annotations

import hashlib
import json
import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
        "best_threshold": artifacts["best_threshold"],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print(f"Sparse N-Gram Artifacts Frozen to {out_dir}")
    return hashes


def freeze_bert_transformer_checkpoint(
    out_dir: str = "models/labe_bert_agency",
    epochs: int = 1,
    lr: float = 2e-5,
) -> dict[str, str]:
    """Fine-tunes BERT on LABE train split (N=2,979) with fixed PyTorch seed and serializes checkpoint."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labe_data = load_labe_dataset()
    train_sents = labe_data["sentences_by_split"]["train"]
    val_sents = labe_data["sentences_by_split"]["validation"]

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_dataset = LABEDataset(
        [s["text"] for s in train_sents],
        [s["label_int"] for s in train_sents],
        tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    print(f"Fine-tuning {model_name} on {len(train_sents)} LABE train sentences...")
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

    # Save fine-tuned checkpoint
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    weights_path = os.path.join(out_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(out_dir, "pytorch_model.bin")

    manifest = {
        "model_name": model_name,
        "checkpoint_revision": "labe_bert_fine_tuned_v1",
        "weights_sha256": compute_file_sha256(weights_path) if os.path.exists(weights_path) else "saved",
        "threshold": 0.50,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"LABE Fine-Tuned BERT Checkpoint Frozen to {out_dir}")
    return manifest


if __name__ == "__main__":
    freeze_ngram_artifacts()
    freeze_bert_transformer_checkpoint()
