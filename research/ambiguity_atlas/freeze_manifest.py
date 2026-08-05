"""Generate cryptographic SHA-256 reproducibility manifest for the study bound to Git commit."""

import os
import sys
import json
import hashlib
import subprocess
from typing import Dict, Any

MANIFEST_PATH = "results/ambiguity_atlas/manifest.json"

FILES_TO_MANIFEST = [
    "data/chaosnli/processed/canonical_items.parquet",
    "results/exploratory/oof_predictions.parquet",
    "research/ambiguity_atlas/configs/atlas_v1.yaml",
    "src/shadowspace/ambiguity_atlas/geometry.py",
    "src/shadowspace/ambiguity_atlas/summaries.py",
    "src/shadowspace/ambiguity_atlas/pair_index.py",
    "src/shadowspace/ambiguity_atlas/posterior.py",
    "src/shadowspace/ambiguity_atlas/retention.py",
    "src/shadowspace/ambiguity_atlas/schemas.py",
    "results/ambiguity_atlas/preflight_report.json",
    "results/ambiguity_atlas/theory_surface.parquet",
    "results/ambiguity_atlas/strict_pairs.parquet",
    "results/ambiguity_atlas/strict_summary.json",
    "results/ambiguity_atlas/approximate_pairs.parquet",
    "results/ambiguity_atlas/posterior_stability.parquet",
    "results/ambiguity_atlas/model_retention.parquet",
    "results/ambiguity_atlas/model_retention_summary.json",
    "results/ambiguity_atlas/atlas_payload.json",
    "docs/viz/ambiguity_atlas/index.html",
]


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of raw byte array."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file on filesystem."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    """Get current HEAD git commit SHA."""
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return res
    except Exception:
        return "UNKNOWN"


def generate_manifest():
    """Generate manifest.json with SHA-256 checksums."""
    print("=== Generating Cryptographic SHA-256 Reproducibility Manifest ===")
    commit_sha = get_git_commit()
    
    file_manifest = {}
    for rel_path in FILES_TO_MANIFEST:
        if os.path.exists(rel_path):
            file_manifest[rel_path] = {
                "size_bytes": os.path.getsize(rel_path),
                "sha256": compute_file_sha256(rel_path),
            }
        else:
            file_manifest[rel_path] = {"status": "MISSING"}
            
    manifest = {
        "manifest_schema": "1.1.0",
        "study_id": "ambiguity_doppelganger_atlas_v1",
        "evidence_level": "exploratory_census",
        "seed": 20260804,
        "source_commit_sha": commit_sha,
        "python_version": sys.version.split()[0],
        "files": file_manifest,
    }

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {MANIFEST_PATH} for commit {commit_sha[:7]} ({len(file_manifest)} files).")


if __name__ == "__main__":
    generate_manifest()
