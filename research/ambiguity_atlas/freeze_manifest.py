"""Generate cryptographic SHA-256 reproducibility manifest bound to Git commit blobs."""

import os
import sys
import json
import hashlib
import subprocess
from typing import Dict, Any

MANIFEST_PATH = "results/ambiguity_atlas/manifest.json"

FILES_TO_MANIFEST = [
    ("data/chaosnli/processed/canonical_items.parquet", "external_file"),
    ("results/exploratory/oof_predictions.parquet", "external_file"),
    ("research/ambiguity_atlas/configs/atlas_v1.yaml", "git_blob"),
    ("src/shadowspace/ambiguity_atlas/geometry.py", "git_blob"),
    ("src/shadowspace/ambiguity_atlas/summaries.py", "git_blob"),
    ("src/shadowspace/ambiguity_atlas/pair_index.py", "git_blob"),
    ("src/shadowspace/ambiguity_atlas/posterior.py", "git_blob"),
    ("src/shadowspace/ambiguity_atlas/retention.py", "git_blob"),
    ("src/shadowspace/ambiguity_atlas/schemas.py", "git_blob"),
    ("results/ambiguity_atlas/preflight_report.json", "git_blob"),
    ("results/ambiguity_atlas/theory_surface.parquet", "git_blob"),
    ("results/ambiguity_atlas/strict_pairs.parquet", "git_blob"),
    ("results/ambiguity_atlas/strict_summary.json", "git_blob"),
    ("results/ambiguity_atlas/approximate_pairs.parquet", "git_blob"),
    ("results/ambiguity_atlas/posterior_stability.parquet", "git_blob"),
    ("results/ambiguity_atlas/model_retention.parquet", "git_blob"),
    ("results/ambiguity_atlas/model_retention_summary.json", "git_blob"),
    ("results/ambiguity_atlas/atlas_payload.json", "git_blob"),
    ("docs/viz/ambiguity_atlas/index.html", "git_blob"),
    ("docs/studies/ambiguity_atlas/PAPER.md", "git_blob"),
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


def read_committed_bytes(commit_sha: str, path: str) -> bytes:
    """Read exact committed blob bytes from Git."""
    return subprocess.check_output(
        ["git", "show", f"{commit_sha}:{path}"],
        stderr=subprocess.DEVNULL
    )


def generate_manifest():
    """Generate manifest.json with SHA-256 checksums bound to Git commit blobs."""
    print("=== Generating Cryptographic SHA-256 Reproducibility Manifest (Git Blob Bound) ===")
    commit_sha = get_git_commit()
    
    file_manifest = {}
    for rel_path, source_type in FILES_TO_MANIFEST:
        if source_type == "git_blob":
            try:
                data = read_committed_bytes(commit_sha, rel_path)
                file_manifest[rel_path] = {
                    "source_type": "git_blob",
                    "size_bytes": len(data),
                    "sha256": compute_bytes_sha256(data),
                }
            except Exception:
                # Fallback to filesystem bytes if not yet committed
                if os.path.exists(rel_path):
                    with open(rel_path, "rb") as f:
                        content = f.read()
                    file_manifest[rel_path] = {
                        "source_type": "git_blob_fallback",
                        "size_bytes": len(content),
                        "sha256": compute_bytes_sha256(content),
                    }
                else:
                    file_manifest[rel_path] = {"status": "MISSING"}
        else:
            if os.path.exists(rel_path):
                file_manifest[rel_path] = {
                    "source_type": "external_file",
                    "size_bytes": os.path.getsize(rel_path),
                    "sha256": compute_file_sha256(rel_path),
                }
            else:
                file_manifest[rel_path] = {"status": "MISSING"}
            
    manifest = {
        "manifest_schema": "1.1.0",
        "study_id": "ambiguity_doppelganger_atlas_v1_1",
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
