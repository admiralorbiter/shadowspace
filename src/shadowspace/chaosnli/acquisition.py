"""Acquisition and checksum verification module for raw ChaosNLI dataset sources."""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHAOSNLI_REPO_URL = "https://github.com/easonnie/ChaosNLI.git"
RAW_FILES = {
    "snli": "chaosNLI_snli.jsonl",
    "mnli": "chaosNLI_mnli_m.jsonl",
    "alphanli": "chaosNLI_alphanli.jsonl",
}
RAW_CDN_BASE = "https://raw.githubusercontent.com/easonnie/ChaosNLI/master"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def acquire_sources(
    raw_dir: Path = Path("data/chaosnli/raw"),
    manifest_path: Path = Path("research/chaosnli/manifests/sources.lock.yaml"),
    force: bool = False,
) -> dict[str, Any]:
    """Acquire ChaosNLI raw JSONL files via git or direct download and compute SHA-256 checksums."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    acquired_files: dict[str, dict[str, Any]] = {}

    # Attempt git clone if git is available, fallback to urllib download
    git_dir = raw_dir / "vendor_ChaosNLI"
    if (force or not any((raw_dir / fn).exists() for fn in RAW_FILES.values())) and not git_dir.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", CHAOSNLI_REPO_URL, str(git_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            pass  # Fallback to direct raw downloads below

    DROPBOX_DATA_URL = "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1"

    # Check if raw files exist directly or inside vendor directory
    for key, filename in RAW_FILES.items():
        target_path = raw_dir / filename
        vendor_file = git_dir / "data" / "chaosNLI_v1.0" / filename
        alt_vendor_file = git_dir / filename

        if vendor_file.exists() and (force or not target_path.exists()):
            target_path.write_bytes(vendor_file.read_bytes())
        elif alt_vendor_file.exists() and (force or not target_path.exists()):
            target_path.write_bytes(alt_vendor_file.read_bytes())
        elif force or not target_path.exists():
            # Try Dropbox direct download first if not present
            try:
                zip_path = raw_dir / "chaosNLI_v1.0.zip"
                if not zip_path.exists():
                    req = urllib.request.Request(DROPBOX_DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_f:
                        out_f.write(resp.read())
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(raw_dir)
                # If extracted into a subfolder chaosNLI_v1.0, move files out
                sub_dir = raw_dir / "chaosNLI_v1.0"
                if sub_dir.exists():
                    for sf in sub_dir.glob("*.jsonl"):
                        (raw_dir / sf.name).write_bytes(sf.read_bytes())
            except Exception:
                # Fallback to direct HTTP fetch from GitHub raw CDN
                url = f"{RAW_CDN_BASE}/{filename}"
                req = urllib.request.Request(url, headers={"User-Agent": "Shadowspace-Research/1.0"})
                with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_f:
                    out_f.write(resp.read())

        if not target_path.exists():
            raise FileNotFoundError(f"Failed to acquire ChaosNLI raw source file: {filename}")

        sha256_hash = compute_sha256(target_path)
        byte_size = target_path.stat().st_size

        # Count lines
        with open(target_path, "r", encoding="utf-8") as f:
            row_count = sum(1 for line in f if line.strip())

        acquired_files[key] = {
            "filename": filename,
            "path": str(target_path),
            "sha256": sha256_hash,
            "size_bytes": byte_size,
            "row_count": row_count,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    manifest_data = {
        "dataset_name": "ChaosNLI",
        "upstream_repo": CHAOSNLI_REPO_URL,
        "files": acquired_files,
    }

    # Save manifest as JSON/YAML readable text
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return manifest_data


def verify_source_checksums(
    raw_dir: Path = Path("data/chaosnli/raw"),
    manifest_path: Path = Path("research/chaosnli/manifests/sources.lock.yaml"),
) -> bool:
    """Verify raw source files against the locked manifest checksums."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest not found at: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_info = manifest.get("files", {})

    all_valid = True
    for key, info in files_info.items():
        filepath = Path(info["path"])
        if not filepath.exists():
            all_valid = False
            continue
        current_hash = compute_sha256(filepath)
        if current_hash != info["sha256"]:
            all_valid = False

    return all_valid
