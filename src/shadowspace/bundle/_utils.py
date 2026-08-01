"""Shared I/O utilities for the bundle package."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
