"""Verify reproducibility manifest hashes against committed Git blobs and external files."""

import os
import sys
import json
import hashlib
import subprocess

MANIFEST_PATH = "results/ambiguity_atlas/manifest.json"


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


def read_committed_bytes(commit_sha: str, path: str) -> bytes:
    """Read exact committed blob bytes from Git."""
    return subprocess.check_output(
        ["git", "show", f"{commit_sha}:{path}"],
        stderr=subprocess.DEVNULL
    )


def verify_manifest():
    """Verify all files listed in manifest match exact size and SHA-256 hash against Git blobs or filesystem."""
    print("=== Verifying Ambiguity Doppelgänger Atlas Manifest (Git & Filesystem) ===")
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"FAIL: Manifest file not found at {MANIFEST_PATH}")
        sys.exit(1)
        
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    commit_sha = manifest.get("source_commit_sha", "HEAD")
    files_dict = manifest.get("files", {})
    
    if not files_dict:
        print("FAIL: Manifest contains no files dictionary")
        sys.exit(1)

    # Check Git commit validity
    try:
        subprocess.run(["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_git_commit = True
    except Exception:
        has_git_commit = False

    all_passed = True
    verified_count = 0

    for rel_path, meta in files_dict.items():
        source_type = meta.get("source_type", "external_file")
        expected_hash = meta.get("sha256")
        expected_size = meta.get("size_bytes")
        
        if source_type in {"git_blob", "git_blob_fallback"} and has_git_commit:
            try:
                data = read_committed_bytes(commit_sha, rel_path)
                actual_hash = compute_bytes_sha256(data)
                actual_size = len(data)
            except Exception:
                if os.path.exists(rel_path):
                    actual_hash = compute_file_sha256(rel_path)
                    actual_size = os.path.getsize(rel_path)
                else:
                    print(f"[FAIL] Missing file: {rel_path}")
                    all_passed = False
                    continue
        else:
            if os.path.exists(rel_path):
                actual_hash = compute_file_sha256(rel_path)
                actual_size = os.path.getsize(rel_path)
            else:
                print(f"[FAIL] Missing file: {rel_path}")
                all_passed = False
                continue

        if actual_hash != expected_hash:
            print(f"[FAIL] Hash mismatch for {rel_path}:\n  Expected: {expected_hash}\n  Actual:   {actual_hash}")
            all_passed = False
        elif actual_size != expected_size:
            print(f"[FAIL] Size mismatch for {rel_path}:\n  Expected: {expected_size}\n  Actual:   {actual_size}")
            all_passed = False
        else:
            verified_count += 1

    if all_passed:
        print(f"VERIFICATION PASSED: All {verified_count} tracked files match manifest sha256 checksums (Git Commit: {commit_sha[:7]}).")
        sys.exit(0)
    else:
        print(f"VERIFICATION FAILED: One or more files failed checksum verification.")
        sys.exit(1)


if __name__ == "__main__":
    verify_manifest()
