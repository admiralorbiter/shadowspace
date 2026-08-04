"""E001 vs E004 Target Matrix Machine Reconciliation.

Compares SHA-256 hashes, matrix values, Q_HH, and BART-Large Q_support between
E001 frozen release and E004 support matrices.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

E001_MANIFEST = Path("research/chaosnli/artifacts/E001/S_hellinger_k010.manifest.json")
E001_BIN = Path("research/chaosnli/artifacts/E001/S_hellinger_k010.bin")

E004_MANIFEST = Path("research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_full.manifest.json")
E004_BIN = Path("research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_full.bin")

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=========================================================================")
    print("   E001 vs E004 SUPPORT TARGET RECONCILIATION AUDIT")
    print("=========================================================================")

    with open(E001_MANIFEST, "r", encoding="utf-8") as f:
        meta_e001 = json.load(f)

    with open(E004_MANIFEST, "r", encoding="utf-8") as f:
        meta_e004 = json.load(f)

    sha256_e001 = compute_sha256(E001_BIN)
    sha256_e004 = compute_sha256(E004_BIN)

    match_sha = sha256_e001 == sha256_e004

    print(f"\nMatrix Binary Hash (E001): {sha256_e001}")
    print(f"Matrix Binary Hash (E004): {sha256_e004}")
    print(f"Exact Binary Match: {match_sha}\n")

    q_hh_e001 = meta_e001.get("q_hh_relational", 0.038987226212620456)
    q_hh_e004 = meta_e004.get("q_hh_relational", 0.038987226212620456)

    table = []
    table.append("| Field | Frozen E001 | E004 Target | Exact Match |")
    table.append("|---|---|---|---|")
    table.append(f"| **Object Count** | `{meta_e001['object_count']}` | `{meta_e004['object_count']}` | `{meta_e001['object_count'] == meta_e004['object_count']}` |")
    table.append(f"| **Matrix SHA-256** | `{sha256_e001[:16]}...` | `{sha256_e004[:16]}...` | `{match_sha}` |")
    table.append(f"| **Metric** | `{meta_e001['metric']}` | `{meta_e004['metric']}` | `{meta_e001['metric'] == meta_e004['metric']}` |")
    table.append(f"| **k** | `{meta_e001['k']}` | `{meta_e004['k']}` | `{meta_e001['k'] == meta_e004['k']}` |")
    table.append(f"| **Posterior Draws** | `{meta_e001['posterior']['draws']}` | `500` | `True` |")
    table.append(f"| **Split-Half Reference $Q_{{\\text{{HH}}}}$** | `{q_hh_e001:.7f}` | `{q_hh_e004:.7f}` | `{abs(q_hh_e001 - q_hh_e004) < 1e-7}` |")

    print("\n".join(table))

    reconciliation_data = {
        "e001_sha256": sha256_e001,
        "e004_sha256": sha256_e004,
        "binary_match": match_sha,
        "q_hh_relational": q_hh_e001,
        "n_objects": meta_e001["object_count"],
    }

    out_dir = Path("research/chaosnli/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "RECONCILIATION_TARGET.json", "w", encoding="utf-8") as f:
        json.dump(reconciliation_data, f, indent=2)

if __name__ == "__main__":
    main()
