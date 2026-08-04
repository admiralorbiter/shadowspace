"""Link Full 3,113-Item E001 Hellinger Matrix into E004 Pilot Support Directory.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

E001_BIN = Path("research/chaosnli/artifacts/E001/S_hellinger_k010.bin")
TARGET_BIN = Path("research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_full.bin")
TARGET_META = Path("research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_full.manifest.json")

def main():
    if not E001_BIN.exists():
        raise FileNotFoundError(f"Missing E001 support matrix: {E001_BIN}")

    TARGET_BIN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(E001_BIN, TARGET_BIN)
    print(f"Copied {E001_BIN} -> {TARGET_BIN} ({TARGET_BIN.stat().st_size} bytes)")

    # Full data Q_hh for k=10 Hellinger from E001 is 0.038987226
    meta = {
        "artifact_id": "E004-full-support-v1",
        "object_count": 3113,
        "k": 10,
        "q_hh_relational": 0.038987226212620456,
        "metric": "hellinger",
    }

    with open(TARGET_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Created full metadata manifest at {TARGET_META}")

if __name__ == "__main__":
    main()
