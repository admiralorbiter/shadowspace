"""E004 Stage 1B GPU Pilot Execution Master Script.

Runs the complete 600-item pilot pipeline:
1. LPE generation (600 items x 6 perms)
2. MCE generation (600 items x 30 reps)
3. LPE extraction & MCE estimation
4. Pilot human support matrix construction (500 Dirichlet draws)
5. 5-fold cross-fitted temperature calibration
6. E004 primary analysis & summary report generation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: List[str]) -> None:
    print(f"\n[RUNNING] {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")

def main():
    parser = argparse.ArgumentParser(description="E004 GPU Pilot Master Runner")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel Ollama HTTP workers")
    parser.add_argument("--model", default="gemma3:12b", help="Ollama model tag")
    args = parser.parse_args()

    print("=========================================================================")
    print("   EXPERIMENT E004 — STAGE 1B GPU PILOT MASTER RUNNER")
    print("=========================================================================")
    print(f"  Model Tag:        {args.model}")
    print(f"  Parallel Workers: {args.workers}")
    print("=========================================================================")

    py = sys.executable

    # 1. Provenance Capture
    run_cmd([py, "research/chaosnli/lab/e004_provenance.py"])

    # 2. LPE Pilot Generation
    run_cmd([py, "research/chaosnli/lab/e004_ollama_runner.py", "--mode", "lpe", "--subset", "pilot", "--workers", str(args.workers), "--model", args.model])

    # 3. MCE Pilot Generation
    run_cmd([py, "research/chaosnli/lab/e004_ollama_runner.py", "--mode", "mce", "--subset", "pilot", "--workers", str(args.workers), "--model", args.model])

    # 4. LPE Extraction & Normalization
    run_cmd([py, "research/chaosnli/lab/e004_lpe_extract.py", "--subset", "pilot"])

    # 5. MCE Estimation
    run_cmd([py, "research/chaosnli/lab/e004_mce_estimate.py", "--subset", "pilot"])

    # 6. Pilot Support Matrix Construction
    run_cmd([py, "research/chaosnli/lab/e004_pilot_support.py", "--subset", "pilot"])

    # 7. Cross-Fitted Calibration
    run_cmd([py, "research/chaosnli/lab/e004_calibrate.py", "--subset", "pilot"])

    # 8. Primary Analysis Engine
    run_cmd([py, "research/chaosnli/lab/e004_analyze.py", "--subset", "pilot"])

    # 9. Summary Markdown Report
    run_cmd([py, "research/chaosnli/lab/e004_postprocess.py"])

    print("\n=========================================================================")
    print("   STAGE 1B PILOT PIPELINE COMPLETE!")
    print("   Summary saved to research/chaosnli/artifacts/E004/summaries/E004_summary.md")
    print("=========================================================================")

if __name__ == "__main__":
    main()
