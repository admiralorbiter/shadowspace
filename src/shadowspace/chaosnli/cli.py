"""CLI sub-commands for ChaosNLI pipeline and analysis."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _emit_status(task: str, status: str = "success", **kwargs: Any) -> None:
    """Utility to print structured JSON status for CLI commands."""
    payload = {
        "task": task,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    print(json.dumps(payload, indent=2))


def register_chaosnli_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'chaosnli' command and its subcommand hierarchy into the main CLI parser."""
    c_parser = subparsers.add_parser(
        "chaosnli", help="ChaosNLI pipeline and research workflow commands"
    )
    c_subparsers = c_parser.add_subparsers(dest="chaosnli_command", required=True)

    # 1. fetch
    fetch_p = c_subparsers.add_parser("fetch", help="Acquire and checksum raw ChaosNLI sources")
    fetch_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )
    fetch_p.add_argument("--force", action="store_true", help="Force redownload source repositories")

    # 2. verify-sources
    ver_p = c_subparsers.add_parser("verify-sources", help="Verify raw source file SHA-256 checksums")
    ver_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 3. normalize
    norm_p = c_subparsers.add_parser("normalize", help="Parse and normalize raw records to canonical Parquet")
    norm_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 4. audit-joins
    audit_p = c_subparsers.add_parser("audit-joins", help="Audit joins with external taxonomy/explanation sets")
    audit_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 5. human-posterior
    post_p = c_subparsers.add_parser("human-posterior", help="Estimate Dirichlet posteriors for human distributions")
    post_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )
    post_p.add_argument("--draws", type=int, default=2000, help="Number of Dirichlet Monte Carlo draws")

    # 6. predict
    pred_p = c_subparsers.add_parser("predict", help="Generate or load model logits for items")
    pred_p.add_argument("--model", type=str, required=True, help="Model identity key (e.g., roberta-large)")
    pred_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 7. calibrate
    cal_p = c_subparsers.add_parser("calibrate", help="Fit temperature scaling on separate calibration split")
    cal_p.add_argument("--model", type=str, required=True, help="Model identity key")
    cal_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 8. build-spaces
    space_p = c_subparsers.add_parser("build-spaces", help="Construct probability representations and distance matrices")
    space_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 9. compute-neighbors
    knn_p = c_subparsers.add_parser("compute-neighbors", help="Compute exact k-NN graph tables")
    knn_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 10. compare-graphs
    comp_p = c_subparsers.add_parser("compare-graphs", help="Compare model graphs against human reference graphs")
    comp_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 11. analyze
    ana_p = c_subparsers.add_parser("analyze", help="Run locked hypothesis statistical testing plan")
    ana_p.add_argument(
        "--plan", type=str, default="research/chaosnli/configs/analysis.yaml", help="Analysis lock path"
    )

    # 12. select-cases
    sel_p = c_subparsers.add_parser("select-cases", help="Automatically select review case items and controls")
    sel_p.add_argument(
        "--packet", type=str, default="research/chaosnli/configs/review_packets.yaml", help="Packet config path"
    )

    # 13. render-packets
    ren_p = c_subparsers.add_parser("render-packets", help="Render HTML/PNG review packets")
    ren_p.add_argument(
        "--packet", type=str, default="research/chaosnli/configs/review_packets.yaml", help="Packet config path"
    )

    # 14. import-codings
    imp_p = c_subparsers.add_parser("import-codings", help="Import blinded human coding records")
    imp_p.add_argument("coding_file", type=str, help="Path to coding CSV/Parquet file")

    # 15. build-bundle
    bun_p = c_subparsers.add_parser("build-bundle", help="Build Shadowspace visualization bundle from frozen outputs")
    bun_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 16. report
    rep_p = c_subparsers.add_parser("report", help="Generate final markdown analysis report")
    rep_p.add_argument(
        "--manifest", type=str, default="research/chaosnli/configs/study.yaml", help="Study manifest path"
    )

    # 17. verify-release
    vr_p = c_subparsers.add_parser("verify-release", help="Verify release integrity and estimate reproducibility")
    vr_p.add_argument("release_dir", type=str, help="Path to release directory")


def handle_chaosnli_command(parsed_args: argparse.Namespace) -> int:
    """Execute the specified chaosnli subcommand."""
    cmd = parsed_args.chaosnli_command

    # Placeholder handlers for skeleton CLI. As modules are implemented, handlers will call module logic.
    if cmd == "fetch":
        manifest_path = Path(parsed_args.manifest)
        _emit_status("chaosnli.fetch", message=f"Fetch requested using manifest: {manifest_path}")
        return 0

    elif cmd == "verify-sources":
        _emit_status("chaosnli.verify-sources", message="Source verification requested.")
        return 0

    elif cmd == "normalize":
        _emit_status("chaosnli.normalize", message="Record normalization requested.")
        return 0

    elif cmd == "audit-joins":
        _emit_status("chaosnli.audit-joins", message="External join audit requested.")
        return 0

    elif cmd == "human-posterior":
        _emit_status("chaosnli.human-posterior", message="Dirichlet posterior estimation requested.")
        return 0

    elif cmd == "predict":
        _emit_status("chaosnli.predict", model=parsed_args.model, message="Model prediction requested.")
        return 0

    elif cmd == "calibrate":
        _emit_status("chaosnli.calibrate", model=parsed_args.model, message="Model calibration requested.")
        return 0

    elif cmd == "build-spaces":
        _emit_status("chaosnli.build-spaces", message="Space construction requested.")
        return 0

    elif cmd == "compute-neighbors":
        _emit_status("chaosnli.compute-neighbors", message="k-NN computation requested.")
        return 0

    elif cmd == "compare-graphs":
        _emit_status("chaosnli.compare-graphs", message="Graph comparison requested.")
        return 0

    elif cmd == "analyze":
        _emit_status("chaosnli.analyze", message="Statistical analysis requested.")
        return 0

    elif cmd == "select-cases":
        _emit_status("chaosnli.select-cases", message="Case selection requested.")
        return 0

    elif cmd == "render-packets":
        _emit_status("chaosnli.render-packets", message="Review packet rendering requested.")
        return 0

    elif cmd == "import-codings":
        _emit_status("chaosnli.import-codings", file=parsed_args.coding_file, message="Import codings requested.")
        return 0

    elif cmd == "build-bundle":
        _emit_status("chaosnli.build-bundle", message="Bundle build requested.")
        return 0

    elif cmd == "report":
        _emit_status("chaosnli.report", message="Report generation requested.")
        return 0

    elif cmd == "verify-release":
        _emit_status("chaosnli.verify-release", release_dir=parsed_args.release_dir, message="Release verification requested.")
        return 0

    print(f"Unknown chaosnli subcommand '{cmd}'", file=sys.stderr)
    return 1
