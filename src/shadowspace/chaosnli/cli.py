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
        from shadowspace.chaosnli.acquisition import acquire_sources

        manifest_path = Path(parsed_args.manifest)
        force = getattr(parsed_args, "force", False)
        res = acquire_sources(manifest_path=manifest_path, force=force)
        _emit_status("chaosnli.fetch", manifest=str(manifest_path), acquired=res["files"])
        return 0

    elif cmd == "verify-sources":
        from shadowspace.chaosnli.acquisition import verify_source_checksums

        manifest_path = Path(parsed_args.manifest)
        valid = verify_source_checksums(manifest_path=manifest_path)
        status = "success" if valid else "failed"
        _emit_status("chaosnli.verify-sources", status=status, valid=valid)
        return 0 if valid else 1

    elif cmd == "normalize":
        from shadowspace.chaosnli.normalize import normalize_dataset

        res = normalize_dataset()
        _emit_status("chaosnli.normalize", summary=res)
        return 0

    elif cmd == "audit-joins":
        _emit_status("chaosnli.audit-joins", status="not_implemented", message="Command not yet implemented.")
        return 2

    elif cmd == "human-posterior":
        from shadowspace.chaosnli.posterior import run_posterior_pipeline

        n_draws = getattr(parsed_args, "draws", 2000)
        res = run_posterior_pipeline(n_draws=n_draws)
        _emit_status("chaosnli.human-posterior", summary=res)
        return 0

    elif cmd == "predict":
        _emit_status("chaosnli.predict", status="not_implemented", model=parsed_args.model, message="Command not yet implemented.")
        return 2

    elif cmd == "calibrate":
        _emit_status("chaosnli.calibrate", status="not_implemented", model=parsed_args.model, message="Command not yet implemented.")
        return 2

    elif cmd == "build-spaces":
        from shadowspace.chaosnli.distances import build_distance_matrix
        import numpy as np
        import polars as pl

        proc_dir = Path("data/chaosnli/processed")
        canon_p = proc_dir / "canonical_items_posterior.parquet"
        if not canon_p.exists():
            canon_p = proc_dir / "canonical_items.parquet"

        df = pl.read_parquet(canon_p)
        prob_cols = ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
        p_matrix = df.select(prob_cols).to_numpy()

        metrics_built = []
        for m in ["hellinger", "jensen_shannon", "total_variation", "euclidean", "aitchison"]:
            dist_mat = build_distance_matrix(p_matrix, metric=m)
            out_file = proc_dir / f"distance_matrix_human_{m}.npy"
            np.save(out_file, dist_mat)
            metrics_built.append(m)

        # Build text embedding distance matrix
        from shadowspace.chaosnli.text_embeddings import build_text_distance_space
        text_info = build_text_distance_space(canonical_items_path=canon_p, output_dir=proc_dir)
        metrics_built.append("text_cosine")

        # Build joint distance matrix (lambda = 0.5)
        from shadowspace.chaosnli.joint_spaces import compute_joint_distance_matrix
        d_hellinger = np.load(proc_dir / "distance_matrix_human_hellinger.npy")
        d_text = np.load(proc_dir / "distance_matrix_text_cosine.npy")
        d_joint = compute_joint_distance_matrix(d_hellinger, d_text, lambda_weight=0.5)
        np.save(proc_dir / "distance_matrix_joint_lambda050.npy", d_joint)
        metrics_built.append("joint_lambda050")

        _emit_status("chaosnli.build-spaces", n_items=len(df), metrics=metrics_built, text_embedding_dim=text_info["embedding_dim"])
        return 0

    elif cmd == "compute-neighbors":
        from shadowspace.chaosnli.distances import build_distance_matrix
        from shadowspace.chaosnli.neighbors import extract_knn_graph, save_knn_graph
        import numpy as np
        import polars as pl

        proc_dir = Path("data/chaosnli/processed")
        canon_p = proc_dir / "canonical_items_posterior.parquet"
        if not canon_p.exists():
            canon_p = proc_dir / "canonical_items.parquet"

        df = pl.read_parquet(canon_p)
        ids = df["object_id"].to_list()
        prob_cols = ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
        p_matrix = df.select(prob_cols).to_numpy()

        saved_files = []
        for k_val in [5, 10, 20, 50]:
            for metric in ["hellinger", "jensen_shannon", "euclidean"]:
                dist_file = proc_dir / f"distance_matrix_human_{metric}.npy"
                if dist_file.exists():
                    dist_mat = np.load(dist_file)
                else:
                    dist_mat = build_distance_matrix(p_matrix, metric=metric)

                _, neighbor_df = extract_knn_graph(dist_mat, ids, k=k_val, space_id="human_opinion", metric_id=metric)
                out_p = save_knn_graph(neighbor_df, output_dir=proc_dir, space_id="human_opinion", metric_id=metric, k=k_val)
                saved_files.append(str(out_p.name))

        _emit_status("chaosnli.compute-neighbors", count=len(saved_files), files=saved_files[:4])
        return 0

    elif cmd == "compare-graphs":
        from shadowspace.chaosnli.graph_metrics import compute_human_split_half_reliability
        import polars as pl

        proc_dir = Path("data/chaosnli/processed")
        canon_p = proc_dir / "canonical_items_posterior.parquet"
        if not canon_p.exists():
            canon_p = proc_dir / "canonical_items.parquet"

        df = pl.read_parquet(canon_p)
        counts = df.select(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).to_numpy()

        reliability_res = compute_human_split_half_reliability(counts, k=10, n_repetitions=20, metric="hellinger")
        _emit_status("chaosnli.compare-graphs", reliability=reliability_res)
        return 0

    elif cmd == "analyze":
        from shadowspace.chaosnli.models import build_canonical_models_table, load_model_predictions
        from shadowspace.chaosnli.model_topology import evaluate_model_topology_recovery
        import polars as pl

        proc_dir = Path("data/chaosnli/processed")
        canon_p = proc_dir / "canonical_items_posterior.parquet"
        if not canon_p.exists():
            canon_p = proc_dir / "canonical_items.parquet"

        df = pl.read_parquet(canon_p)
        model_results = load_model_predictions()
        model_df = build_canonical_models_table(model_results, canonical_items_path=canon_p)

        eval_res = evaluate_model_topology_recovery(model_results, canonical_items_path=df)
        _emit_status("chaosnli.analyze", n_models=len(model_results), evaluations=eval_res)
        return 0

    elif cmd == "select-cases":
        _emit_status("chaosnli.select-cases", status="not_implemented", message="Command not yet implemented.")
        return 2

    elif cmd == "render-packets":
        _emit_status("chaosnli.render-packets", status="not_implemented", message="Command not yet implemented.")
        return 2

    elif cmd == "import-codings":
        _emit_status("chaosnli.import-codings", status="not_implemented", file=parsed_args.coding_file, message="Command not yet implemented.")
        return 2

    elif cmd == "build-bundle":
        _emit_status("chaosnli.build-bundle", status="not_implemented", message="Command not yet implemented.")
        return 2

    elif cmd == "report":
        _emit_status("chaosnli.report", status="not_implemented", message="Command not yet implemented.")
        return 2

    elif cmd == "verify-release":
        _emit_status("chaosnli.verify-release", status="not_implemented", release_dir=parsed_args.release_dir, message="Command not yet implemented.")
        return 2

    print(f"Unknown chaosnli subcommand '{cmd}'", file=sys.stderr)
    return 1
