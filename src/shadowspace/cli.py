"""Command line interface for Shadowspace.

Usage:
    shadowspace generate synthetic [--classes 4] [--seed 20260801] [--output data/bundles/synthetic-v1]
    shadowspace generate calibration [--output data/bundles/calibration-v1]
    shadowspace validate-bundle <bundle_dir>
    shadowspace datasets list [--bundle-dir data/bundles]
    shadowspace datasets fetch [--datasets all] [--output data/bundles] [--seed 20260801] [--force] [--include-downloads]
    shadowspace datasets info <key>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shadowspace.bundle.reader import BundleValidator
from shadowspace.generators.calibration import generate_calibration_bundle
from shadowspace.generators.synthetic import generate_synthetic_bundle


def main(args: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(prog="shadowspace", description="Shadowspace CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic artifact bundles")
    gen_subparsers = gen_parser.add_subparsers(dest="target", required=True)

    # generate synthetic
    synth_parser = gen_subparsers.add_parser("synthetic", help="Generate 4-class synthetic bundle")
    synth_parser.add_argument(
        "--classes", type=int, default=4, help="Number of classes (default: 4)"
    )
    synth_parser.add_argument(
        "--seed", type=int, default=20260801, help="Random seed (default: 20260801)"
    )
    synth_parser.add_argument(
        "--samples", type=int, default=2000, help="Number of samples (default: 2000)"
    )
    synth_parser.add_argument(
        "--output",
        type=str,
        default="data/bundles/synthetic-v1",
        help="Output directory path",
    )

    # generate calibration
    calib_parser = gen_subparsers.add_parser(
        "calibration", help="Generate 3-class calibration bundle"
    )
    calib_parser.add_argument(
        "--output",
        type=str,
        default="data/bundles/calibration-v1",
        help="Output directory path",
    )

    # --- validate-bundle ---
    val_parser = subparsers.add_parser("validate-bundle", help="Validate an artifact bundle")
    val_parser.add_argument("bundle_dir", type=str, help="Path to bundle directory")

    # --- import-csv ---
    import_csv_p = subparsers.add_parser(
        "import-csv", help="Import a CSV file into a Shadowspace bundle"
    )
    import_csv_p.add_argument("--input", type=str, required=True, help="Input CSV path")
    import_csv_p.add_argument("--output", type=str, required=True, help="Output directory path")
    import_csv_p.add_argument("--id-col", type=str, default=None, help="Name of ID column")
    import_csv_p.add_argument(
        "--label-col", type=str, default=None, help="Name of true label column"
    )
    import_csv_p.add_argument(
        "--feature-cols", type=str, default=None, help="Comma-separated feature column names"
    )
    import_csv_p.add_argument(
        "--normalize", action="store_true", help="Apply softmax row-wise (for raw logits)"
    )
    import_csv_p.add_argument(
        "--name", type=str, default="imported_dataset", help="Dataset name identifier"
    )

    # --- import-parquet ---
    import_pq_p = subparsers.add_parser(
        "import-parquet", help="Import a Parquet file into a Shadowspace bundle"
    )
    import_pq_p.add_argument("--input", type=str, required=True, help="Input Parquet path")
    import_pq_p.add_argument("--output", type=str, required=True, help="Output directory path")
    import_pq_p.add_argument("--id-col", type=str, default=None, help="Name of ID column")
    import_pq_p.add_argument(
        "--label-col", type=str, default=None, help="Name of true label column"
    )
    import_pq_p.add_argument(
        "--feature-cols", type=str, default=None, help="Comma-separated feature column names"
    )
    import_pq_p.add_argument(
        "--normalize", action="store_true", help="Apply softmax row-wise (for raw logits)"
    )
    import_pq_p.add_argument(
        "--name", type=str, default="imported_dataset", help="Dataset name identifier"
    )

    # --- datasets ---
    ds_parser = subparsers.add_parser("datasets", help="Manage benchmark datasets")
    ds_subparsers = ds_parser.add_subparsers(dest="subcommand", required=True)

    # datasets list
    ds_list = ds_subparsers.add_parser("list", help="List registered benchmark datasets")
    ds_list.add_argument(
        "--bundle-dir", type=str, default="data/bundles", help="Bundles root directory"
    )

    # datasets fetch
    ds_fetch = ds_subparsers.add_parser(
        "fetch", help="Fetch benchmark datasets and generate bundles"
    )
    ds_fetch.add_argument(
        "--datasets", type=str, default="all", help="Comma-separated dataset keys or 'all'"
    )
    ds_fetch.add_argument(
        "--output", type=str, default="data/bundles", help="Bundles output directory"
    )
    ds_fetch.add_argument("--seed", type=int, default=20260801, help="Random seed")
    ds_fetch.add_argument("--force", action="store_true", help="Re-fetch even if bundle exists")
    ds_fetch.add_argument(
        "--include-downloads",
        action="store_true",
        help="Include datasets requiring internet download",
    )

    # datasets info
    ds_info = ds_subparsers.add_parser("info", help="Get details for a benchmark dataset")
    ds_info.add_argument("key", type=str, help="Dataset key")

    # --- chaosnli ---
    from shadowspace.chaosnli.cli import register_chaosnli_subparser

    register_chaosnli_subparser(subparsers)

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "generate":
        if parsed_args.target == "synthetic":
            out_manifest = generate_synthetic_bundle(
                output_dir=parsed_args.output,
                n_classes=parsed_args.classes,
                seed=parsed_args.seed,
                n_samples=parsed_args.samples,
            )
            print(f"Successfully generated synthetic bundle at: {out_manifest.parent}")
            return 0
        elif parsed_args.target == "calibration":
            out_manifest = generate_calibration_bundle(output_dir=parsed_args.output)
            print(f"Successfully generated calibration bundle at: {out_manifest.parent}")
            return 0

    elif parsed_args.command == "validate-bundle":
        validator = BundleValidator(parsed_args.bundle_dir)
        res = validator.validate()
        if res.is_valid:
            print(f"[OK] Bundle at {parsed_args.bundle_dir} is VALID.")
            for w in res.warnings:
                print(f"  Warning: {w}")
            return 0
        else:
            print(f"[FAIL] Bundle at {parsed_args.bundle_dir} is INVALID:")
            for e in res.errors:
                print(f"  Error: {e}")
            return 1

    elif parsed_args.command in ("import-csv", "import-parquet"):
        from shadowspace.importers.csv_importer import import_csv_bundle, import_parquet_bundle

        feat_cols = (
            [c.strip() for c in parsed_args.feature_cols.split(",")]
            if parsed_args.feature_cols
            else None
        )
        if parsed_args.command == "import-csv":
            manifest = import_csv_bundle(
                csv_path=parsed_args.input,
                output_dir=parsed_args.output,
                id_column=parsed_args.id_col,
                label_column=parsed_args.label_col,
                feature_columns=feat_cols,
                normalize=parsed_args.normalize,
                dataset_name=parsed_args.name,
            )
        else:
            manifest = import_parquet_bundle(
                parquet_path=parsed_args.input,
                output_dir=parsed_args.output,
                id_column=parsed_args.id_col,
                label_column=parsed_args.label_col,
                feature_columns=feat_cols,
                normalize=parsed_args.normalize,
                dataset_name=parsed_args.name,
            )
        print(f"Successfully imported bundle to: {manifest.parent}")
        return 0

    elif parsed_args.command == "datasets":
        from shadowspace.datasets.bundle_discovery import scan_bundle_dir
        from shadowspace.datasets.fetchers.sklearn_datasets import fetch_dataset
        from shadowspace.datasets.registry import REGISTRY

        if parsed_args.subcommand == "list":
            bundle_dir = Path(parsed_args.bundle_dir)
            existing = scan_bundle_dir(bundle_dir)
            print("Shadowspace Benchmark Datasets:")
            print("-" * 68)
            for key, spec in REGISTRY.items():
                status = "[OK: fetched]" if key in existing else "[AVAILABLE]"
                dl_flag = " (internet download required)" if spec.requires_download else ""
                print(f"  {key:<20} {spec.display_name:<38} {status:<14}{dl_flag}")
            return 0

        elif parsed_args.subcommand == "info":
            key = parsed_args.key
            if key not in REGISTRY:
                print(f"Error: Unknown dataset key '{key}'.", file=sys.stderr)
                return 1
            spec = REGISTRY[key]
            print(f"Dataset Key   : {spec.key}")
            print(f"Display Name  : {spec.display_name}")
            print(f"Classes (K)   : {spec.n_classes}")
            print(f"Requires DL   : {spec.requires_download}")
            print(f"Description   : {spec.description}")
            return 0

        elif parsed_args.subcommand == "fetch":
            out_dir = Path(parsed_args.output)
            if parsed_args.datasets == "all":
                keys_to_fetch = [
                    k
                    for k, v in REGISTRY.items()
                    if not v.requires_download or parsed_args.include_downloads
                ]
            else:
                keys_to_fetch = [k.strip() for k in parsed_args.datasets.split(",") if k.strip()]

            print(f"Fetching {len(keys_to_fetch)} dataset(s) into {out_dir}...")
            for key in keys_to_fetch:
                if key not in REGISTRY:
                    print(f"  [SKIP] Unknown dataset key '{key}'", file=sys.stderr)
                    continue
                spec = REGISTRY[key]
                if spec.requires_download and not parsed_args.include_downloads:
                    print(f"  [SKIP] '{key}' requires download (pass --include-downloads to fetch)")
                    continue
                try:
                    manifest_p = fetch_dataset(
                        key=key,
                        output_dir=out_dir,
                        seed=parsed_args.seed,
                        force=parsed_args.force,
                    )
                    print(f"  [OK] {key} -> {manifest_p.parent}")
                except Exception as err:
                    print(f"  [FAIL] {key}: {err}", file=sys.stderr)
            return 0

    elif parsed_args.command == "chaosnli":
        from shadowspace.chaosnli.cli import handle_chaosnli_command

        return handle_chaosnli_command(parsed_args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
