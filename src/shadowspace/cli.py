"""Command line interface for Shadowspace.

Usage:
    shadowspace generate synthetic [--classes 4] [--seed 20260801] [--output data/bundles/synthetic-v1]
    shadowspace generate calibration [--output data/bundles/calibration-v1]
    shadowspace validate-bundle <bundle_dir>
"""

from __future__ import annotations

import argparse
import sys

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
    synth_parser.add_argument("--classes", type=int, default=4, help="Number of classes (default: 4)")
    synth_parser.add_argument("--seed", type=int, default=20260801, help="Random seed (default: 20260801)")
    synth_parser.add_argument("--samples", type=int, default=2000, help="Number of samples (default: 2000)")
    synth_parser.add_argument(
        "--output",
        type=str,
        default="data/bundles/synthetic-v1",
        help="Output directory path",
    )

    # generate calibration
    calib_parser = gen_subparsers.add_parser("calibration", help="Generate 3-class calibration bundle")
    calib_parser.add_argument(
        "--output",
        type=str,
        default="data/bundles/calibration-v1",
        help="Output directory path",
    )

    # --- validate-bundle ---
    val_parser = subparsers.add_parser("validate-bundle", help="Validate an artifact bundle")
    val_parser.add_argument("bundle_dir", type=str, help="Path to bundle directory")

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
