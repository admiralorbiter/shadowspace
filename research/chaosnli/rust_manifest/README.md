# ChaosNLI Rust Analysis Engine

This crate runs the paired-estimand and multi-seed reference-surface analyses used by the
ChaosNLI study.

Run it from the Shadowspace repository root so the default repository-relative paths resolve:

```bash
cargo run --release --locked --manifest-path research/chaosnli/rust_manifest/Cargo.toml
```

The generated model-probability input is intentionally ignored by Git. The default inputs are:

- `data/chaosnli/processed/canonical_items_posterior.json`
- `research/chaosnli/rust_manifest/model_probs.json`

The default outputs are ignored recomputation artifacts:

- `research/chaosnli/artifacts/paired_estimand_results.json`
- `research/chaosnli/artifacts/multi_seed_reference_surface.json`

They do not compete with the single committed release source,
`results/canonical_results.json`. Promote audited recomputation outputs with
`research/chaosnli/manifests/promote_canonical_release.py`.

Every path can be overridden, which also permits running the binary from another directory:

```bash
cargo run --release --locked \
  --manifest-path research/chaosnli/rust_manifest/Cargo.toml \
  -- \
  --items /path/to/canonical_items_posterior.json \
  --models /path/to/model_probs.json \
  --paired-output /path/to/paired_estimand_results.json \
  --surface-output /path/to/multi_seed_reference_surface.json
```

Use `--help` to list the available path options without running an analysis.

## Phase diagram only

The independent `phase_diagram` binary recomputes the complete boundary-tie phase grid
without rerunning paired-estimand or reference-surface analyses. It reads the existing
canonical Hellinger distance matrix for the empirical ChaosNLI reference:

```bash
cargo run --release --locked \
  --manifest-path research/chaosnli/rust_manifest/Cargo.toml \
  --bin phase_diagram
```

Its default ignored output is
`research/chaosnli/artifacts/phase_diagram_100reps_rust.json`. Use `--help` after the
binary separator (`-- --help`) for phase-specific options, including reduced smoke-test
settings and an alternate comma-separated vote-depth grid.
