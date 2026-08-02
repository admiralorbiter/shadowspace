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

The default outputs are:

- `results/paired_estimand_results.yaml`
- `results/multi_seed_reference_surface.json`

Every path can be overridden, which also permits running the binary from another directory:

```bash
cargo run --release --locked \
  --manifest-path research/chaosnli/rust_manifest/Cargo.toml \
  -- \
  --items /path/to/canonical_items_posterior.json \
  --models /path/to/model_probs.json \
  --paired-output /path/to/paired_estimand_results.yaml \
  --surface-output /path/to/multi_seed_reference_surface.json
```

Use `--help` to list the available path options without running an analysis.
