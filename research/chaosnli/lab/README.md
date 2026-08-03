# ChaosNLI Laboratory

This directory contains the experimental laboratory for the ChaosNLI research program.

See [`plans/ROADMAP.md`](plans/ROADMAP.md) for the full research plan, current experiment
status, and forward agenda.

## Quick Reference

| Path | Purpose |
|---|---|
| `registry/` | Experiment specs (TOML) — source of truth for design decisions |
| `summaries/` | Run outputs and human-readable summaries |
| `plans/` | Research design docs and roadmap |
| `artifacts/` | Persisted support matrices (f32) + run manifests (created at runtime) |

## Active Experiments

| Experiment | Status | Spec |
|---|---|---|
| E001: Posterior Edge-Support Graph | `pilot_partial` — re-run required | [E001.toml](registry/E001.toml) |
| E002: Temperature Calibration | `pilot` — redesigned, re-run required | [E002.toml](registry/E002.toml) |

## Run Order

Always run E001 before E002. E002 loads E001's persisted f32 support matrices.

```bash
cd ../rust_manifest
cargo run --release --bin e001_edge_support
# then:
cargo run --release --bin e002_temperature_scaling
```
