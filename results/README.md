# ChaosNLI canonical results

`canonical_results.json` is the only committed, release-facing quantitative
artifact for the ChaosNLI research package. Documentation and visualizations
must read values from this file rather than from recomputation outputs.

The canonical contract distinguishes:

- direct human-pair simulation summaries;
- focal-item bootstrap summaries;
- posterior-predictive versus observed reference values;
- plug-in multinomial reference-surface values;
- paired model recovery versus fixed-observed recovery;
- canonical Study 1 results versus exploratory Study 2 results.

## Recomputing and promotion

Analysis programs write intermediate outputs beneath
`research/chaosnli/artifacts/`, which is intentionally ignored by Git. This
prevents a partial or exploratory run from overwriting the release artifact.

After the required analyses have been rerun, promote them explicitly with:

```powershell
python research/chaosnli/manifests/promote_canonical_release.py `
  --canonical-core research/chaosnli/artifacts/round8_canonical_results.yaml `
  --reference-surface research/chaosnli/artifacts/multi_seed_reference_surface.json `
  --geometry research/chaosnli/artifacts/geometry_and_hbar_audit.json `
  --phase research/chaosnli/artifacts/phase_diagram_100reps.json `
  --output results/canonical_results.json
```

Promotion validates the locked model set, headline estimands, surface shape,
geometry table, phase-diagram grid, and VariErr result, and records each input
hash. Review the resulting diff before committing it.

The phase-diagram cells were recomputed with the corrected boundary definition,
cross-checked against the independent Rust implementation, and promoted from a
fresh 100-repetition artifact.

After independently validating a fresh phase artifact, promote only that
component without changing the already-locked result sections:

```powershell
python research/chaosnli/manifests/promote_phase_component.py `
  --phase research/chaosnli/artifacts/phase_diagram_100reps.json `
  --cross-check research/chaosnli/artifacts/phase_diagram_100reps_rust.json

python research/chaosnli/manifests/promote_row_order_component.py `
  --audit research/chaosnli/artifacts/row_order_audit.json
```

## Data scope

Raw ChaosNLI data, supplied model predictions, processed tables, matrices, and
recomputation artifacts are not committed. Their paths and hashes are locked in
`research/chaosnli/configs/study.yaml` and
`research/chaosnli/configs/model_artifacts.json`.
