# Ambiguity Doppelgänger Atlas: What Confidence and Entropy Cannot Tell You

## Project Overview

The **Ambiguity Doppelgänger Atlas** explores a fundamental mathematical information loss mechanism inherent in common 1D probability distribution scalar summaries: **majority label, maximum probability (confidence), and Shannon entropy**.

In three-class classification tasks (such as Natural Language Inference: *Entailment*, *Neutral*, *Contradiction*), a summary such as:
- **Majority**: Entailment
- **Confidence**: 60%
- **Entropy**: 1.30 bits

is **two-to-one (degenerate)**. It cannot distinguish between:
1. $p^+ = (0.60, 0.30, 0.10)$ — 30% Neutral, 10% Contradiction
2. $p^- = (0.60, 0.10, 0.30)$ — 10% Neutral, 30% Contradiction

These two probability distributions occupy opposite minority-disagreement directions in the 2D probability simplex $\Delta^2$, yet appear completely identical on standard dashboards.

### Core Objectives
1. **Mathematical Proof**: Formalize the Minority-Swap Collision Theorem and derive closed-form expressions for Hellinger, Fisher–Rao, Jensen–Shannon, and Aitchison distances between doppelgänger pairs.
2. **Empirical Census**: Discover exact and approximate human ambiguity doppelgänger pairs in ChaosNLI.
3. **Posterior Stability Audit**: Evaluate whether identified doppelgänger pairs remain distinct under annotation noise using Dirichlet posterior sampling ($\theta \sim \text{Dirichlet}(c + 0.5)$).
4. **Frozen Model Retention Audit**: Measure whether held-out predictions from models (DeBERTa-v3, RoBERTa, Electra across calibration tiers 0–4) preserve, collapse, attenuate, or invert human minority disagreement directions.
5. **Interactive Doppelgänger Atlas**: Build an interactive HTML/SVG browser tool with Simplex visualizer, Reveal Mode, and Game Mode for intuitive exploration of lost ambiguity structure.

---

## Repository Structure

- `src/shadowspace/ambiguity_atlas/`: Reusable core mathematical and analytical modules.
  - `geometry.py`: Mirror distribution parameterization, binary/summary entropy, distance metrics.
  - `summaries.py`: Summary map operations and permutation distance utilities.
  - `pair_index.py`: Polars exact grouping and blockwise approximate pair discovery algorithms.
  - `posterior.py`: Dirichlet posterior sampling and stability classification logic.
  - `retention.py`: Model orientation evaluation, contrast retention ratios, and classification categories.
  - `schemas.py`: Validation schemas for data invariants and outputs.
- `research/ambiguity_atlas/`: Orchestration scripts and experiment entrypoints.
  - `configs/atlas_v1.yaml`: Frozen hyperparameters and policy definitions.
  - `preflight.py`: Data preflight verification script.
  - `run_theory.py`: Analytical surface validation script.
  - `run_strict_census.py`: Exact count-permutation doppelgänger census.
  - `run_approximate_census.py`: Approximate tolerance grid and Pareto frontier search.
  - `run_posterior_audit.py`: Dirichlet posterior uncertainty analysis.
  - `run_model_retention.py`: Model prediction orientation retention audit.
  - `build_atlas.py`: JSON payload generator for the interactive HTML atlas.
  - `freeze_manifest.py`: Cryptographic reproducibility manifest builder.
- `tests/ambiguity_atlas/`: Unit test suite.
- `docs/studies/ambiguity_atlas/`: Comprehensive documentation and mathematical notes.
- `docs/viz/ambiguity_atlas/`: Interactive visualizer HTML/CSS/JS.
- `results/ambiguity_atlas/`: Outputs, Parquet tables, and final payload artifacts.

---

## Execution Prerequisites

- **Data Inputs**:
  - `data/chaosnli/processed/canonical_items.parquet` (3,113 ChaosNLI item vote distributions)
  - `results/exploratory/oof_predictions.parquet` (Held-out probability vectors across models and tiers)
- **Environment**: Python 3.10+ with NumPy, Polars, SciPy, PyArrow, scikit-learn, and PyYAML.
- **Dependencies**: No generative models, transformers inference, Ollama, or GPU compute are required.
