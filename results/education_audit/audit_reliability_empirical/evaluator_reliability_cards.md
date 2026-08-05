# ER-2: Evaluator Reliability Cards (Full LABE Test Counterfactual Benchmark, N=1,492 Pairs)

## Evaluator Benchmark Summary Table

| Evaluator Instrument | Independent? | MASD (95% BCa CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail Risk | TOST Equivalence? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Lexicon Keyword Density** | `Yes` | `0.0000 [0.0000, 0.0000]` | `0.00%` | `+0.0000` | `0.0000` | `PASSED` |
| **Sparse N-Gram Baseline Ensemble** | `Yes` | `0.0070 [0.0064, 0.0077]` | `0.74%` | `+0.0041` | `0.0403` | `PASSED` |
| **LABE Independent BERT Transformer Classifier** | `Yes` | `0.0037 [0.0036, 0.0039]` | `0.00%` | `+0.0017` | `0.0122` | `PASSED` |

## Independent Evaluator Consensus & Agreement

- **Primary Epsilon ($\epsilon$)**: `0.01`
- **Mean Consensus Stability**: `0.9214`
- **All-Evaluator Agreement Rate**: `76.61%`
- **Majority Agreement Rate**: `99.80%`
- **Opposite-Direction Disagreement Rate**: `0.20%`