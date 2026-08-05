# ER-2: Evaluator Reliability Cards (Full LABE Test Counterfactual Benchmark, N=1,492 Pairs)

## Evaluator Benchmark Summary Table

| Evaluator Instrument | Independent? | MASD (95% BCa CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail Risk | TOST Equivalence? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Lexicon Keyword Density** | `Yes` | `0.0000 [0.0000, 0.0000]` | `0.00%` | `+0.0000` | `0.0000` | `PASSED` |
| **Sparse N-Gram Baseline Ensemble** | `Yes` | `0.0074 [0.0067, 0.0080]` | `0.80%` | `+0.0041` | `0.0442` | `PASSED` |
| **LABE Independent BERT Transformer Classifier** | `Yes` | `0.0050 [0.0048, 0.0052]` | `0.00%` | `-0.0037` | `0.0186` | `PASSED` |

## Independent Evaluator Consensus & Agreement

- **Primary Epsilon (\\epsilon)**: `0.01`
- **Mean Consensus Stability**: `0.9004`
- **All-Evaluator Agreement Rate**: `72.52%`
- **Majority Agreement Rate**: `97.59%`
- **Opposite-Direction Disagreement Rate**: `2.41%`