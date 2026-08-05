# ER-2R2: Evaluator Reliability Cards (Fast Cluster Inference, N_clusters Sentence Units)

## Benchmark Corpus: `natural_substitutions`

| Evaluator Instrument | Independent? | MASD (95% Cluster CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail | TOST Mean Equivalence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Lexicon Keyword Density** | `Yes` | `0.0000 [0.0000, 0.0000]` | `0.00%` | `+0.0000` | `0.0000` | `Mean Signed Drift Equivalence Passed` |
| **Sparse N-Gram Baseline Ensemble** | `Yes` | `0.0172 [0.0152, 0.0194]` | `2.02%` | `+0.0082` | `0.0691` | `Mean Signed Drift Equivalence Passed` |
| **LABE Fine-Tuned BERT Transformer Classifier** | `Yes` | `0.0042 [0.0027, 0.0061]` | `0.81%` | `-0.0016` | `0.0479` | `Mean Signed Drift Equivalence Passed` |

### Substantive 2-Evaluator Agreement & 3x3 Category Cross-Tabulation

- **Substantive Evaluators**: `sparse_ngram_ensemble, labe_bert_transformer`
- **Exact Category Agreement Rate**: `40.08%`
- **Conditional Non-Zero Agreement Rate**: `4.52%`
- **Opposite-Direction Disagreement Rate**: `6.07%`

#### 3x3 Cross-Tabulation Table (Sparse N-Gram rows vs LABE BERT columns):

| N-Gram \ BERT | Negative | Zero | Positive |
| :--- | :--- | :--- | :--- |
| **Negative** | `2` | `24` | `4` |
| **Zero** | `7` | `92` | `1` |
| **Positive** | `11` | `101` | `5` |

## Benchmark Corpus: `controlled_injection`

| Evaluator Instrument | Independent? | MASD (95% Cluster CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail | TOST Mean Equivalence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Lexicon Keyword Density** | `Yes` | `0.0000 [0.0000, 0.0000]` | `0.00%` | `+0.0000` | `0.0000` | `Mean Signed Drift Equivalence Passed` |
| **Sparse N-Gram Baseline Ensemble** | `Yes` | `0.0086 [0.0085, 0.0087]` | `2.31%` | `-0.0085` | `0.0176` | `Mean Signed Drift Equivalence Passed` |
| **LABE Fine-Tuned BERT Transformer Classifier** | `Yes` | `0.0312 [0.0309, 0.0315]` | `11.13%` | `+0.0308` | `0.0688` | `Mean Signed Drift Equivalence Failed` |

### Substantive 2-Evaluator Agreement & 3x3 Category Cross-Tabulation

- **Substantive Evaluators**: `sparse_ngram_ensemble, labe_bert_transformer`
- **Exact Category Agreement Rate**: `48.53%`
- **Conditional Non-Zero Agreement Rate**: `0.00%`
- **Opposite-Direction Disagreement Rate**: `24.79%`

#### 3x3 Cross-Tabulation Table (Sparse N-Gram rows vs LABE BERT columns):

| N-Gram \ BERT | Negative | Zero | Positive |
| :--- | :--- | :--- | :--- |
| **Negative** | `0` | `1` | `118` |
| **Zero** | `4` | `231` | `122` |
| **Positive** | `0` | `0` | `0` |
