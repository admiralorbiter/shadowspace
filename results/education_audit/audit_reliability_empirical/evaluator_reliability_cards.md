# ER-2R2: Evaluator Reliability Cards (Fast Cluster Inference, N_clusters Sentence Units)

## Benchmark Corpus: `natural_substitutions`

| Evaluator Instrument | Independent? | MASD (95% Cluster CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail | TOST Mean Equivalence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Lexicon Keyword Density** | `Yes` | `0.0000 [0.0000, 0.0000]` | `0.00%` | `+0.0000` | `0.0000` | `Mean Signed Drift Equivalence Passed` |
| **Sparse N-Gram Baseline Ensemble** | `Yes` | `0.0172 [0.0152, 0.0194]` | `2.02%` | `+0.0082` | `0.0691` | `Mean Signed Drift Equivalence Passed` |
| **LABE Fine-Tuned BERT Transformer Classifier** | `Yes` | `0.0044 [0.0036, 0.0054]` | `0.40%` | `+0.0024` | `0.0278` | `Mean Signed Drift Equivalence Passed` |

### Substantive 2-Evaluator Agreement & 3x3 Category Cross-Tabulation

- **Substantive Evaluators**: `sparse_ngram_ensemble, labe_bert_transformer`
- **Exact Category Agreement Rate**: `38.06%`
- **Conditional Non-Zero Agreement Rate**: `4.37%`
- **Opposite-Direction Disagreement Rate**: `4.45%`

#### 3x3 Cross-Tabulation Table (Sparse N-Gram rows vs LABE BERT columns):

| N-Gram \ BERT | Negative | Zero | Positive |
| :--- | :--- | :--- | :--- |
| **Negative** | `0` | `24` | `6` |
| **Zero** | `2` | `87` | `11` |
| **Positive** | `5` | `105` | `7` |

## Benchmark Corpus: `controlled_injection`

| Evaluator Instrument | Independent? | MASD (95% Cluster CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail | TOST Mean Equivalence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Lexicon Keyword Density** | `Yes` | `0.0000 [0.0000, 0.0000]` | `0.00%` | `+0.0000` | `0.0000` | `Mean Signed Drift Equivalence Passed` |
| **Sparse N-Gram Baseline Ensemble** | `Yes` | `0.0086 [0.0085, 0.0087]` | `2.31%` | `-0.0085` | `0.0176` | `Mean Signed Drift Equivalence Passed` |
| **LABE Fine-Tuned BERT Transformer Classifier** | `Yes` | `0.0119 [0.0118, 0.0121]` | `0.42%` | `+0.0118` | `0.0221` | `Mean Signed Drift Equivalence Passed` |

### Substantive 2-Evaluator Agreement & 3x3 Category Cross-Tabulation

- **Substantive Evaluators**: `sparse_ngram_ensemble, labe_bert_transformer`
- **Exact Category Agreement Rate**: `48.74%`
- **Conditional Non-Zero Agreement Rate**: `0.00%`
- **Opposite-Direction Disagreement Rate**: `24.79%`

#### 3x3 Cross-Tabulation Table (Sparse N-Gram rows vs LABE BERT columns):

| N-Gram \ BERT | Negative | Zero | Positive |
| :--- | :--- | :--- | :--- |
| **Negative** | `0` | `1` | `118` |
| **Zero** | `1` | `232` | `124` |
| **Positive** | `0` | `0` | `0` |
