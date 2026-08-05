# ER-2R & ER-2S: Evaluator Reliability Cards (Cluster-Aware Inference, N=373 Sentence Clusters)

## Benchmark Corpus: `natural_substitutions` (N=212 Pairs)

| Evaluator Instrument | Independent? | MASD (95% Cluster BCa CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail | TOST Mean Equivalence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Lexicon Keyword Density** | `Yes` | `0.0000 [0.0000, 0.0000]` | `0.00%` | `+0.0000` | `0.0000` | `Mean Signed Drift Equivalence Passed` |
| **Sparse N-Gram Baseline Ensemble** | `Yes` | `0.0068 [0.0059, 0.0076]` | `0.47%` | `+0.0035` | `0.0381` | `Mean Signed Drift Equivalence Passed` |
| **LABE Fine-Tuned BERT Transformer Classifier** | `Yes` | `0.0042 [0.0039, 0.0045]` | `0.00%` | `-0.0028` | `0.0152` | `Mean Signed Drift Equivalence Passed` |

### Substantive 2-Evaluator Agreement (Excludes Deterministic Negative Control)
- **Substantive Evaluators**: `sparse_ngram_ensemble, labe_bert_transformer`
- **Mean Substantive Consensus Stability**: `0.9245`
- **Exact Category Agreement Rate**: `84.91%`
- **Opposite-Direction Disagreement Rate**: `0.94%`

## Benchmark Corpus: `controlled_injection` (N=1,492 Pairs)

| Evaluator Instrument | Independent? | MASD (95% Cluster BCa CI) | CFR (Flip Rate) | Signed Drift | CVaR_.95 Tail | TOST Mean Equivalence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Lexicon Keyword Density** | `Yes` | `0.0000 [0.0000, 0.0000]` | `0.00%` | `+0.0000` | `0.0000` | `Mean Signed Drift Equivalence Passed` |
| **Sparse N-Gram Baseline Ensemble** | `Yes` | `0.0074 [0.0067, 0.0080]` | `0.80%` | `+0.0041` | `0.0442` | `Mean Signed Drift Equivalence Passed` |
| **LABE Fine-Tuned BERT Transformer Classifier** | `Yes` | `0.0050 [0.0048, 0.0052]` | `0.00%` | `-0.0037` | `0.0186` | `Mean Signed Drift Equivalence Passed` |

### Substantive 2-Evaluator Agreement (Excludes Deterministic Negative Control)
- **Substantive Evaluators**: `sparse_ngram_ensemble, labe_bert_transformer`
- **Mean Substantive Consensus Stability**: `0.9004`
- **Exact Category Agreement Rate**: `72.52%`
- **Opposite-Direction Disagreement Rate**: `2.41%`