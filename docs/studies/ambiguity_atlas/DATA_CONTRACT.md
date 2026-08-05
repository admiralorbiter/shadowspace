# Data Contract and Invariant Specifications

## 1. Input Datasets

### 1.1 Canonical Items (`data/chaosnli/processed/canonical_items.parquet`)
- **Total rows**: 3,113
- **Primary key**: `object_id` (String)
- **Required columns**:
  - `object_id`: Unique item identifier
  - `source_dataset`: Dataset source (`snli` or `mnli`)
  - `premise`: Premise text
  - `hypothesis`: Hypothesis text
  - `human_count_entailment`: Integer vote count for Entailment
  - `human_count_neutral`: Integer vote count for Neutral
  - `human_count_contradiction`: Integer vote count for Contradiction
  - `human_p_entailment`: Float probability for Entailment
  - `human_p_neutral`: Float probability for Neutral
  - `human_p_contradiction`: Float probability for Contradiction
  - `human_entropy_bits`: Shannon entropy in bits
  - `human_majority_label`: Class label with maximum votes (`entailment`, `neutral`, or `contradiction`)

### 1.2 OOF Predictions (`results/exploratory/oof_predictions.parquet`)
- **Total rows**: 9,339 (3,113 items $\times$ 3 models)
- **Primary key**: (`object_id`, `model_name`)
- **Required columns**:
  - `object_id`: Matches canonical items
  - `model_name`: Model identifier (`deberta-v3-large`, `roberta-large`, `electra-large`)
  - `fold_id`: Cross-validation fold (0 to 4)
  - `q_raw_e`, `q_raw_n`, `q_raw_c`: Raw model output probabilities
  - `q_t1_e`, `q_t1_n`, `q_t1_c`: Calibration Tier 1 (Scalar temperature)
  - `q_t2_e`, `q_t2_n`, `q_t2_c`: Calibration Tier 2 (Diagonal ILR)
  - `q_t3_e`, `q_t3_n`, `q_t3_c`: Calibration Tier 3 (Affine ILR)
  - `q_t4_e`, `q_t4_n`, `q_t4_c`: Calibration Tier 4 (Nonlinear ILR)

---

## 2. Invariants & Validation Rules

1. **Probability Sum Invariant**: For every row, $\sum p_i = 1.0 \pm 10^{-5}$ and $\sum q_i = 1.0 \pm 10^{-5}$.
2. **Count Integrity**: All `human_count_*` values must be non-negative integers.
3. **Majority Consistency**: `human_majority_label` must correspond to the class index with maximum vote count.
4. **Entropy Integrity**: `human_entropy_bits` must match $-\sum p_i \log_2 p_i$ within tolerance $10^{-4}$.
5. **No Nulls / NaNs**: Key columns must contain zero null or infinite values.
