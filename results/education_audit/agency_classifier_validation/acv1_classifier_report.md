# Phase ACV-1: LABE-Trained Sparse N-Gram Agency Classifier Baseline Report

- **Model Baseline**: `Sparse N-Gram TF-IDF (1-3) + Ensemble (LR + GB)`
- **LABE Commit SHA**: `e8cc42d86df007fd05e3ae0c27c127b7a0a6165c`
- **Random Seed**: `101`
- **Training Split**: N=2979 (Positive Rate = 51.1%)
- **Validation Split**: N=372 (Positive Rate = 51.6%)
- **Tuned Decision Threshold**: `0.49` (Validation F1 = 0.924)

## Primary Locked-Test Performance Metrics (N=373)

- **Precision**: 84.0%
- **Recall**: 95.0%
- **F1 Score**: 0.892
- **AUROC**: 0.949
- **Log Loss**: 0.347
- **Brier Score**: 0.103

## Key Finding

The sparse n-gram agency classifier baseline achieves a **locked test F1 of 0.892 (AUROC = 0.949)**, substantially outperforming the exact agency lexicon (F1 = 0.436) by capturing broader lexical and phrasal agency patterns.