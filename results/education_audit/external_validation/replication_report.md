# Milestone EV-1: Exact-Lexicon External Benchmark on LABE LAC & Wan 2023 Analysis

- **Wan 2023 Source Hash**: `c3ccc244b85a2e9ef9e671970a4f5cc41fc698b51770daa00d2a16df969f58be`
- **LABE Commit SHA**: `e8cc42d86df007fd05e3ae0c27c127b7a0a6165c`

## LABE LAC Split Performance (Precision / Recall / F1)

- **Train Split (N=2979)**: Precision = 76.9%, Recall = 31.1%, F1 = 0.443
- **Val Split (N=372)**: Precision = 78.3%, Recall = 28.1%, F1 = 0.414
- **Test Split — PRIMARY (N=373)**: Precision = 75.3%, Recall = 30.7%, F1 = 0.436
- **All Splits — Exploratory (N=3724)**: Precision = 76.9%, Recall = 30.7%, F1 = 0.439

## Wan et al. 2023 ChatGPT Letter Agency Uncertainty Analysis

- **Pairs Evaluated**: 60
- **Mean Agency Delta (Joseph - Kelly)**: -0.104 per 100 words
- **Median Agency Delta**: +0.000 per 100 words
- **Standard Deviation**: 0.372 | **IQR**: 0.327
- **95% Bootstrap Confidence Interval**: [-0.199, -0.014]
- **Directional Counts**: Favors Kelly: 25 / 60 | Favors Joseph: 15 / 60 | Zero Diff: 20 / 60