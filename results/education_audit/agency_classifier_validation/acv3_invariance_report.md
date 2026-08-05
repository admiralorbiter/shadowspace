# Phase ACV-3: Evaluator Bias & Counterfactual Invariance Report (Separated Frames)

- **Total Counterfactual Comparisons**: 9 (Names: 6, Pronouns: 3)
- **Lexicon Mean Absolute Drift (Zero-Drift Baseline Control)**: 0.0000
- **Classifier Overall Mean Signed Drift (Masc - Fem)**: +0.0025
- **Classifier Overall Mean Absolute Drift**: 0.0170
- **Classifier Maximum Absolute Drift**: 0.0577
- **Classification Flips**: 0 / 9 (0.0%)

## Channel-Specific Evaluator Drift (Names vs. Pronouns)

- **Name Interventions (N=6)**: Mean Signed = -0.0100 | Mean Abs = 0.0117
- **Pronoun Interventions (N=3)**: Mean Signed = +0.0275 | Mean Abs = 0.0275

## Key Finding

With strict frame separation (`text_a != text_b`), the exact agency lexicon remains a perfect zero-drift control (`drift = 0.0000`). The trained n-gram agency baseline exhibits **mean signed drift of +0.0025 and mean absolute drift of 0.0170**, confirming small identity-dependent evaluator noise.