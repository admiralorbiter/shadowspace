# Phase ACV-3: Evaluator Bias & Counterfactual Invariance Report

- **Counterfactual Identity Comparisons**: 18
- **Lexicon Mean Drift (Zero-Drift Baseline Control)**: 0.0000
- **Classifier Mean Evaluator Drift**: 0.0177
- **Classifier Maximum Evaluator Drift**: 0.0577
- **Classification Flips**: 0 / 18 (0.0%)

## Evaluator Drift by Identity Swap Category

- **Name Substitutions (e.g. Michael vs. Sarah)**: Mean Drift = 0.0196
- **Pronoun Substitutions (e.g. He vs. She)**: Mean Drift = 0.0138

## Key Conclusion

The exact agency lexicon acts as a perfect zero-drift baseline control (`drift = 0.0000`). The contextual classifier exhibits **mean evaluator drift of 0.0177 and a classification flip rate of 0.0%**, proving that contextual model evaluators can introduce identity-dependent measurement noise when scoring identical achievement text.