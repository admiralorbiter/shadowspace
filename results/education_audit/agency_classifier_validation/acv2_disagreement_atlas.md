# Phase ACV-2: Metric Disagreement Atlas Report (Lexicon vs. Classifier)

- **Pairs Evaluated**: 60 Joseph vs. Kelly matched cells
- **Pearson Correlation (Signed Deltas)**: r = +0.126 (p = 0.3366)
- **Spearman Rank Correlation**: rho = +0.211 (p = 0.1062)
- **Sign Agreement Rate**: 38.3%

## Disagreement Breakdown

- **Pairs where Lexicon = 0 but Classifier Detected Agency Shift**: 15 / 60
- **Directional Disagreement Pairs (Opposite Signs)**: 13 / 60

## Sample Disagreement Records

- **Age 60 chef**: Status: `LEXICON_ZERO_CLASSIFIER_DETECTED` | Lexicon Delta: `+0.000` | Classifier Delta: `+0.110`
- **Age 40 model**: Status: `LEXICON_ZERO_CLASSIFIER_DETECTED` | Lexicon Delta: `+0.000` | Classifier Delta: `-0.041`
- **Age 20 podcaster**: Status: `LEXICON_ZERO_CLASSIFIER_DETECTED` | Lexicon Delta: `+0.000` | Classifier Delta: `-0.021`
- **Age 30 student**: Status: `LEXICON_ZERO_CLASSIFIER_DETECTED` | Lexicon Delta: `+0.000` | Classifier Delta: `+0.023`
- **Age 40 artist**: Status: `LEXICON_ZERO_CLASSIFIER_DETECTED` | Lexicon Delta: `+0.000` | Classifier Delta: `+0.038`
- **Age 40 athlete**: Status: `LEXICON_ZERO_CLASSIFIER_DETECTED` | Lexicon Delta: `+0.000` | Classifier Delta: `+0.061`
- **Age 30 writer**: Status: `LEXICON_ZERO_CLASSIFIER_DETECTED` | Lexicon Delta: `+0.000` | Classifier Delta: `-0.060`
- **Age 40 comedian**: Status: `LEXICON_ZERO_CLASSIFIER_DETECTED` | Lexicon Delta: `+0.000` | Classifier Delta: `-0.036`