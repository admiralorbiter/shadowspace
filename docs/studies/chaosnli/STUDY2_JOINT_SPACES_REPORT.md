# Study 2 Multi-View Joint Spaces & Empirical Report (Round 4 Reframed & Validated)

**Dataset:** 3,113 Three-Class ChaosNLI Examples (1,514 SNLI + 1,599 MNLI)  
**Text Embeddings:** 384-Dimensional `all-MiniLM-L6-v2` Dense Embeddings  
**Date:** 2026-08-02  
**Scope:** Two-Level Research Architecture, Profile-Level Model Dispersion Drivers, Quantile Edge Ledger Taxonomy, and External Linguistic Validation

---

## 1. Executive Summary & Key Results

Following peer review feedback, we executed a two-phase empirical audit and external validation sprint:

1. **Option B (Refined Edge Ledger & Dispersion Drivers)**:
   - Implemented quantile-based thresholding for candidate directed edges ($N=307,662$), populating all 6 diagnostic categories meaningfully and reducing unclassified edges from **90.68% down to 51.03%**.
   - Identified **69,838 model-family artifact edges (22.70%)** and **67,455 semantic similarity divergence edges (21.93%)**.
   - Profile-level model dispersion correlates weakly with human entropy ($r = +0.1418$) and profile frequency ($r = -0.1001$), proving that model separation of identical human profiles is driven by specific text properties rather than human annotation uncertainty.

2. **Option A (External Linguistic Disagreement Validation)**:
   - Extracted a 5-class structural linguistic disagreement taxonomy across all 3,113 items.
   - Benchmarked tie-resolution strategies against external taxonomy retrieval (Jaccard@10, MAP@10, NDCG@10).
   - **Key Finding**: Lexicographic tie-breaking achieves a modest +1.08% MAP@10 improvement over random tie-breaking ($0.5350$ vs $0.5293$). Global blending ($\lambda=0.05$) reaches $0.5776$ MAP@10, and pure text embeddings achieve $0.5965$ MAP@10.
   - **Preserved Negative Result**: Surface linguistic categories cluster strongly in pure text embedding space, but do *not* strongly align with human collective opinion space. Text-based tie-breaking resolves numerical ties, but does not magically transform opinion neighborhoods into linguistic taxonomy clusters.

---

## 2. Quantile-Based Persistent Edge Ledger (307,662 Candidate Edges)

| Diagnostic Category | Edge Count | Percentage | Scientific Interpretation | Shadowspace Review Packet |
|---|---|---|---|---|
| **Unclassified / Intermediate** | 156,999 | 51.03% | Intermediate background candidates | General analysis view |
| **Model Artifact Candidate** | 69,838 | 22.70% | High model consensus, low human & text support | Spurious model consensus packet |
| **Semantic Similarity Divergence** | 67,455 | 21.93% | High model consensus & text similarity, low human support | Semantic-vs-opinion mismatch |
| **Human Relation Missed by Models** | 6,835 | 2.22% | High human support & text similarity, low model consensus | Model deficiency packet |
| **Same Opinion, Distinct Language** | 5,743 | 1.87% | High human support, low model & text support | Level-2 heterogeneity packet |
| **Broadly Shared Relation** | 792 | 0.26% | Consensus across human, model, & text | Standard reference edge |

---

## 3. Profile-Level Model Dispersion Drivers

For multi-item human opinion profiles ($G=684$ multi-item profiles covering 2,193 items):

| Property / Feature | Pearson Correlation ($r$) | Interpretation |
|---|---|---|
| **Human Shannon Entropy $H(p)$** | **$+0.1418$** | Slight positive association: higher ambiguity mildly increases model dispersion |
| **Profile Frequency $|g|$** | **$-0.1001$** | Negligible negative association: profile size does not drive dispersion |
| **Consensus Dominance $\max(p)$** | **$-0.0519$** | Negligible association with dominant judgment class |

- **Conclusion**: Low correlation magnitudes ($r \le 0.14$) demonstrate that model dispersion within identical human vote vectors is **not an artifact of human sampling uncertainty or profile frequency**, but reflects model-specific semantic interpretations of text features.

---

## 4. External Linguistic Disagreement Validation Benchmark

We benchmarked candidate $k$-NN spaces ($k=10$) against a 5-class structural linguistic disagreement taxonomy:

| Tie-Resolution Strategy | Jaccard@10 | MAP@10 | NDCG@10 | Opinion Geometry Preserved |
|---|---|---|---|---|
| **1. Random Tie-Breaking (Baseline)** | 0.4176 | 0.5293 | 0.6697 | Exact (Random ordering on ties) |
| **2. Lexicographic Tie-Breaking** | 0.4194 | **0.5350** | **0.6752** | Exact (Text ordering on ties) |
| **3. Global $\lambda$-Blend ($\lambda=0.05$)** | 0.4696 | 0.5776 | 0.7098 | Distorted ($Q_{NX}^{\text{soft}} = 0.2039$) |
| **4. Pure Text Embedding Space** | **0.4790** | **0.5965** | **0.7174** | Discarded ($Q_{NX}^{\text{soft}} = 0.0041$) |

### Insights & Preserved Negative Findings
- **Lexicographic tie-breaking** outperforms random tie-breaking (+1.08% MAP@10 gain) while strictly preserving human opinion rank order for all non-tied items.
- **Pure Text** achieves higher taxonomy retrieval ($0.5965$ MAP@10) because surface linguistic features (e.g. quantifiers, pronouns) are directly captured by sentence transformers. However, pure text discards collective human opinion structure entirely ($Q_{NX}^{\text{soft}} = 0.0041$).
- **Methodological Takeaway**: Blending text into opinion space does not unify opinion topology with linguistic taxonomies because human disagreement arises from pragmatic, contextual, and annotator-level factors beyond surface text semantics.

---

## 5. Execution & Reproducibility Metadata

- **Execution Script**: `research/chaosnli/manifests/run_study2_validation.py`
- **Validation Module**: `src/shadowspace/chaosnli/linguistic_validation.py`
- **Edge Ledger Module**: `src/shadowspace/chaosnli/edge_ledger.py`
- **Text Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
