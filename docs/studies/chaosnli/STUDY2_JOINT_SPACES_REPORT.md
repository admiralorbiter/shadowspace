# Study 2 Multi-View Joint Spaces & Empirical Report

**Dataset:** 3,113 Three-Class ChaosNLI Examples (1,514 SNLI + 1,599 MNLI)  
**Text Embeddings:** 384-Dimensional `all-MiniLM-L6-v2` Dense Embeddings  
**Date:** 2026-08-01 (Revised post Round-3 review)  
**Scope:** Multi-View Opinion-Text Blending, Technical Tie Resolution, Intra-Profile Semantic Heterogeneity, and Reframed Hypothesis 7

---

## 1. Summary of Study 2 Quantitative Findings

| Estimand / Property | Value | Description |
|---|---|---|
| **Canonical Dataset Size** | **3,113 items** | 100 human judgments per item ($N=3,113$) |
| **Text Embedding Dimension** | **384 dims** | Dense `all-MiniLM-L6-v2` sentence embeddings |
| **Pure Opinion Zero-Distance Ties ($\lambda=0.00$)** | **3,381 zero ties** | Pairwise ties with exact $d_H = 0.0$ in 3-class simplex |
| **Joint Space Zero-Distance Ties ($\lambda=0.05$)** | **0 zero ties** | **100% zero-distance profile tie resolution** |
| **Joint Space Soft $Q_{NX}^{\text{soft}}(10)$ ($\lambda=0.05$)** | **0.2039 (20.39%)** | **Opinion neighborhood retention at $\lambda=0.05$** |
| **Mean Intra-Profile Text Distance** | **0.9562 Cosine** | Cosine distance among items sharing exact vote vectors |
| **Mean Overall Dataset Text Distance** | **0.9720 Cosine** | Cosine distance across all pairs in the dataset |
| **Hypothesis 7 Result** | **Proof of Concept** | Technical tie resolution demonstrated; linguistic validity pending external evaluation |

---

## 2. Deep Dive: Multi-View Blending ($\lambda \in [0.0, 1.0]$)

### Mathematical Formulation
The alpha-blended joint distance matrix $D_{\text{joint}}(\lambda)$ combines normalized Hellinger opinion distance $D_{\text{opinion}}$ and text Cosine distance $D_{\text{text}}$:
$$d_{\text{joint}}(u, v; \lambda) = \sqrt{(1 - \lambda) d_{\text{opinion}}^2(u, v) + \lambda d_{\text{text}}^2(u, v)}, \qquad \lambda \in [0.0, 1.0].$$

*Note on Metric Properties & Distance Scaling:* Standard Cosine distance $1 - \cos(u,v)$ does not strictly satisfy the triangle inequality. For formal metric space applications, normalized angular distance $d_{\text{angular}}(u,v) = \frac{1}{\pi}\arccos(\cos(u,v))$ should be used. Furthermore, because Hellinger distance lies in $[0, 1]$ and standard Cosine distance lies in $[0, 2]$, component distances must be normalized to matching scale ranges prior to blending for $\lambda$ to be strictly interpretable.

### Empirical Blend Curve

| $\lambda$ (Text Weight) | Zero-Distance Pairwise Ties Remaining | Soft $Q_{NX}$ Opinion Recovery | Research Finding |
|---|---|---|---|
| **0.00 (Pure Opinion)** | **3,381 zero ties** | **1.0000** | High profile density ties ($d_H = 0.0$) |
| **0.05 (Joint Space)** | **0 zero ties** | **0.2039** | **Technical tie resolution (20.39% opinion neighborhood retention)** |
| **0.10** | **0 zero ties** | **0.1169** | Complete tie resolution (11.69% opinion retention) |
| **0.20** | **0 zero ties** | **0.0627** | Complete tie resolution (6.27% opinion retention) |
| **0.50** | **0 zero ties** | **0.0237** | Balanced multi-view space |
| **1.00 (Pure Text)** | **0 zero ties** | **0.0041** | Pure text semantic similarity space |

---

## 3. Key Research Insights

1. **Technical Tie Resolution at $\lambda = 0.05$**:
   Blending a small amount ($\lambda = 0.05$) of text semantic distance into the discrete Hellinger probability space eliminates all 3,381 zero-distance profile ties. At $\lambda = 0.05$, the joint space retains $Q_{NX}^{\text{soft}} = 0.2039$ overlap with the pure opinion graph (meaning ~79.6% of neighborhood mass shifts due to text reordering). Note that comparing this 0.2039 self-fidelity score directly to the human 50/50 split-half reliability ($0.0426$) is methodologically invalid because the joint space directly incorporates the reference opinion matrix.

2. **Intra-Profile Semantic Heterogeneity**:
   Items sharing identical 100-vote human distributions have a mean intra-profile text Cosine distance of **0.9562**, compared to an overall dataset text distance of **0.9720** (a modest difference of 0.0158). This demonstrates that items with identical human vote vectors are not identical in text representation space, though they are modestly more similar than arbitrary dataset pairs.

3. **Hypothesis 7 Reframing & Scientific Status**:
   While text blending successfully resolves grid density ties numerically, establishing Hypothesis 7 as a substantive scientific claim requires demonstrating that text-based tie breaking better predicts independent disagreement taxonomies (e.g., Jiang & de Marneffe categories or VariErr annotations) than random or ID-based tie-breaking. Until external linguistic validation is completed, Study 2 serves as a technical proof of concept for multi-view tie resolution.

---

## 4. Reproducibility & Embedding Metadata

- **Text Encoder Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Input Serialization Format**: `Premise: {premise} Hypothesis: {hypothesis}`
- **Pooling & Normalization**: Mean pooling, L2 normalized output vectors
- **Distance Matrix File**: `data/chaosnli/processed/distance_matrix_text_cosine.npy`
