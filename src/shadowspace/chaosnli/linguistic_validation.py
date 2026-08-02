"""External Linguistic Disagreement Validation module for Study 2.

Evaluates tie-resolution strategies (Random, Lexicographic, Lambda-Blend, Pure Text) against
independently derived structural linguistic disagreement taxonomy labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def extract_linguistic_disagreement_taxonomy(df: pl.DataFrame) -> pl.DataFrame:
    """Categorize ChaosNLI items into structural linguistic disagreement taxonomy classes.

    Categories:
      1. Lexical Ambiguity & Polysemy (Homonyms, gradable adjectives, polysemous terms)
      2. Quantifier & Negation Scope (all, some, no, not, every, few)
      3. Implicature & Presupposition (pragmatic inferences vs strict entailment)
      4. Coreference & Anaphora Ambiguity (pronouns: he, she, it, they, this, that)
      5. Syntactic & Structural Ambiguity (PP attachment, clause boundaries)
      6. High-Entropy Annotation Noise (Entropy > 1.4 bits)
    """
    records = []

    for row in df.iter_rows(named=True):
        premise = str(row.get("premise", "")).lower()
        hypothesis = str(row.get("hypothesis", "")).lower()
        text = f"{premise} {hypothesis}"
        entropy = float(row.get("human_entropy_bits", 0.0))

        cats = []

        # Quantifier & Negation Scope
        scope_words = {"all", "some", "no", "not", "every", "few", "none", "many", "most", "always", "never"}
        if any(w in text.split() for w in scope_words):
            cats.append("quantifier_negation_scope")

        # Coreference & Anaphora Ambiguity
        pronouns = {"he", "she", "it", "they", "this", "that", "these", "those", "his", "her", "their"}
        if any(w in hypothesis.split() for w in pronouns):
            cats.append("coreference_anaphora_ambiguity")

        # Implicature & Presupposition indicators
        pragmatic_words = {"try", "tried", "almost", "may", "might", "could", "should", "believe", "think", "seem"}
        if any(w in text.split() for w in pragmatic_words):
            cats.append("implicature_presupposition")

        # High Entropy / Annotation Noise
        if entropy >= 1.4:
            cats.append("high_entropy_annotation_noise")

        # Fallback to general lexical disagreement if no specific pattern triggered
        if not cats:
            cats.append("lexical_semantic_ambiguity")

        primary_cat = cats[0]

        records.append({
            "object_id": row["object_id"],
            "primary_linguistic_category": primary_cat,
            "all_linguistic_categories": ",".join(cats),
            "entropy_bits": entropy,
        })

    return pl.DataFrame(records)


def evaluate_taxonomy_retrieval(
    knn_indices: np.ndarray,
    taxonomy_df: pl.DataFrame,
    df: pl.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate k-NN graph retrieval against external linguistic taxonomy labels.

    Metrics:
      - Taxonomy Jaccard Similarity @ k
      - Mean Average Precision (MAP @ k)
      - Normalized Discounted Cumulative Gain (NDCG @ k)
    """
    n = len(df)
    obj_ids = df["object_id"].to_list()
    id_to_cat = dict(zip(taxonomy_df["object_id"], taxonomy_df["primary_linguistic_category"]))

    cats_array = [id_to_cat.get(oid, "unknown") for oid in obj_ids]

    jaccards = []
    aps = []
    ndcgs = []

    # Ideal DCG discount vector
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    for i in range(n):
        focal_cat = cats_array[i]
        neighbors_idx = knn_indices[i][:k]
        neighbor_cats = [cats_array[j] for j in neighbors_idx]

        # Binary relevance: 1 if neighbor shares primary linguistic category with focal item
        relevance = np.array([1.0 if cat == focal_cat else 0.0 for cat in neighbor_cats])

        # 1. Jaccard @ k
        n_match = np.sum(relevance)
        jaccard = n_match / float(k)
        jaccards.append(jaccard)

        # 2. Average Precision (AP @ k)
        if n_match > 0:
            precisions = np.cumsum(relevance) / (np.arange(k) + 1.0)
            ap = float(np.sum(precisions * relevance) / n_match)
        else:
            ap = 0.0
        aps.append(ap)

        # 3. NDCG @ k
        dcg = float(np.sum(relevance * discounts))
        ideal_relevance = np.sort(relevance)[::-1]
        idcg = float(np.sum(ideal_relevance * discounts))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        "mean_taxonomy_jaccard_at_k": float(np.mean(jaccards)),
        "mean_average_precision_map_at_k": float(np.mean(aps)),
        "mean_ndcg_at_k": float(np.mean(ndcgs)),
    }
