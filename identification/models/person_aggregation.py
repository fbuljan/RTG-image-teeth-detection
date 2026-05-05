"""
Person-level embedding aggregation.

Given multiple tooth embeddings from one person, aggregate them into a single
person-level embedding. Used for both gallery construction (offline) and
multi-tooth queries.

Aggregation methods:
- mean: simple mean pooling, then L2 normalize
- max: element-wise max pooling, then L2 normalize
- weighted: weighted mean (placeholder for learned weights, currently uniform)

All aggregators take an array of shape (N, D) where N is number of teeth and D
is embedding dim, and return a (D,) unit vector.
"""

from typing import Optional

import numpy as np


def aggregate_mean(embeddings: np.ndarray) -> np.ndarray:
    """Mean of N embeddings, then L2 normalize."""
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot aggregate empty embeddings")
    pooled = embeddings.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm < 1e-12:
        return pooled
    return pooled / norm


def aggregate_max(embeddings: np.ndarray) -> np.ndarray:
    """Element-wise max of N embeddings, then L2 normalize."""
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot aggregate empty embeddings")
    pooled = embeddings.max(axis=0)
    norm = np.linalg.norm(pooled)
    if norm < 1e-12:
        return pooled
    return pooled / norm


def aggregate_weighted(embeddings: np.ndarray,
                       weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted mean of N embeddings, then L2 normalize.

    Args:
        embeddings: (N, D) array
        weights: (N,) array of non-negative weights. If None, uniform.
    """
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot aggregate empty embeddings")
    if weights is None:
        weights = np.ones(embeddings.shape[0])
    weights = weights / weights.sum()
    pooled = (embeddings * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(pooled)
    if norm < 1e-12:
        return pooled
    return pooled / norm


AGGREGATORS = {
    "mean": aggregate_mean,
    "max": aggregate_max,
    "weighted": aggregate_weighted,
}


def aggregate_by_person(embeddings: np.ndarray, labels: np.ndarray,
                         method: str = "mean") -> tuple:
    """
    Aggregate per-tooth embeddings into per-person embeddings.

    Args:
        embeddings: (N_teeth, D) tooth embeddings
        labels: (N_teeth,) integer person ID for each tooth
        method: aggregation method ("mean", "max", "weighted")

    Returns:
        person_embeddings: (N_persons, D) aggregated embeddings
        person_labels: (N_persons,) integer labels matching the embeddings
    """
    if method not in AGGREGATORS:
        raise ValueError(f"Unknown method: {method}. Choose from {list(AGGREGATORS)}")
    aggregator = AGGREGATORS[method]

    unique_persons = np.unique(labels)
    person_embeddings = np.zeros((len(unique_persons), embeddings.shape[1]),
                                  dtype=embeddings.dtype)
    for i, person in enumerate(unique_persons):
        mask = labels == person
        person_embeddings[i] = aggregator(embeddings[mask])

    return person_embeddings, unique_persons


def sample_query_teeth(embeddings: np.ndarray, labels: np.ndarray,
                        n_query: int, rng: np.random.RandomState):
    """
    For each person, randomly sample n_query teeth as the query, return the rest as gallery.

    Used for the "given N teeth, identify the person" experiment.

    Returns:
        query_embeddings: (n_persons, n_query, D)
        gallery_embeddings_list: list of (n_remaining, D) per person
        person_labels: (n_persons,)
    """
    unique_persons = np.unique(labels)
    query_per_person = []
    gallery_per_person = []
    valid_persons = []

    for person in unique_persons:
        idx = np.where(labels == person)[0]
        if len(idx) <= n_query:
            # Skip persons without enough teeth for both query and gallery
            continue
        rng.shuffle(idx)
        q_idx = idx[:n_query]
        g_idx = idx[n_query:]
        query_per_person.append(embeddings[q_idx])
        gallery_per_person.append(embeddings[g_idx])
        valid_persons.append(person)

    return query_per_person, gallery_per_person, np.array(valid_persons)
