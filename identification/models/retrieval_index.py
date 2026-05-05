"""
FAISS-based retrieval index for tooth embeddings.

Wraps faiss.IndexFlatIP for cosine similarity search on L2-normalized embeddings.
Supports adding/removing persons and batched queries.

Usage:
    index = RetrievalIndex(dim=128)
    index.add(person_embeddings, person_ids)
    distances, neighbor_ids = index.search(query_embedding, k=10)
"""

from typing import List, Tuple, Union

import faiss
import numpy as np


class RetrievalIndex:
    """
    Wrapper around faiss.IndexFlatIP for cosine similarity search.

    Embeddings must be L2-normalized for cosine = dot product.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.person_ids: List[Union[str, int]] = []  # parallel array

    def add(self, embeddings: np.ndarray, person_ids: List[Union[str, int]]):
        """Add embeddings to the index. Embeddings must be (N, dim), L2-normalized."""
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {embeddings.shape[1]}")
        if len(person_ids) != embeddings.shape[0]:
            raise ValueError("person_ids length must match embeddings count")
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.person_ids.extend(person_ids)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[List[Union[str, int]]]]:
        """
        Search for k nearest neighbors.

        Args:
            query: (Q, dim) array of L2-normalized query embeddings, or (dim,) for single query
            k: number of neighbors

        Returns:
            similarities: (Q, k) cosine similarities (or (k,) for single query)
            neighbor_ids: list of person_ids per query
        """
        single = query.ndim == 1
        if single:
            query = query[None, :]
        query = np.ascontiguousarray(query, dtype=np.float32)

        sims, indices = self.index.search(query, k)
        neighbor_ids = [
            [self.person_ids[idx] if 0 <= idx < len(self.person_ids) else None
             for idx in row]
            for row in indices
        ]

        if single:
            return sims[0], neighbor_ids[0]
        return sims, neighbor_ids

    def __len__(self) -> int:
        return self.index.ntotal

    def reset(self):
        """Clear the index."""
        self.index.reset()
        self.person_ids = []

    def save(self, path: str):
        """Save index to disk (FAISS binary + person_ids JSON)."""
        import json
        from pathlib import Path
        path = Path(path)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".ids.json"), "w") as f:
            json.dump([str(p) for p in self.person_ids], f)

    @classmethod
    def load(cls, path: str, dim: int = 128) -> "RetrievalIndex":
        """Load index from disk."""
        import json
        from pathlib import Path
        path = Path(path)
        idx = cls(dim=dim)
        idx.index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".ids.json")) as f:
            idx.person_ids = json.load(f)
        return idx
