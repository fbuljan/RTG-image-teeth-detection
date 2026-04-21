"""
PK batch sampler for metric learning.

Yields batches of P persons × K samples per person, ensuring each batch
contains diverse persons for effective triplet/contrastive mining.
"""

import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    Batch sampler that yields P persons × K samples per person.

    Args:
        labels: Integer label for each dataset sample (person ID as int).
        p: Number of persons per batch.
        k: Number of samples per person per batch.
    """

    def __init__(self, labels: List[int], p: int, k: int):
        self.p = p
        self.k = k

        # Build label → indices mapping
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # Filter to labels with at least 1 sample (should be all)
        self.labels = list(self.label_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle person order each epoch
        label_order = self.labels.copy()
        random.shuffle(label_order)

        # Yield batches of P persons × K samples
        for start in range(0, len(label_order) - self.p + 1, self.p):
            batch_labels = label_order[start:start + self.p]
            batch_indices = []

            for label in batch_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    chosen = random.sample(indices, self.k)
                else:
                    # Sample with replacement for persons with < K teeth
                    chosen = random.choices(indices, k=self.k)
                batch_indices.extend(chosen)

            yield batch_indices

    def __len__(self) -> int:
        return len(self.labels) // self.p
