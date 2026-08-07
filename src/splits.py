"""Reproducible train/validation splitting for discovered image pairs."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np


T = TypeVar("T")


def split_pairs(
    pairs: Sequence[T], val_fraction: float = 0.2, seed: int = 42
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Split pairs deterministically while preserving their original order.

    Validation membership is selected using NumPy's seeded generator. The rounded
    validation count is constrained so both subsets are non-empty.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if len(pairs) < 2:
        raise ValueError("at least two pairs are required for a train/validation split")

    validation_count = min(
        len(pairs) - 1, max(1, int(round(len(pairs) * val_fraction)))
    )
    generator = np.random.default_rng(seed)
    validation_indices = set(
        int(index) for index in generator.permutation(len(pairs))[:validation_count]
    )
    train = tuple(pair for index, pair in enumerate(pairs) if index not in validation_indices)
    validation = tuple(
        pair for index, pair in enumerate(pairs) if index in validation_indices
    )
    return train, validation
