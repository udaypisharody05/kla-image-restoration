"""Near-duplicate scene grouping and a leakage-aware secondary split.

Experiment 22 found that 119 groups of training pairs share a near-identical GT
scene under independent noise realizations, and that 36 of those groups straddle
the canonical train/validation split -- so 38 of 640 canonical validation images
have a near-identical twin in train, making absolute validation numbers slightly
optimistic.

This module builds a **secondary, diagnostic-only** split that keeps each scene
group wholly on one side. It deliberately does not touch the canonical split:
every experiment from 1 onward was measured against that split, and changing it
would make new numbers incomparable with the entire experiment log.
"""

from collections.abc import Sequence

import numpy as np

from .dataset_discovery import ImagePair
from .degradation import connected_components, perceptual_signature
from .io_utils import load_image_array


DEFAULT_GT_MSE_THRESHOLD = 1e-3


def find_scene_groups(
    pairs: Sequence[ImagePair], threshold: float = DEFAULT_GT_MSE_THRESHOLD
) -> list[list[str]]:
    """Group pair ids whose GT images are near-identical.

    An 8x8 average-hash nominates candidates cheaply, then every candidate pair
    is **confirmed** by real GT mean-squared error -- the hash alone collides for
    unrelated images, so skipping the confirmation would badly over-group.
    Returns only groups with more than one member, each sorted, ordered
    deterministically by smallest member.
    """
    signatures: dict[str, list[str]] = {}
    arrays: dict[str, np.ndarray] = {}
    for pair in pairs:
        gt = np.asarray(load_image_array(pair.target_path), dtype=np.float32)
        arrays[pair.pair_id] = gt
        signatures.setdefault(perceptual_signature(gt), []).append(pair.pair_id)

    edges: list[tuple[str, str]] = []
    for candidates in signatures.values():
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                left, right = candidates[i], candidates[j]
                if float(np.mean((arrays[left] - arrays[right]) ** 2)) < threshold:
                    edges.append((left, right))

    components = connected_components([pair.pair_id for pair in pairs], edges)
    return [group for group in components if len(group) > 1]


def group_aware_split(
    pairs: Sequence[ImagePair],
    groups: Sequence[Sequence[str]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[tuple[ImagePair, ...], tuple[ImagePair, ...]]:
    """Split *pairs* so no scene group straddles the boundary.

    Grouped pairs move as a unit; ungrouped pairs are individually assignable.
    Shuffling uses ``numpy.random.default_rng(seed)``, matching
    ``src/splits.py``'s convention, so the result is deterministic. The realized
    validation fraction can differ slightly from *val_fraction* because whole
    groups are indivisible.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    by_id = {pair.pair_id: pair for pair in pairs}
    membership = {member: index for index, group in enumerate(groups) for member in group}

    units: list[list[str]] = [list(group) for group in groups]
    units.extend([pair.pair_id] for pair in pairs if pair.pair_id not in membership)
    units.sort(key=lambda unit: min(unit))  # deterministic pre-shuffle ordering

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(units))
    target = int(round(len(pairs) * val_fraction))

    validation_ids: list[str] = []
    train_ids: list[str] = []
    for position in order:
        unit = units[position]
        if len(validation_ids) + len(unit) <= target:
            validation_ids.extend(unit)
        else:
            train_ids.extend(unit)

    to_pairs = lambda ids: tuple(by_id[i] for i in sorted(ids) if i in by_id)
    return to_pairs(train_ids), to_pairs(validation_ids)


def count_cross_split_groups(
    groups: Sequence[Sequence[str]],
    train_pairs: Sequence[ImagePair],
    validation_pairs: Sequence[ImagePair],
) -> dict[str, int]:
    """How badly a split leaks: groups spanning both sides, and images affected."""
    train_ids = {pair.pair_id for pair in train_pairs}
    validation_ids = {pair.pair_id for pair in validation_pairs}
    spanning = [
        group
        for group in groups
        if any(member in train_ids for member in group)
        and any(member in validation_ids for member in group)
    ]
    return {
        "groups_spanning_split": len(spanning),
        "validation_images_with_train_twin": sum(
            1 for group in spanning for member in group if member in validation_ids
        ),
    }
