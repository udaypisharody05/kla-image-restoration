"""Shared synthetic and optional real-dataset pytest fixtures."""

import os
from pathlib import Path

import numpy as np
import pytest

from src.dataset_discovery import discover_layout, discover_pairs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal valid paired dataset without external files."""
    root = tmp_path / "data" / "Data-public"
    train_inputs = root / "train" / "train" / "NoisyLR"
    targets = root / "train" / "train" / "GT"
    test_inputs = root / "Test_NoisyLR" / "NoisyLR"
    for directory in (train_inputs, targets, test_inputs):
        directory.mkdir(parents=True)

    for sample_id, offset in (("000001", 0.0), ("000000", 0.1)):
        degraded = np.array(
            [[-0.2 + offset, 0.25], [0.75, 1.3 + offset]], dtype=np.float32
        )
        target = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        np.save(train_inputs / f"{sample_id}.npy", degraded)
        np.save(targets / f"{sample_id}.npy", target)
    np.save(test_inputs / "000100.npy", np.zeros((2, 2), dtype=np.float32))
    return root.parent


@pytest.fixture
def real_dataset_dir() -> Path:
    """Resolve the optional full dataset or skip integration tests clearly."""
    configured = os.environ.get("SEMICON_DATA_DIR")
    candidate = Path(configured) if configured else REPOSITORY_ROOT / "data"
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    candidate = candidate.resolve()
    help_message = (
        "Full hackathon dataset unavailable. Place it under "
        f"{REPOSITORY_ROOT / 'data'} or set SEMICON_DATA_DIR to its directory."
    )
    if not candidate.is_dir():
        pytest.skip(help_message)
    try:
        layout = discover_layout(candidate)
        pairs = discover_pairs(layout)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        pytest.skip(f"{help_message} Discovery error: {exc}")
    if not pairs.pairs:
        pytest.skip(f"{help_message} No valid training pairs were found.")
    return candidate
