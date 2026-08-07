"""Lightweight end-to-end baseline validation against the optional real dataset."""

import math
from pathlib import Path

import pytest

from evaluate_baseline import evaluate_validation_baseline
from src.dataset_discovery import discover_layout, discover_pairs
from src.splits import split_pairs


pytestmark = pytest.mark.integration


def test_real_bicubic_baseline_pipeline(real_dataset_dir: Path) -> None:
    pairs = discover_pairs(discover_layout(real_dataset_dir)).pairs
    train, validation = split_pairs(pairs, val_fraction=0.2, seed=42)
    assert len(pairs) == 3200
    assert len(train) == 2560
    assert len(validation) == 640

    result = evaluate_validation_baseline(
        real_dataset_dir,
        val_fraction=0.2,
        seed=42,
        max_samples=3,
    )
    assert result["evaluated_samples"] == 3
    assert math.isfinite(result["psnr_db"])
    assert math.isfinite(result["ssim"])
    assert 0.0 <= result["ssim"] <= 1.0
