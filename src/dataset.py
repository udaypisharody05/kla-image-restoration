"""Lazy PyTorch datasets and DataLoader construction for restoration arrays."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset_discovery import ImagePair
from .io_utils import load_image_array


Sample = dict[str, torch.Tensor | str]


def _grayscale_tensor(array: np.ndarray, path: Path, role: str) -> torch.Tensor:
    """Validate a raw grayscale array and convert it to float32 CHW format."""
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(
            f"{role} must be a 2D grayscale array, got shape {values.shape}: {path}"
        )
    if min(values.shape) < 1:
        raise ValueError(f"{role} has an empty spatial dimension: {path}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{role} must contain numeric values, got {values.dtype}: {path}")
    if not np.isfinite(values).all():
        nan_count = int(np.isnan(values).sum())
        inf_count = int(np.isinf(values).sum())
        raise ValueError(
            f"{role} contains non-finite values (NaN={nan_count}, Inf={inf_count}): "
            f"{path}"
        )
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return torch.from_numpy(contiguous).unsqueeze(0)


class PairedRestorationDataset(Dataset[Sample]):
    """Lazily load existing discovered input/target pairs as float32 tensors."""

    def __init__(
        self,
        pairs: Sequence[ImagePair],
        scale: int = 2,
        transform: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
    ) -> None:
        if scale < 1:
            raise ValueError("scale must be a positive integer")
        self.pairs = tuple(pairs)
        self.scale = scale
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Sample:
        pair = self.pairs[index]
        input_tensor = _grayscale_tensor(
            load_image_array(pair.input_path), pair.input_path, "NoisyLR input"
        )
        target_tensor = _grayscale_tensor(
            load_image_array(pair.target_path), pair.target_path, "GT target"
        )
        input_height, input_width = input_tensor.shape[-2:]
        target_height, target_width = target_tensor.shape[-2:]
        expected_shape = (input_height * self.scale, input_width * self.scale)
        if (target_height, target_width) != expected_shape:
            raise ValueError(
                f"GT target must be {self.scale}x the input spatial size; input="
                f"{(input_height, input_width)}, target={(target_height, target_width)}, "
                f"expected={expected_shape}, pair={pair.pair_id}"
            )
        if self.transform is not None:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
            if not isinstance(input_tensor, torch.Tensor) or not isinstance(
                target_tensor, torch.Tensor
            ):
                raise TypeError("Paired transform must return two torch.Tensor values")
            if (
                input_tensor.ndim != 3
                or target_tensor.ndim != 3
                or input_tensor.shape[0] != 1
                or target_tensor.shape[0] != 1
            ):
                raise ValueError("Paired transform must return grayscale [1,H,W] tensors")
            if input_tensor.dtype != torch.float32 or target_tensor.dtype != torch.float32:
                raise ValueError("Paired transform must preserve torch.float32 dtype")
            if not torch.isfinite(input_tensor).all() or not torch.isfinite(
                target_tensor
            ).all():
                raise ValueError("Paired transform produced non-finite values")
            if target_tensor.shape[-2:] != tuple(
                dimension * self.scale for dimension in input_tensor.shape[-2:]
            ):
                raise ValueError(
                    f"Paired transform broke the {self.scale}x spatial relationship"
                )
        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": pair.input_path.name,
        }


class RestorationTestDataset(Dataset[Sample]):
    """Lazily load unpaired competition inputs while preserving filenames."""

    def __init__(self, input_paths: Sequence[Path]) -> None:
        self.input_paths = tuple(Path(path) for path in input_paths)

    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(self, index: int) -> Sample:
        path = self.input_paths[index]
        input_tensor = _grayscale_tensor(
            load_image_array(path), path, "Test NoisyLR input"
        )
        return {"input": input_tensor, "filename": path.name}


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Create a standard DataLoader with optional reproducible shuffle order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )
