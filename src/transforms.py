"""Spatially aligned paired preprocessing for LR/GT tensors."""

from collections.abc import Callable, Sequence

import torch


PairedTransform = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def _crop_size_tuple(crop_size: int | tuple[int, int]) -> tuple[int, int]:
    size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if len(size) != 2 or min(size) < 1:
        raise ValueError(f"crop_size must contain two positive dimensions, got {size}")
    return int(size[0]), int(size[1])


def _validate_pair(
    input_tensor: torch.Tensor, target_tensor: torch.Tensor, scale: int
) -> None:
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    if input_tensor.ndim != 3 or target_tensor.ndim != 3:
        raise ValueError("Paired transforms expect channel-first [C,H,W] tensors")
    if input_tensor.shape[0] != target_tensor.shape[0]:
        raise ValueError("Input and target channel counts must match")
    expected_target = (
        input_tensor.shape[-2] * scale,
        input_tensor.shape[-1] * scale,
    )
    if target_tensor.shape[-2:] != expected_target:
        raise ValueError(
            f"Target must be {scale}x the input spatial size; input="
            f"{tuple(input_tensor.shape[-2:])}, target={tuple(target_tensor.shape[-2:])}"
        )


def aligned_paired_crop(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    crop_size: int | tuple[int, int],
    y: int,
    x: int,
    scale: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop LR at ``(y,x)`` and GT at the exactly scaled coordinates."""
    _validate_pair(input_tensor, target_tensor, scale)
    crop_height, crop_width = _crop_size_tuple(crop_size)
    input_height, input_width = input_tensor.shape[-2:]
    maximum_y = input_height - crop_height
    maximum_x = input_width - crop_width
    if maximum_y < 0 or maximum_x < 0:
        raise ValueError(
            f"LR crop {(crop_height, crop_width)} exceeds input size "
            f"{(input_height, input_width)}"
        )
    if not 0 <= y <= maximum_y or not 0 <= x <= maximum_x:
        raise ValueError(
            f"Crop origin {(y, x)} is outside valid range "
            f"y=[0,{maximum_y}], x=[0,{maximum_x}]"
        )
    target_y, target_x = y * scale, x * scale
    return (
        input_tensor[:, y : y + crop_height, x : x + crop_width].contiguous(),
        target_tensor[
            :,
            target_y : target_y + crop_height * scale,
            target_x : target_x + crop_width * scale,
        ].contiguous(),
    )


def apply_paired_geometry(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    *,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    rotation_k: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply identical flips and a multiple-of-90-degree rotation to a pair."""
    rotation_k %= 4
    transformed_input, transformed_target = input_tensor, target_tensor
    if horizontal_flip:
        transformed_input = torch.flip(transformed_input, dims=(-1,))
        transformed_target = torch.flip(transformed_target, dims=(-1,))
    if vertical_flip:
        transformed_input = torch.flip(transformed_input, dims=(-2,))
        transformed_target = torch.flip(transformed_target, dims=(-2,))
    if rotation_k:
        transformed_input = torch.rot90(
            transformed_input, k=rotation_k, dims=(-2, -1)
        )
        transformed_target = torch.rot90(
            transformed_target, k=rotation_k, dims=(-2, -1)
        )
    return transformed_input.contiguous(), transformed_target.contiguous()


class PairedRandomCrop:
    """Random aligned LR/GT crop including every legal boundary coordinate."""

    def __init__(
        self,
        crop_size: int | tuple[int, int] = 64,
        scale: int = 2,
        generator: torch.Generator | None = None,
    ) -> None:
        self.crop_size = _crop_size_tuple(crop_size)
        self.scale = scale
        self.generator = generator

    def __call__(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_height, input_width = input_tensor.shape[-2:]
        crop_height, crop_width = self.crop_size
        maximum_y = input_height - crop_height
        maximum_x = input_width - crop_width
        if maximum_y < 0 or maximum_x < 0:
            raise ValueError(
                f"LR crop {self.crop_size} exceeds input size "
                f"{(input_height, input_width)}"
            )
        y = int(
            torch.randint(0, maximum_y + 1, (), generator=self.generator).item()
        )
        x = int(
            torch.randint(0, maximum_x + 1, (), generator=self.generator).item()
        )
        return aligned_paired_crop(
            input_tensor, target_tensor, self.crop_size, y, x, self.scale
        )


class PairedRandomGeometricAugmentation:
    """Random paired flips and uniformly selected right-angle rotation."""

    def __init__(
        self,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        rotate90: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        for name, probability in (
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("vertical_flip_probability", vertical_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.rotate90 = rotate90
        self.generator = generator

    def __call__(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizontal = bool(
            torch.rand((), generator=self.generator).item()
            < self.horizontal_flip_probability
        )
        vertical = bool(
            torch.rand((), generator=self.generator).item()
            < self.vertical_flip_probability
        )
        rotation_k = (
            int(torch.randint(0, 4, (), generator=self.generator).item())
            if self.rotate90
            else 0
        )
        return apply_paired_geometry(
            input_tensor,
            target_tensor,
            horizontal_flip=horizontal,
            vertical_flip=vertical,
            rotation_k=rotation_k,
        )


class PairedCompose:
    """Apply paired transforms sequentially."""

    def __init__(self, transforms: Sequence[PairedTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            input_tensor, target_tensor = transform(input_tensor, target_tensor)
        return input_tensor, target_tensor


def create_training_transform(
    crop_size: int | tuple[int, int] = 64,
    scale: int = 2,
    augment: bool = True,
    horizontal_flip_probability: float = 0.5,
    vertical_flip_probability: float = 0.5,
    rotate90: bool = True,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> PairedCompose:
    """Construct the standard aligned crop then spatial augmentation pipeline.

    Supplying ``seed`` or ``generator`` is useful for deterministic single-worker
    tests. With neither supplied, PyTorch's process-local RNG is used, including
    DataLoader worker seeds. Generator state advances on every sample access.
    """
    if generator is not None and seed is not None:
        raise ValueError("Provide either generator or seed, not both")
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    transforms: list[PairedTransform] = [
        PairedRandomCrop(crop_size=crop_size, scale=scale, generator=generator)
    ]
    if augment:
        transforms.append(
            PairedRandomGeometricAugmentation(
                horizontal_flip_probability=horizontal_flip_probability,
                vertical_flip_probability=vertical_flip_probability,
                rotate90=rotate90,
                generator=generator,
            )
        )
    return PairedCompose(transforms)
