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


def gradient_energy_map(input_tensor: torch.Tensor) -> torch.Tensor:
    """Simple per-pixel gradient-magnitude score, summed over channels.

    Central-difference approximation of |dI/dx| + |dI/dy| (a cheap Sobel-like
    proxy for local high-frequency content), zero-padded at the border so the
    returned map has the same ``[H,W]`` shape as the input. Used only to
    *rank* candidate crop origins for informative-patch sampling
    (``sample_informative_crop_origin``) -- not a training signal itself.
    """
    if input_tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got shape {tuple(input_tensor.shape)}")
    channels = input_tensor.to(torch.float32)
    height, width = channels.shape[-2:]
    dx = torch.zeros((channels.shape[0], height, width), dtype=torch.float32)
    dy = torch.zeros((channels.shape[0], height, width), dtype=torch.float32)
    dx[:, :, 1:] = torch.abs(channels[:, :, 1:] - channels[:, :, :-1])
    dy[:, 1:, :] = torch.abs(channels[:, 1:, :] - channels[:, :-1, :])
    return (dx + dy).sum(dim=0)


def sample_informative_crop_origin(
    input_tensor: torch.Tensor,
    crop_size: int | tuple[int, int],
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Sample a crop origin ``(y, x)`` with probability proportional to the
    summed gradient energy of the crop window it defines.

    Every valid origin (the window fits entirely inside ``input_tensor``) has
    a positive chance of being chosen -- this is weighted-random sampling
    toward high-information regions, not a deterministic argmax -- so the
    same image can still yield different informative crops across epochs
    while remaining biased away from flat, low-information regions. Window
    sums are computed in ``O(H*W)`` via a 2D prefix-sum table (integral
    image), not one pass per candidate origin, so this stays cheap enough to
    run inside a ``Dataset.__getitem__``.
    """
    crop_height, crop_width = _crop_size_tuple(crop_size)
    input_height, input_width = input_tensor.shape[-2:]
    maximum_y = input_height - crop_height
    maximum_x = input_width - crop_width
    if maximum_y < 0 or maximum_x < 0:
        raise ValueError(
            f"LR crop {(crop_height, crop_width)} exceeds input size "
            f"{(input_height, input_width)}"
        )
    energy = gradient_energy_map(input_tensor)
    # Zero-padded prefix-sum table: integral[i, j] = sum of energy[:i, :j].
    integral = torch.zeros((input_height + 1, input_width + 1), dtype=torch.float32)
    integral[1:, 1:] = torch.cumsum(torch.cumsum(energy, dim=0), dim=1)
    bottom_right = integral[crop_height : crop_height + maximum_y + 1, crop_width : crop_width + maximum_x + 1]
    top_right = integral[0 : maximum_y + 1, crop_width : crop_width + maximum_x + 1]
    bottom_left = integral[crop_height : crop_height + maximum_y + 1, 0 : maximum_x + 1]
    top_left = integral[0 : maximum_y + 1, 0 : maximum_x + 1]
    window_sums = bottom_right - top_right - bottom_left + top_left
    # A small positive floor keeps every origin sampleable (nonzero
    # probability) even for a perfectly flat window, instead of only ever
    # picking among whichever windows happen to have nonzero energy.
    weights = window_sums.reshape(-1) + 1e-6
    flat_index = int(torch.multinomial(weights, 1, generator=generator).item())
    y = flat_index // (maximum_x + 1)
    x = flat_index % (maximum_x + 1)
    return y, x


class PairedHardPatchCrop:
    """Aligned LR/GT crop biased toward high-gradient-energy regions.

    Uses ``sample_informative_crop_origin`` on the LR tensor, then crops both
    LR and GT at the scaled-consistent coordinates via the same
    ``aligned_paired_crop`` helper ``PairedRandomCrop`` uses -- so alignment
    can never diverge between the two crop strategies.
    """

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
        y, x = sample_informative_crop_origin(input_tensor, self.crop_size, self.generator)
        return aligned_paired_crop(input_tensor, target_tensor, self.crop_size, y, x, self.scale)


class PairedMixedCrop:
    """Per-sample mixture of ``PairedHardPatchCrop`` and ``PairedRandomCrop``.

    With probability ``hard_patch_prob`` (default 0.5) use the
    gradient-energy-weighted informative crop; otherwise fall back to a plain
    uniform-random crop -- so training is never restricted to exclusively
    high-gradient regions (per the project spec: mix, don't replace).
    ``hard_patch_prob=0.0``/``1.0`` are valid and select one strategy
    unconditionally. The mixture decision itself is drawn from the same
    ``generator``, so the whole pipeline (mixture choice + crop origin) is a
    pure function of the generator's seed and call sequence.
    """

    def __init__(
        self,
        crop_size: int | tuple[int, int] = 64,
        scale: int = 2,
        hard_patch_prob: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> None:
        if not 0.0 <= hard_patch_prob <= 1.0:
            raise ValueError(f"hard_patch_prob must be between 0 and 1, got {hard_patch_prob}")
        self.hard_patch_prob = hard_patch_prob
        self.generator = generator
        self.hard_patch_crop = PairedHardPatchCrop(crop_size=crop_size, scale=scale, generator=generator)
        self.random_crop = PairedRandomCrop(crop_size=crop_size, scale=scale, generator=generator)

    def __call__(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_hard_patch = bool(
            torch.rand((), generator=self.generator).item() < self.hard_patch_prob
        )
        crop = self.hard_patch_crop if use_hard_patch else self.random_crop
        return crop(input_tensor, target_tensor)


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
    hard_patch_sampling: bool = False,
    hard_patch_prob: float = 0.5,
) -> PairedCompose:
    """Construct the standard aligned crop then spatial augmentation pipeline.

    Supplying ``seed`` or ``generator`` is useful for deterministic single-worker
    tests. With neither supplied, PyTorch's process-local RNG is used, including
    DataLoader worker seeds. Generator state advances on every sample access.

    ``hard_patch_sampling=False`` (the default) uses plain ``PairedRandomCrop``,
    byte-for-byte the historical behavior. When ``True``, ``PairedMixedCrop``
    is used instead: with probability ``hard_patch_prob`` the crop origin is
    drawn weighted toward high-gradient-energy regions
    (``sample_informative_crop_origin``), otherwise a plain uniform-random crop
    is used, exactly like before -- see ``src/transforms.py::PairedMixedCrop``.
    """
    if generator is not None and seed is not None:
        raise ValueError("Provide either generator or seed, not both")
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    crop: PairedTransform
    if hard_patch_sampling:
        crop = PairedMixedCrop(
            crop_size=crop_size, scale=scale, hard_patch_prob=hard_patch_prob, generator=generator
        )
    else:
        crop = PairedRandomCrop(crop_size=crop_size, scale=scale, generator=generator)
    transforms: list[PairedTransform] = [crop]
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
