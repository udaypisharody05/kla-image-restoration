"""x8 geometric self-ensemble (dihedral-8 test-time augmentation) for square-image
restoration models.

Inference-only: no retraining, no change to model/loss/optimizer/scheduler. For each
of the 8 symmetries of a square image (the dihedral group D4: 4 rotations, each with
or without a horizontal flip), transform the LR input, run the model, undo the
transform on the HR output, and average the 8 raw (unclipped) predictions. Metric-time
clipping, if any, is entirely the caller's responsibility -- this module never clips.
"""

from collections.abc import Iterator
import torch


def _rotate(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate the last two dims by 90*k degrees. k is taken mod 4 by torch.rot90."""
    return torch.rot90(x, k, dims=(-2, -1))


def _hflip(x: torch.Tensor) -> torch.Tensor:
    """Flip along the width (last) dimension. Its own exact inverse."""
    return torch.flip(x, dims=(-1,))


def d4_transforms() -> Iterator[tuple[bool, int]]:
    """The 8 unique elements of the dihedral group D4, as (flip, rotation_k) pairs.

    4 rotations (k=0,1,2,3) x {no flip, horizontal flip} = 8 unique combinations --
    every square-image symmetry, none repeated.
    """
    for flip in (False, True):
        for k in (0, 1, 2, 3):
            yield flip, k


def forward_transform(x: torch.Tensor, flip: bool, k: int) -> torch.Tensor:
    """Apply flip (if any), then rotate by 90*k degrees."""
    if flip:
        x = _hflip(x)
    return _rotate(x, k)


def inverse_transform(x: torch.Tensor, flip: bool, k: int) -> torch.Tensor:
    """Exact inverse of ``forward_transform(., flip, k)``.

    Un-rotate first (by -k), then un-flip (flip is self-inverse, so applying the
    same flip again undoes it) -- the reverse order of the forward transform.
    """
    x = _rotate(x, -k)
    if flip:
        x = _hflip(x)
    return x


@torch.inference_mode()
def predict_x8(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Average the model's prediction over all 8 D4 transforms of *inputs*.

    ``inputs``: ``[N, C, H, W]`` (any batch size, any channel count -- this
    project uses grayscale ``C=1`` on square ``H==W`` images, but nothing here
    assumes that). Returns the raw (unclipped) mean prediction, same dtype and
    device as the model's output. Individual per-transform predictions are never
    clipped -- only the final average is returned, so any clipping (e.g. for
    PSNR/SSIM) must happen after this call, on the averaged result.
    """
    was_training = model.training
    model.eval()
    try:
        predictions = []
        for flip, k in d4_transforms():
            transformed_input = forward_transform(inputs, flip, k)
            transformed_output = model(transformed_input)
            predictions.append(inverse_transform(transformed_output, flip, k))
        return torch.stack(predictions, dim=0).mean(dim=0)
    finally:
        model.train(was_training)
