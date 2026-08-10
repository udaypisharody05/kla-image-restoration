"""Weighted averaging of raw predictions from multiple checkpoints (model ensembling).

Inference-only: no retraining, no change to any individual model. Combines
already-computed raw (unclipped) predictions from two or more checkpoints into
one ensemble prediction via a weighted arithmetic mean. Never clips -- metric-time
clipping, if any, is entirely the caller's responsibility, exactly like ``src/tta.py``.
"""

import torch


def weighted_average_predictions(
    predictions: list[torch.Tensor], weights: list[float] | None = None
) -> torch.Tensor:
    """Weighted arithmetic mean of two or more raw predictions of identical shape.

    ``weights`` defaults to equal weighting and need not sum to 1 -- they are
    normalized internally (divided by their sum) as long as every weight is
    positive. Batch/channel/spatial dimensions are preserved unchanged; the
    output is never clipped.
    """
    if len(predictions) < 2:
        raise ValueError(f"Need at least 2 predictions to ensemble, got {len(predictions)}")

    first_shape = predictions[0].shape
    for index, prediction in enumerate(predictions):
        if prediction.shape != first_shape:
            raise ValueError(
                f"Shape mismatch: prediction[0] has shape {tuple(first_shape)}, "
                f"prediction[{index}] has shape {tuple(prediction.shape)}"
            )

    if weights is None:
        weights = [1.0] * len(predictions)
    if len(weights) != len(predictions):
        raise ValueError(
            f"Got {len(predictions)} predictions but {len(weights)} weights -- counts must match"
        )
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"All weights must be positive, got {weights}")

    weight_sum = sum(weights)
    normalized_weights = [weight / weight_sum for weight in weights]

    result = normalized_weights[0] * predictions[0]
    for weight, prediction in zip(normalized_weights[1:], predictions[1:]):
        result = result + weight * prediction
    return result
