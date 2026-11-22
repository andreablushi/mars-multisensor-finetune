"""How well a model's points fit SWIM's, target by target."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dataset.loader import TARGETS
from evaluation.models.metrics import IceScore, TilePredictions


def root_mean_squared_error(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Return the root mean squared error of one target over every point.

    Args:
        predicted: The prediction of every point. (T, P * P)
        truth: SWIM's value there. (T, P * P)

    Returns:
        rmse: The root of the mean squared difference.
    """
    return float(np.sqrt(np.sum((predicted - truth) ** 2) / truth.size))


def r_squared(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Return the coefficient of determination of one target over every point.

    Args:
        predicted: The prediction of every point. (T, P * P)
        truth: SWIM's value there. (T, P * P)

    Returns:
        r2: One less the squared error over the spread of the truth around its mean.
    """
    error = np.sum((predicted - truth) ** 2)
    return float(1 - error / np.sum((truth - truth.mean()) ** 2))


def ice_consistency_scores(
    predictions: TilePredictions,
    targets: Mapping[str, np.ndarray],
    share: str,
    protocol: str,
) -> list[IceScore]:
    """Return one model's RMSE and R2 of each target over every point of its tiles.

    Args:
        predictions: Every point's prediction of each target, keyed by tile.
            (P * P, Z)
        targets: SWIM's values there, keyed by tile over at least those tiles.
        share: The share of the training tiles the model was trained on.
        protocol: The model's protocol.

    Returns:
        scores: The RMSE and R2 of each target, in TARGETS order.
    """
    tiles = sorted(predictions)
    predicted, wanted = (
        np.stack([held[tile] for tile in tiles]) for held in (predictions, targets)
    )  # (T, P * P, Z)
    return [
        IceScore(share, protocol, name, metric, value)
        for at, name in enumerate(TARGETS)
        for metric, value in (
            ("RMSE", root_mean_squared_error(predicted[..., at], wanted[..., at])),
            ("R2", r_squared(predicted[..., at], wanted[..., at])),
        )
    ]
