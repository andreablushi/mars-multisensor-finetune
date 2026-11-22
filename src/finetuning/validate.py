"""Scoring every protocol's model on the validation tiles, and keeping the best."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from mars_multisensor_mae.logs.tracker import console_logger
from torch import Tensor
from torch.utils.data import DataLoader

from architecture import BenchmarkModel
from evaluation.metrics import ice_consistency_scores
from evaluation.models.metrics import IceScore
from evaluation.predict import ice_consistency_predictions_by_model

log = console_logger(__name__)


def validation_scores(
    models: Mapping[str, BenchmarkModel],
    loader: DataLoader,
    targets: Mapping[str, np.ndarray],
    share: str,
    device: torch.device,
) -> list[IceScore]:
    """Return each model's scores over one split, read as the test reads it.

    Args:
        models: Each protocol's model, on the device.
        loader: The split, in batches, each tile read as the budget crops covering
            SWIM's points.
        targets: SWIM's value of every point for each target, keyed by the tile.
            (P * P, Z)
        share: What the share of the tiles trained on is logged as.
        device: Where the models run.

    Returns:
        scores: Each protocol's RMSE and R2 of every target.
    """
    predictions = ice_consistency_predictions_by_model(
        {(share, protocol): model for protocol, model in models.items()},
        loader,
        device,
    )
    return [
        score
        for (_, protocol), held in predictions.items()
        for score in ice_consistency_scores(held, targets, share, protocol)
    ]


def best_validation_weights(
    models: Mapping[str, BenchmarkModel],
    scores: Sequence[IceScore],
    best: Mapping[str, float],
    kept: Mapping[str, dict[str, Tensor]],
    share: str,
    step: int,
) -> tuple[dict[str, float], dict[str, dict[str, Tensor]]]:
    """Return each model's best mean validation R2, and its trained weights there.

    Args:
        models: Each protocol's model.
        scores: The validation's scores, each protocol's RMSE and R2 of every target.
        best: Each protocol's best mean R2 so far.
        kept: Each protocol's trained parameters at its best so far.
        share: What the share of the tiles trained on is logged as.
        step: The step validated at.

    Returns:
        best: Each protocol's best mean R2, this validation included.
        kept: Each protocol's trained parameters at that best, keyed by parameter.
    """
    best, kept = dict(best), dict(kept)
    for protocol, model in models.items():
        fit = float(
            np.mean(
                [
                    one.value
                    for one in scores
                    if one.protocol == protocol and one.metric == "R2"
                ]
            )
        )
        log.info("%s %s step %d validation R2 %.4f", protocol, share, step, fit)
        if fit > best[protocol]:
            best[protocol] = fit
            kept[protocol] = {
                name: one.detach().cpu().clone()
                for name, one in model.named_parameters()
                if one.requires_grad
            }
    return best, kept
