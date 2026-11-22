"""Predicting SWIM's points with one model, and with several over one split."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from mars_multisensor_mae.architecture.models.context import Context
from torch.utils.data import DataLoader

from architecture import BenchmarkModel
from dataset.models.swim import SwimBatch
from evaluation.models.metrics import TilePredictions


def predict_ice_consistency(
    model: BenchmarkModel, batch: SwimBatch, context: Context
) -> TilePredictions:
    """Return each tile's prediction by one model, a point the mean of its crops'.

    Args:
        model: The model, on the device.
        batch: The crops, a tile's crops side by side, and the points each predicts.
        context: Every instrument's tokens of the crops, as the model's MAE reads them.

    Returns:
        predictions: Each tile's prediction of every point and target. (P * P, Z)
    """
    # Each point is the mean of the crops of its tile that predict it
    weights = torch.as_tensor(batch.scored, dtype=torch.float32).unsqueeze(-1)
    crops = {
        tile: [at for at, one in enumerate(batch.tiles) if one == tile]
        for tile in dict.fromkeys(batch.tiles)
    }
    predicted = model.probe(context).float().cpu() * weights
    return {
        tile: (predicted[at].sum(0) / weights[at].sum(0)).numpy()
        for tile, at in crops.items()
    }


def ice_consistency_predictions_by_model(
    models: Mapping[tuple[str, str], BenchmarkModel],
    loader: DataLoader,
    device: torch.device,
) -> dict[tuple[str, str], TilePredictions]:
    """Return every model's prediction of each tile of one split.

    Args:
        models: Each model, on the device, keyed by share and protocol.
        loader: The split, in batches, each tile read as the budget crops covering
            SWIM's points.
        device: Where the models run.

    Returns:
        predictions: Each model's prediction of every tile, keyed as the models.
    """
    predictions = {key: {} for key in models}
    with torch.no_grad():
        for batch in loader:
            tokens = {name: one.to(device) for name, one in batch.tokens.items()}
            with torch.autocast(device.type, dtype=torch.float16):
                # A frozen MAE is shared by every share, so each MAE reads once
                contexts = {}
                for key, model in models.items():
                    if id(model.mae) not in contexts:
                        encoded = model.mae.embed_tokens(tokens)
                        contexts[id(model.mae)] = encoded.patch_tokens_context()
                    predictions[key].update(
                        predict_ice_consistency(model, batch, contexts[id(model.mae)])
                    )
    return predictions
