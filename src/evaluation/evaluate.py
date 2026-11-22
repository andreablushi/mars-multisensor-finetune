"""Predicting the test tiles with every protocol's model, and logging the scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial

import numpy as np
import torch
from mars_multisensor_mae.dataset.loader import tile_loader
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.patches import drawn_arrays
from mars_multisensor_mae.dataset.store import DatasetBuild
from wandb.sdk.wandb_run import Run

from architecture.model import (
    PROTOCOLS,
    BenchmarkModel,
)
from dataset.loader import read_covering_crops_batch
from evaluation.metrics import rmse_and_r2
from evaluation.predict import crop_mean_predictions
from logs.benchmark import log_benchmark
from logs.images.ice import ice_image


def evaluate(
    models: Mapping[str, Mapping[str, BenchmarkModel]],
    build: DatasetBuild,
    test: Sequence[str],
    targets: Mapping[str, np.ndarray],
    instruments: Sequence[str],
    layout: PatchLayout,
    budget: Mapping[str, int],
    batch_size: int,
    tiles: Sequence[str],
    shown: str,
    seed: int,
    workers: int,
    device: torch.device,
    run: Run,
) -> None:
    """Log every model's test scores and the drawn tiles.

    Args:
        models: Each protocol's model, on the device, trained, keyed by the share
            of the tiles it was trained on, the largest last.
        build: The build the test tiles are read from.
        test: The test tiles.
        targets: SWIM's value of every point for each target, keyed by the
            tile over at least the test tiles. (P * P, Z)
        instruments: The instruments the models read.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        budget: How many patches of each instrument a test crop keeps.
        batch_size: How many tiles one batch holds, each read as its covering crops.
        tiles: The test tiles drawn as SWIM's grid beside the models', in
            qualitative order.
        shown: The instrument whose every observation of a tile is drawn beneath.
        seed: What fixes each fixed cut.
        workers: How many processes read tiles beside the models.
        device: Where the models run.
        run: The tracked run.
    """
    loader = tile_loader(
        build,
        test,
        layout.only(instruments),
        budget,
        1.0,
        seed,
        batch_size,
        workers,
        False,
        read_batch=partial(read_covering_crops_batch, targets=targets),
    )
    predictions = {
        share: {protocol: {} for protocol in held} for share, held in models.items()
    }
    still = next(protocol for protocol, tuned in PROTOCOLS.items() if not tuned)
    for held in models.values():
        for model in held.values():
            model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_predictions = crop_mean_predictions(models, batch, still, device)
            for share, held in batch_predictions.items():
                for protocol, by_tile in held.items():
                    predictions[share][protocol].update(by_tile)
    drawn = predictions[list(models)[-1]]
    by_tile = build.read_observation_metadata()
    images = [
        ice_image(
            drawn_arrays(build, shown, by_tile[tile][shown], layout),
            targets[tile],
            {protocol: held[tile] for protocol, held in drawn.items()},
        )
        for tile in tiles
    ]
    scores = {share: rmse_and_r2(held, targets) for share, held in predictions.items()}
    log_benchmark(run, scores, images)
