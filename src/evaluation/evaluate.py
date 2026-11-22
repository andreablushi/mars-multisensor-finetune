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

from architecture import BenchmarkModel
from dataset.loader import read_covering_crops_batch
from evaluation.metrics import ice_consistency_scores
from evaluation.predict import ice_consistency_predictions_by_model
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
    flat = {
        (share, protocol): model
        for share, held in models.items()
        for protocol, model in held.items()
    }
    for model in flat.values():
        model.eval()
    predictions = ice_consistency_predictions_by_model(flat, loader, device)
    drawn = list(models)[-1]
    by_tile = build.read_observation_metadata()
    images = [
        ice_image(
            drawn_arrays(build, shown, by_tile[tile][shown], layout),
            targets[tile],
            {
                protocol: held[tile]
                for (share, protocol), held in predictions.items()
                if share == drawn
            },
        )
        for tile in tiles
    ]
    scores = [
        score
        for (share, protocol), held in predictions.items()
        for score in ice_consistency_scores(held, targets, share, protocol)
    ]
    log_benchmark(run, scores, images)
