"""Reading SWIM's targets under the tiles, their split, and the crops scored on them."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from ice_consistency.analysis.position import grid_offsets_from_tile_centre
from ice_consistency.configs import GRID_SPACING_KM
from ice_consistency.models import IceLabel
from mars_multisensor_dataset.common.maths.physics import METRES_PER_KM
from mars_multisensor_mae.configs.load import named_seed
from mars_multisensor_mae.dataset.loader import (
    padded_tokens,
    read_cached_batch,
    read_cached_tile,
)
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.store import DatasetBuild
from sklearn.model_selection import StratifiedGroupKFold

from dataset.models.swim import SwimBatch

TARGETS = {
    "1_5m": "consistency_1_5m",
    "5m": "consistency_5m",
    "rs": "radar_surface_power",
}

SQUARE_DEG = 10.0

REACH_M = GRID_SPACING_KM * METRES_PER_KM / 2

SWIM_POINTS = np.stack(grid_offsets_from_tile_centre(), -1).reshape(-1, 2)


def reached_swim_points(spots: np.ndarray) -> np.ndarray:
    """Return which of SWIM's points lie within reach of some measured patches.

    Args:
        spots: East and north of each measured patch, in metres. (S, 2)

    Returns:
        reached: Whether each point lies within REACH_M of a patch. (P * P,)
    """
    apart = np.linalg.norm(SWIM_POINTS[:, None] - spots[None], axis=-1)  # (P * P, S)
    return apart.min(1, initial=math.inf) <= REACH_M


def read_budget_crops_batch(
    identities: Sequence[str],
    build: DatasetBuild,
    layout: PatchLayout,
    budget: Mapping[str, int],
    targets: Mapping[str, np.ndarray],
    scale: float,
    cache: bool = True,
) -> SwimBatch:
    """Return each tile of a batch as one budget crop around a centre drawn anew.

    Args:
        identities: The tiles of the batch.
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        budget: How many patches of each instrument a crop keeps.
        targets: SWIM's value of every point for each target, keyed by the tile.
            (P * P, Z)
        scale: The most the budget is multiplied by, drawn once per batch from 1.
        cache: Whether a tile missing from the cache is kept there once cut.

    Returns:
        batch: The crops, their targets and the points within reach of their
            measured patches.
    """
    tokens, tiles = read_cached_batch(
        identities, build, layout, budget, scale, None, cache
    )
    spots = torch.cat([one.position[..., :2] for one in tokens.values()], 1).numpy()
    present = torch.cat([one.present for one in tokens.values()], 1).numpy()
    scored = [reached_swim_points(held[kept]) for held, kept in zip(spots, present)]
    return SwimBatch(
        tokens, tiles, np.stack([targets[tile] for tile in tiles]), np.stack(scored)
    )


def read_covering_crops_batch(
    identities: Sequence[str],
    build: DatasetBuild,
    layout: PatchLayout,
    budget: Mapping[str, int],
    targets: Mapping[str, np.ndarray],
    cache: bool = True,
) -> SwimBatch:
    """Return each tile of a batch as the budget crops that cover SWIM's points.

    Args:
        identities: The tiles of the batch.
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        budget: How many patches of each instrument a crop keeps.
        targets: SWIM's value of every point for each target, keyed by the tile.
            (P * P, Z)
        cache: Whether a tile missing from the cache is kept there once cut.

    Returns:
        batch: The crops, their targets and the points each predicts: those within
            reach of its measured patches, and those of no crop in the crop centred
            nearest them.
    """
    samples, tiles, scored = [], [], []
    for identity in identities:
        sample = read_cached_tile(build, identity, layout, cache)
        grounds = sample.measured_ground_spots  # (G, 2)
        apart = np.linalg.norm(SWIM_POINTS[:, None] - grounds[None], axis=-1)
        left = apart.min(1) <= REACH_M
        centres, reached = [], []
        # Each crop is centred on the spot nearest the first point no crop reaches yet
        while left.any() or not centres:
            first = int(np.argmax(left))
            centre = grounds[apart[first].argmin()]
            crop = sample.budget_cut(budget, centre)
            near = reached_swim_points(crop.measured_ground_spots)
            near[first] = True
            left &= ~near
            centres.append(centre)
            reached.append(near)
            samples.append(build.normalised_sample(crop))
        reached = np.stack(reached)  # (C, P * P)
        # A point no crop reaches is predicted by the crop centred nearest it
        lone = ~reached.any(0)
        nearest = np.linalg.norm(
            SWIM_POINTS[:, None] - np.stack(centres)[None], axis=-1
        )
        reached[nearest.argmin(1)[lone], np.flatnonzero(lone)] = True
        tiles += [identity] * len(reached)
        scored.append(reached)
    return SwimBatch(
        padded_tokens(samples),
        tiles,
        np.stack([targets[tile] for tile in tiles]),
        np.concatenate(scored),
    )


def balanced_kfold_tile_split(
    labels: Mapping[str, IceLabel], folds: int, seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Return the labelled tiles split by place into training, validation and test.

    Args:
        labels: Each tile's ice at both depths, keyed by the tile.
        folds: How many equal folds the tiles are dealt into, the first testing, the
            second validating and the rest training.
        seed: The run's seed, which fixes the deal.

    Returns:
        training: The training tiles.
        validation: The validation tiles.
        test: The test tiles.
    """
    tiles = sorted(labels)
    squares = [
        (
            math.floor(labels[tile].centre_lat / SQUARE_DEG),
            math.floor(labels[tile].centre_lon / SQUARE_DEG),
        )
        for tile in tiles
    ]
    strata = [(labels[tile].ice_1_5m, labels[tile].ice_5m) for tile in tiles]
    dealt = StratifiedGroupKFold(
        folds, shuffle=True, random_state=named_seed(seed, "benchmark/split")
    ).split(tiles, [str(one) for one in strata], [str(one) for one in squares])
    test, validation, *rest = (held for _, held in dealt)
    return tuple(
        [tiles[at] for at in sorted(held)]
        for held in (np.concatenate(rest), validation, test)
    )


def swim_targets_by_tile(labels: Mapping[str, IceLabel]) -> dict[str, np.ndarray]:
    """Return SWIM's value of every point of each tile for each target.

    Args:
        labels: Each tile's ice at both depths and its radar, keyed by the tile.

    Returns:
        targets: The values in TARGETS order, rows from the north and points from
            the west, keyed by the tile. (P * P, Z)
    """
    return {
        tile: np.stack([getattr(one, field) for field in TARGETS.values()], -1).astype(
            np.float32
        )
        for tile, one in labels.items()
    }
