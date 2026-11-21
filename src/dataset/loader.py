"""The batches of covering crops SWIM's points are scored on, the splits and targets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from ice_consistency.analysis.position import grid_offsets_from_tile_centre
from ice_consistency.models import IceLabel
from mars_multisensor_mae.architecture.models.tokens import Tokens
from mars_multisensor_mae.configs.load import named_seed
from mars_multisensor_mae.dataset.loader import padded_tokens, read_cached_tile
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.store import DatasetBuild
from sklearn.model_selection import StratifiedGroupKFold

TARGETS = {
    "1_5m": "consistency_1_5m",
    "5m": "consistency_5m",
    "rs": "radar_surface_power",
}


def read_covering_crops_batch(
    identities: Sequence[str],
    build: DatasetBuild,
    layout: PatchLayout,
    budget: Mapping[str, int],
    reach_m: float,
    cache: bool = True,
) -> tuple[dict[str, Tokens], list[str], np.ndarray]:
    """Return each tile of a batch as the budget crops that cover SWIM's points.

    Args:
        identities: The tiles of the batch.
        build: The build the tiles are read from.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
        budget: How many patches of each instrument a crop keeps.
        reach_m: How near a read patch a point is scored, in metres.
        cache: Whether a tile missing from the cache is kept there once cut.

    Returns:
        batch: Each instrument's patches over every crop, keyed as ODE names it.
        identities: The tile each crop belongs to, a tile's crops side by side.
        scored: Which points each crop predicts: those within reach of its read
            patches, and those of no crop in the crop centred nearest them. (C, P * P)
    """
    points = np.stack(grid_offsets_from_tile_centre(), -1).reshape(-1, 2)  # (P * P, 2)
    samples, tiles, scored = [], [], []
    for identity in identities:
        sample = read_cached_tile(build, identity, layout, cache)
        grounds = sample.measured_ground_spots  # (G, 2)
        apart = np.linalg.norm(points[:, None] - grounds[None], axis=-1)  # (P * P, G)
        left = apart.min(1) <= reach_m
        centres, reached = [], []
        # Each crop is centred on the spot nearest the first point no crop reaches yet
        while left.any() or not centres:
            first = int(np.argmax(left))
            centre = grounds[apart[first].argmin()]
            crop = sample.budget_cut(budget, centre)
            spots = crop.measured_ground_spots
            near = (
                np.linalg.norm(points[:, None] - spots[None], axis=-1).min(1) <= reach_m
            )
            near[first] = True
            left &= ~near
            centres.append(centre)
            reached.append(near)
            samples.append(build.normalised_sample(crop))
        reached = np.stack(reached)  # (C, P * P)
        # A point no crop reaches is predicted by the crop centred nearest it
        lone = ~reached.any(0)
        nearest = np.linalg.norm(points[:, None] - np.stack(centres)[None], axis=-1)
        reached[nearest.argmin(1)[lone], np.flatnonzero(lone)] = True
        tiles += [identity] * len(reached)
        scored.append(reached)
    return padded_tokens(samples), tiles, np.concatenate(scored)


def tiles_split_by_square(
    labels: Mapping[str, IceLabel], folds: int, block_deg: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Return the labelled tiles split by place into training, validation and test.

    Args:
        labels: Each tile's ice at both depths, keyed by the tile.
        folds: How many equal folds the tiles are dealt into, the first testing, the
            second validating and the rest training.
        block_deg: How wide a square of the map is, in degrees, each one dealt whole.
        seed: The run's seed, which fixes the deal.

    Returns:
        training: The training tiles.
        validation: The validation tiles.
        test: The test tiles.
    """
    tiles = sorted(labels)
    squares = [
        (
            math.floor(labels[tile].centre_lat / block_deg),
            math.floor(labels[tile].centre_lon / block_deg),
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


def swim_targets_loader(labels: Mapping[str, IceLabel]) -> dict[str, np.ndarray]:
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
