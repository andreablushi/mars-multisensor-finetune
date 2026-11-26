"""The builds and SWIM's labels a run reads, off the platform's store or off disk."""

from __future__ import annotations

import io
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import NoReturn

from ice_consistency.analysis.artifacts import ICE_LABELS
from ice_consistency.configs import ICE_LABELS_NAME
from ice_consistency.models import IceLabel
from mars_multisensor_dataset.common.disk import parquet
from mars_multisensor_mae.dataset.store import DatasetBuild

from configs.paths import build_root
from configs.schema import Config

DIGITALHUB = "digitalhub"

DISK = "disk"


def missing_object(root: Path, path: str) -> NoReturn:
    """Fail on an object a build read off disk alone does not hold.

    Args:
        root: Where the build sits.
        path: The object, relative to the build.

    Raises:
        FileNotFoundError: Always, naming the object.
    """
    raise FileNotFoundError(f"{root / path} is missing, and dataset.store={DISK}")


def published_dataset_build(config: Config, build: str) -> DatasetBuild:
    """Return one published build of the dataset, read as dataset.store says.

    Args:
        config: The run's config.
        build: The build to read, published as <version>-dataset-<build>.

    Returns:
        build: The build, off disk where held and from the store otherwise.
    """
    return published_build(config, f"{config.dataset.version}-dataset-{build}")


def published_build(config: Config, name: str, subfolder: str = "") -> DatasetBuild:
    """Return one published build, read from a folder of it as dataset.store says.

    Args:
        config: The run's config.
        name: The artifact the build is published in.
        subfolder: The folder of the artifact the build sits in, or "" for its top.

    Returns:
        build: The build, off disk where held and from the store otherwise, which
            cuts its missing tiles only when read from the store.

    Raises:
        ValueError: When dataset.store names neither the platform nor the disk.
    """
    root = build_root(name, config.dataset.root) / subfolder
    store = config.dataset.store
    if store == DISK:
        fetch = partial(missing_object, root)
    elif store == DIGITALHUB:
        from dhub.store import published_fetch

        fetch = published_fetch(config, name, subfolder)
    else:
        raise ValueError(f"dataset.store={store} is not {DIGITALHUB} or {DISK}")
    ready = config.dataset.ready_root
    held = DatasetBuild(
        root=root,
        fetch=fetch,
        clips=config.dataset.clip,
        sharad_atmosphere_cropping=config.dataset.sharad_atmosphere_cropping,
        acquisition={
            name: tuple(fields) if config.training.acquisition else ()
            for name, fields in config.dataset.acquisition.items()
        },
        ready_root=None if ready is None else build_root(name, ready) / subfolder,
        cuts=store == DIGITALHUB,
    )
    if config.dataset.stats_version != config.dataset.version:
        trained = replace(
            config,
            dataset=replace(config.dataset, version=config.dataset.stats_version),
        )
        held.stats = published_dataset_build(trained, config.dataset.build).read_stats()
    return held


def published_ice_labels(config: Config) -> dict[str, IceLabel]:
    """Return the ice SWIM finds under every tile it covers, read as dataset.store says.

    Args:
        config: The run's config.

    Returns:
        labels: Each tile's ice at both depths, keyed by the tile.
    """
    name = f"{config.dataset.version}-ice-consistency"
    if config.dataset.store == DISK:
        held = build_root(name, config.dataset.root) / ICE_LABELS_NAME
    else:
        from dhub.store import published_fetch

        held = io.BytesIO(published_fetch(config, name)(ICE_LABELS_NAME))
    rows = parquet.read_rows(IceLabel, ICE_LABELS, held)
    return {one.tile: one for one in rows}
