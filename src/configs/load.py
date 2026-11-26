"""Reading a benchmark's config: the pretrained run's logged values and its yamls."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, fields

from mars_multisensor_mae.configs.schema import ModelConfig
from mars_multisensor_mae.logs.evaluation import EVALUATION
from mars_multisensor_mae.logs.tracker import TRAINING, offline_tracking, tracked_run
from omegaconf import OmegaConf

from configs.paths import CONFIGS_ROOT, logged_config_path
from configs.schema import Config, DatasetConfig, TrainingConfig

PRETRAINED = {
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
}

LOGGED = ("dataset", "model", "training", "evaluation")

PLATFORM_KEYS = {"root", "store", "ready_root"}


def logged_pretrained_config(run_name: str) -> dict[str, object]:
    """Return the config the pretrained run's training or evaluation run logged.

    Args:
        run_name: What the pretrained run is called, "random" for one never trained.

    Returns:
        config: The logged config, read offline from the file the prepare step wrote.

    Raises:
        ValueError: When no training or evaluation run of the name is tracked.
    """
    if offline_tracking():
        return json.loads(logged_config_path(run_name).read_text())
    tracked = tracked_run(run_name, TRAINING) or tracked_run(run_name, EVALUATION)
    if tracked is None:
        raise ValueError(f"{run_name}: no training or evaluation run is tracked")
    return tracked.config


def load_config(
    run_name: str, overrides: Sequence[str] = ()
) -> tuple[Config, dict[str, object]]:
    """Return what one benchmark is settled from, and the config its run records.

    Args:
        run_name: What the pretrained run is called, its training run read, or its
            evaluation run for one never trained, such as "random".
        overrides: What to change, as dotted keys: `benchmark.probe.depth=2`.

    Returns:
        config: The benchmark's choices, every key checked against the schema, the
            clip bounds W&B hands back as whole numbers read as floats, the
            PLATFORM_KEYS this platform's own and never the logged ones.
        recorded: The pretrained run's logged sections with the config over them,
            as the benchmark run logs it.
    """
    held = logged_pretrained_config(run_name)
    logged = {name: held.get(name, {}) for name in LOGGED}
    logged["dataset"]["clip"] = {
        name: [None if one is None else float(one) for one in bounds]
        for name, bounds in logged["dataset"]["clip"].items()
    }
    read = {
        name: {
            key: value
            for key, value in logged[name].items()
            if key in {one.name for one in fields(schema)} - PLATFORM_KEYS
        }
        for name, schema in PRETRAINED.items()
    } | {
        name: OmegaConf.load(CONFIGS_ROOT / f"{name}.yaml")
        for name in ("benchmark", "digitalhub")
    }
    config = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(Config),
            OmegaConf.merge(read, OmegaConf.from_dotlist(list(overrides))),
        )
    )
    recorded = OmegaConf.to_container(OmegaConf.merge(logged, asdict(config)))
    return config, recorded
