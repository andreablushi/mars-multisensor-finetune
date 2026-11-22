"""Building the pretrained model a config and a build settle, and the models over it."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from mars_multisensor_mae.architecture.load import built_cross_sensor_mae
from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.dataset.models.patch import PatchLayout
from mars_multisensor_mae.dataset.store import DatasetBuild
from torch import Tensor

from architecture import BenchmarkModel
from architecture.finetune_model import FinetunedModel
from architecture.probe_model import AttentiveProbeModel
from configs.schema import Config

PROTOCOLS = {"attentive-probe": AttentiveProbeModel, "fine-tune": FinetunedModel}


def built_pretrained_model(
    config: Config, build: DatasetBuild
) -> tuple[CrossSensorMAE, torch.device, PatchLayout]:
    """Return the pretrained run's model a build's patch layout settles, on its device.

    Args:
        config: What the run reads and the model it benchmarks.
        build: The build whose instruments settle the patch layout.

    Returns:
        model: The model, on its device, its weights as drawn.
        device: Where it runs.
        layout: How each instrument is cut into patches, and the shape the model
            reads one in.
    """
    return built_cross_sensor_mae(
        build,
        config.model.instruments,
        config.dataset.patchsize,
        config.training.acquisition_in_encoders,
        config.model.encoder_dim,
        config.model.encoder_heads,
        config.model.encoder_depth,
        config.model.crossencoder_depth,
        config.model.decoder_dim,
        config.model.decoder_heads,
        config.model.decoder_depth,
        config.dataset.seed,
    )


def built_benchmark_model(
    config: Config,
    mae: CrossSensorMAE,
    device: torch.device,
    heads: Mapping[str, Mapping[str, Tensor]] | None = None,
) -> dict[str, BenchmarkModel]:
    """Return each protocol's model over one MAE, its probes drawn alike.

    Args:
        config: What the run reads and the model it benchmarks.
        mae: The MAE, its weights left as they are, shared by the frozen protocol.
        device: Where the models run.
        heads: The trained weights to load, keyed by protocol, then by parameter,
            or None to keep them as drawn.

    Returns:
        models: The attentive probe over the frozen MAE and over the MAE with its
            encoders trained, on the device, keyed by protocol.
    """
    models = {}
    for protocol, model_class in PROTOCOLS.items():
        model = model_class(
            mae,
            config.model.encoder_dim,
            config.model.encoder_heads,
            config.benchmark.probe.depth,
            config.dataset.seed,
        ).to(device)
        if heads:
            model.load_state_dict(model.state_dict() | dict(heads[protocol]))
        models[protocol] = model
    return models
