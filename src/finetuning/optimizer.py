"""The optimizer and learning rate schedule one protocol's model steps with."""

from __future__ import annotations

import torch
from mars_multisensor_mae.training.optimizer import (
    decayed_parameter_groups,
    grouped_optimizer_and_schedule,
)
from torch.optim.lr_scheduler import LambdaLR

from architecture import BenchmarkModel


def optimizer_and_schedule(
    model: BenchmarkModel,
    head_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    num_steps: int,
) -> tuple[torch.optim.AdamW, LambdaLR]:
    """Return AdamW over a model's trained weights, the probe and the MAE apart.

    Args:
        model: The model, its frozen weights left out.
        head_learning_rate: The probe's peak learning rate.
        encoder_learning_rate: The encoders' peak learning rate.
        weight_decay: The AdamW weight decay of every matrix.
        warmup_steps: How many steps the rate climbs before the cosine decay.
        num_steps: How many steps the schedule runs for.

    Returns:
        optimizer: AdamW with torch's betas, the probe at the head rate and the MAE at
            the encoder rate, vectors undecayed.
        scheduler: The warmup and cosine decay its rates follow.
    """
    trained = [one for one in model.mae.parameters() if one.requires_grad]
    return grouped_optimizer_and_schedule(
        decayed_parameter_groups(
            model.probe.parameters(), head_learning_rate, weight_decay
        )
        + decayed_parameter_groups(trained, encoder_learning_rate, weight_decay),
        (0.9, 0.999),
        None,
        warmup_steps,
        num_steps,
    )
