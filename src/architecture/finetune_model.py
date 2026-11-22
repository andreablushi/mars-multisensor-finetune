"""The model the fine-tune protocol trains: the probe over a copy of the MAE."""

from __future__ import annotations

from copy import deepcopy

from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.architecture.models.tokens import Tokens
from mars_multisensor_mae.configs.load import named_rng
from torch import Tensor, nn

from architecture.components.attentive_probe import AttentiveProbe
from dataset.loader import TARGETS


class FinetunedModel(nn.Module):
    """The attentive probe over a copy of the MAE, its encoders trained too.

    Attributes:
        mae: A copy of the MAE without its decoders, its encoders and cross-sensor
            encoder left to train.
        probe: The attentive probe, reading every instrument's tokens.
    """

    def __init__(
        self, mae: CrossSensorMAE, dim: int, heads: int, depth: int, seed: int
    ) -> None:
        """Build the model over a copy of one MAE, its weights left as they are.

        Args:
            mae: The MAE, copied without its decoders and left untouched.
            dim: The token width the MAE hands its tokens at.
            heads: How many attention heads the probe runs.
            depth: How many cross-attention blocks the probe stacks.
            seed: What fixes the probe as drawn.
        """
        super().__init__()
        self.mae = deepcopy(mae, {id(mae.decoders): nn.ModuleDict()}).requires_grad_(
            False
        )
        self.mae.encoders.requires_grad_(True)
        self.mae.crossencoder.requires_grad_(True)
        for encoder in self.mae.encoders.values():
            encoder.checkpointed = False
        with named_rng(seed, "benchmark/probe"):
            self.probe = AttentiveProbe(dim, heads, depth, len(TARGETS))

    def forward(self, batch: dict[str, Tokens]) -> Tensor:
        """Return every point's prediction of each target over one batch.

        Args:
            batch: Each instrument's patches over the batch, on the device.

        Returns:
            prediction: One per point and target. (B, P * P, Z)
        """
        return self.probe(self.mae.embed_tokens(batch).patch_tokens_context())
