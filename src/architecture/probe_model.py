"""The model the attentive-probe protocol trains: the probe over the frozen MAE."""

from __future__ import annotations

from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.architecture.models.tokens import Tokens
from mars_multisensor_mae.configs.load import named_rng
from torch import Tensor, nn

from architecture.components.attentive_probe import AttentiveProbe
from dataset.loader import TARGETS


class AttentiveProbeModel(nn.Module):
    """The attentive probe over the MAE, which stays frozen whole and shared.

    Attributes:
        mae: The MAE itself, frozen whole.
        probe: The attentive probe, reading every instrument's tokens.
    """

    def __init__(
        self, mae: CrossSensorMAE, dim: int, heads: int, depth: int, seed: int
    ) -> None:
        """Build the model over one MAE, frozen whole, its weights left as they are.

        Args:
            mae: The MAE, shared with every model that reads it frozen.
            dim: The token width the MAE hands its tokens at.
            heads: How many attention heads the probe runs.
            depth: How many cross-attention blocks the probe stacks.
            seed: What fixes the probe as drawn.
        """
        super().__init__()
        self.mae = mae.requires_grad_(False)
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
