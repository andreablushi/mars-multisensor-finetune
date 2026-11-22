"""One model's predictions of SWIM's points, and one row of the scores they come to."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

TilePredictions = dict[str, np.ndarray]


class IceScore(NamedTuple):
    """One row of the scores table.

    Attributes:
        share: The share of the training tiles the model was trained on.
        protocol: The protocol, "attentive-probe" or "fine-tune".
        target: The target scored, as TARGETS names it.
        metric: The metric's name, "RMSE" or "R2".
        value: What it came to.
    """

    share: str
    protocol: str
    target: str
    metric: str
    value: float
