"""One batch of budget crops, with SWIM's targets and the points each is scored on."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from mars_multisensor_mae.architecture.models.tokens import Tokens


class SwimBatch(NamedTuple):
    """One batch of budget crops, with SWIM's targets and the points each is scored on.

    Attributes:
        tokens: Each instrument's patches over every crop, keyed as ODE names it.
        tiles: The tile each crop belongs to, a tile's crops side by side.
        targets: SWIM's value of every point of each crop's tile, in TARGETS order.
            (C, P * P, Z)
        scored: Which points each crop is scored on. (C, P * P)
    """

    tokens: dict[str, Tokens]
    tiles: list[str]
    targets: np.ndarray
    scored: np.ndarray
