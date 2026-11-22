"""The attentive probe, and the two models that read the MAE's tokens through it."""

from architecture.finetune_model import FinetunedModel
from architecture.probe_model import AttentiveProbeModel

BenchmarkModel = AttentiveProbeModel | FinetunedModel
