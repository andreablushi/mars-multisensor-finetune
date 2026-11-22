"""What a benchmark config may hold: the pretrained run's values, and its own."""

from __future__ import annotations

from dataclasses import dataclass

from mars_multisensor_mae.configs.schema import DigitalHubConfig, ModelConfig


@dataclass
class DatasetConfig:
    """What the pretrained run read of the published dataset, and how it cut it.

    Attributes:
        version: The dataset version every build is read from.
        build: The build the run trained on, whose stats normalise another version.
        stats_version: The version whose training build sets the normalising
            constants, the version itself for runs that logged none.
        root: Where every build sits on this machine, under a directory of its own name.
        clip: The low and high each instrument's values are clipped to, None for open.
        sharad_atmosphere_cropping: Whether SHARAD is cropped below its atmosphere.
        acquisition: The acquisition angles of each instrument, as the index names them.
        patchsize: A patch's extent per axis and instrument, an omitted axis whole.
        budget: How many patches of each instrument a crop keeps.
        seed: The number every named draw derives its own seed from.
    """

    version: str
    build: str
    root: str
    clip: dict[str, list[float | None]]
    sharad_atmosphere_cropping: bool
    acquisition: dict[str, list[str]]
    patchsize: dict[str, dict[str, int]]
    budget: dict[str, int]
    seed: int
    stats_version: str = "${dataset.version}"


@dataclass
class TrainingConfig:
    """Which acquisition angles the pretrained run's model reads.

    Attributes:
        acquisition: Whether the decoders read the acquisition angles.
        acquisition_in_encoders: Whether the encoders read them too, true for runs
            that logged none.
    """

    acquisition: bool
    acquisition_in_encoders: bool = True


@dataclass
class ProbeConfig:
    """The attentive probe read over the model's tokens.

    Attributes:
        depth: How many cross-attention blocks the attentive probe stacks.
    """

    depth: int


@dataclass
class FinetuneConfig:
    """How the attentive probe is trained, frozen over the model and fine-tuned.

    Attributes:
        batch_size: How many tiles one training batch holds, each cut to the budget.
        budget_scale: The most a training batch's budget is multiplied by.
        passes: How many passes over a share of the tiles the head trains for, its
            steps never fewer than min_steps.
        min_steps: The fewest steps a share of the tiles trains for.
        warmup_share: The share of a schedule's steps the rate climbs for before
            the cosine decay.
        head_learning_rate: The head's peak learning rate.
        encoder_learning_rate: The encoders' peak learning rate when fine-tuned.
        weight_decay: The AdamW weight decay.
        max_grad_norm: The norm every step's gradient is clipped to.
    """

    batch_size: int
    budget_scale: float
    passes: int
    min_steps: int
    warmup_share: float
    head_learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    max_grad_norm: float


@dataclass
class ValidationConfig:
    """How each share's head is picked on the validation tiles.

    Attributes:
        sample_size: How many tiles of the k-fold validation fold every validation
            reads, the first of one draw.
        per_share: How many times a share of the tiles is validated, evenly apart,
            kept at its best R2.
    """

    sample_size: int
    per_share: int


@dataclass
class BenchmarkEvaluationConfig:
    """How the benchmark head is measured on the validation and test tiles.

    Attributes:
        batch_size: How many tiles one validation or test batch holds, each read
            as the budget crops covering SWIM's points.
        tiles: The test tiles drawn as SWIM's grid beside the predicted ones, in
            qualitative order.
        shown: The instrument whose every observation of a tile is drawn beneath.
    """

    batch_size: int
    tiles: list[str]
    shown: str


@dataclass
class BenchmarkConfig:
    """What a trained model is probed and fine-tuned on: SWIM's ice under each tile.

    Attributes:
        build: The build whose labelled tiles are read, published as
            <version>-dataset-<build>.
        instruments: The instruments the head reads.
        folds: How many equal folds the labelled tiles are dealt into, the first
            testing, the second validating and the rest training.
        fractions: The shares of the training tiles each protocol is trained on.
        probe: The head's architecture.
        finetune: How the head is trained.
        validation: How each share's head is picked.
        evaluation: How the head is measured.
    """

    build: str
    instruments: list[str]
    folds: int
    fractions: list[float]
    probe: ProbeConfig
    finetune: FinetuneConfig
    validation: ValidationConfig
    evaluation: BenchmarkEvaluationConfig


@dataclass
class Config:
    """One benchmark: the pretrained run's values it reads, and how it probes them.

    Attributes:
        dataset: What the pretrained run read.
        model: What the pretrained run trained.
        training: Which acquisition angles the pretrained model reads.
        benchmark: How the model is probed and fine-tuned on SWIM's ice.
        digitalhub: Where and how the benchmark runs on DigitalHub.
    """

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    benchmark: BenchmarkConfig
    digitalhub: DigitalHubConfig
