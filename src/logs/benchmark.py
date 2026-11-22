"""Logging what one benchmark comes to: its steps, its scores, its drawn tiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import wandb
from mars_multisensor_mae.logs.images.panels import DrawnImage
from mars_multisensor_mae.logs.tracker import RESULTS_STEP
from wandb.sdk.wandb_run import Run

from evaluation.models.metrics import IceScore

BENCHMARK = "benchmark"

FINETUNE = f"{BENCHMARK}-finetune"

HEADS = "heads"

STEP = f"{BENCHMARK}/step"


def log_step(run: Run, step: int, share: str, losses: Mapping[str, float]) -> None:
    """Log one training step, every protocol's loss against the step.

    Args:
        run: The tracked run.
        step: Which step it is, from 1.
        share: What the share of the tiles trained on is logged as.
        losses: The training loss, keyed by protocol.
    """
    run.log(
        {STEP: step}
        | {
            f"{BENCHMARK}/{protocol}/{share}/train/loss": one
            for protocol, one in losses.items()
        }
    )


def log_validation(run: Run, step: int, scores: Sequence[IceScore]) -> None:
    """Log one validation, every protocol's scores against the step.

    Args:
        run: The tracked run.
        step: The step validated at.
        scores: Each protocol's validation RMSE and R2 of every target.
    """
    run.define_metric(f"{BENCHMARK}/*", step_metric=STEP)
    run.log(
        {STEP: step}
        | {
            f"{BENCHMARK}/{protocol}/{share}/validation/{target}/{metric}": value
            for share, protocol, target, metric, value in scores
        }
    )


def log_benchmark(
    run: Run,
    scores: Sequence[IceScore],
    images: Sequence[DrawnImage],
) -> None:
    """Log the test scores and drawn tiles at one step for every run, or after it.

    Args:
        run: The tracked run.
        scores: Every model's test RMSE and R2 of each target.
        images: Each drawn tile's image and caption, logged as qualitative 1 on.
    """
    run.summary.update(
        {
            f"{BENCHMARK}/{protocol}/{share}/test/{target}/{metric}": value
            for share, protocol, target, metric, value in scores
        }
    )
    run.log(
        {
            f"{BENCHMARK}/scores": wandb.Table(
                columns=["share", "protocol", "target", "metric", "value"],
                data=[list(row) for row in scores],
            ),
            **{
                f"{BENCHMARK}/qualitative-{at}": wandb.Image(image, caption=caption)
                for at, (image, caption) in enumerate(images, 1)
            },
        },
        step=max(RESULTS_STEP, run.step),
    )
