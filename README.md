# SWIM Ice Benchmark of the Multi-Sensor Masked Autoencoder

The SWIM ice benchmark of a model trained by
[mars-multisensor-mae](https://github.com/andreablushi/mars-multisensor-mae): an
attentive probe over its tokens, frozen and fine-tuned, on the labels of
[mars-multisensor-benchmark](https://github.com/andreablushi/mars-multisensor-benchmark).

## Setup

The MAE, the benchmark labels and the dataset install editable from clones beside
this one, so local code always runs them as checked out there.

```bash
git clone https://github.com/andreablushi/mars-multisensor-mae ../mars-multisensor-mae
git clone https://github.com/andreablushi/mars-multisensor-benchmark ../mars-multisensor-benchmark
git clone https://github.com/andreablushi/mars-multisensor-dataset ../mars-multisensor-dataset
uv sync
source .venv/bin/activate
git config core.hooksPath .githooks            # once per clone, ruff runs before a push
uv run ruff check . && uv run ruff format .    # lint, over the whole repo
```

## DigitalHub setup

```bash
uv sync --group digitalhub
dhcli register <your-digitalhub-core-endpoint>
dhcli login                                    # opens a browser tab
cp .env.example .env                           # once, then fill it in
```

The platform project needs these secrets: `DHCORE_PERSONAL_ACCESS_TOKEN`,
`WANDB_ENTITY`, `WANDB_PROJECT` and `WANDB_API_KEY`, the same as the MAE's.

## Benchmark

```bash
uv run python scripts/benchmark.py default/mae-small
uv run --group digitalhub python scripts/benchmark.py --dh --ref main default/mae-small
uv run --group digitalhub python scripts/benchmark.py --dh --mae-ref latent-lite hf-latent/latent-acq-gap-base
```

- It trains an attentive probe on SWIM's subsurface ice consistency, 1 to 5 m and
  below 5 m, and on its radar surface power, which comes from SHARAD alone, over
  the run's latest W&B model, frozen and fine-tuned, both in one loop on the same
  batches and from the same draw, once on each share of the training tiles
  `benchmark.fractions` names (1%, 5%, 10% and 50%), a smaller share held
  in every larger one. One W&B run of job type `benchmark` holds the training
  curves, the heads and the test scores.
- `--finetune` fine-tunes alone, `--evaluate` scores the last logged heads alone, and
  neither runs both.
- The probe regresses SWIM's three targets at every grid point of a tile, each tile
  cut to the dataset's budget around one centre in training, up to
  `benchmark.finetune.budget_scale` times it (1, the size the test reads), with only the
  points near a read patch scored. Validation and test read each tile as the budget
  crops that cover its points. Fine-tuning trains the encoders and the cross-sensor
  encoder whole, so on `random` it is training from scratch. The loss divides each
  target by its spread over the training tiles.
- Each share trains for `benchmark.finetune.passes` over its tiles, never fewer
  than `min_steps` steps, and each protocol has its own optimizer, schedule
  and gradient scaler.
- RMSE and R2 measure the fit over every SWIM cell of the test tiles. Each protocol
  is kept at its best validation R2, averaged over the targets, measured
  `benchmark.validation.per_share` times a share on
  `benchmark.validation.sample_size` validation tiles.
- The drawn tiles show SWIM's targets beside the head's, a row per target, each
  coloured in viridis over the range its panels span.
- Fewer `benchmark.instruments` than the model reads name the run for those left
  out, such as `default/mae-small-no-sharad`.

- The model is rebuilt from the values its W&B training run logged (its
  evaluation run for `random`), so overrides only change `benchmark.*` and the
  `digitalhub.*` keys of this repository's `configs/`.
- `--mae-ref` names the MAE branch or tag a job installs, for models of an
  experiment branch; a local run reads the MAE checkout beside this one.

## Requirements for DigitalHub

A job installs `requirements.txt` when it starts, which takes the MAE, the labels
and the dataset from their GitHub main. Export it again whenever `uv.lock` changes,
then commit and push it:

```bash
uv export --no-hashes --no-dev --group digitalhub --no-emit-project --no-sources -o requirements.txt
uv lock                                        # the export relocks without sources
```
