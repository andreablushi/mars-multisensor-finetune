"""Preparing one benchmark on Leonardo's login node: what its offline job reads."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv
from mars_multisensor_mae.architecture.load import trained_run
from mars_multisensor_mae.logs.tracker import downloaded_artifact

from configs.load import logged_pretrained_config
from configs.paths import ARTIFACTS_ROOT, ENV_PATH, logged_config_path


def prepare_benchmark(run_name: str) -> None:
    """Download the config the benchmarked run logged and the model it trained.

    Args:
        run_name: What the run is called, a last part of "random" reading no model.
    """
    path = logged_config_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(logged_pretrained_config(run_name)))
    if trained_run(run_name):
        downloaded_artifact(run_name, ARTIFACTS_ROOT, "model")


if __name__ == "__main__":
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("run_name", help="what the benchmarked run is called")
    load_dotenv(ENV_PATH)
    prepare_benchmark(parsed.parse_args().run_name)
