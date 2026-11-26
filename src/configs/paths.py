"""Where the repository keeps what a benchmark reads and writes."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = REPO_ROOT / ".env"

CONFIGS_ROOT = REPO_ROOT / "configs"

OUTPUT_ROOT = Path(os.environ.get("MAE_OUTPUT_ROOT", REPO_ROOT))

ARTIFACTS_ROOT = OUTPUT_ROOT / "artifacts"

BENCHMARKS_ROOT = OUTPUT_ROOT / "benchmarks"

HEADS_FILE = "heads.pt"

LOGGED_CONFIG_FILE = "logged_config.json"


def build_root(build: str, root: str) -> Path:
    """Return where one build of the dataset sits on this machine.

    Args:
        build: The build's name.
        root: The folder holding the builds, relative to the repository or absolute.

    Returns:
        root: The build's folder.
    """
    return REPO_ROOT / root / build


def heads_path(run_name: str) -> Path:
    """Return where a run's fine-tuned benchmark heads are written on this machine.

    Args:
        run_name: What the run is called.

    Returns:
        path: The heads' file.
    """
    return BENCHMARKS_ROOT / run_name / HEADS_FILE


def logged_config_path(run_name: str) -> Path:
    """Return where an offline job reads the config a pretrained run logged.

    Args:
        run_name: What the pretrained run is called.

    Returns:
        path: The config's file, written by the prepare step before the job.
    """
    return ARTIFACTS_ROOT / run_name / LOGGED_CONFIG_FILE
