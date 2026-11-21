"""Where the repository keeps what a benchmark reads and writes."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = REPO_ROOT / ".env"

CONFIGS_ROOT = REPO_ROOT / "configs"

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"

HEADS_FILE = "heads.pt"


def build_root(build: str, root: str) -> Path:
    """Return where one build of the dataset sits on this machine.

    Args:
        build: The build's name.
        root: The folder holding the builds, relative to the repository.

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
