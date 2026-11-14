"""Minimal path configuration for the Constellation Detection project.

Provides OS-agnostic and environment-agnostic path resolution for project roots,
datasets, and experiment outputs without hardcoding local or Google Colab paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PROJECT_ROOT = "CONSTELLATION_PROJECT_ROOT"
ENV_DATA_DIR = "CONSTELLATION_DATA_DIR"
ENV_OUTPUT_DIR = "CONSTELLATION_OUTPUT_DIR"


def get_default_project_root() -> Path:
    """Resolve the default repository root relative to this module."""
    env_root = os.environ.get(ENV_PROJECT_ROOT)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Container for configurable project directory paths.

    Attributes:
        project_root: Root directory of the project repository.
        data_dir: Directory for datasets and reference catalogs.
        output_dir: Directory for experiment outputs, logs, and checkpoints.
    """

    project_root: Path
    data_dir: Path
    output_dir: Path

    @classmethod
    def from_paths(
        cls,
        project_root: str | os.PathLike[str] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
    ) -> ProjectPaths:
        """Create a ProjectPaths configuration from explicit paths or environment defaults.

        Args:
            project_root: Optional override for the project root directory.
            data_dir: Optional override for the dataset directory. Falls back to
                the ``CONSTELLATION_DATA_DIR`` environment variable or ``<project_root>/data``.
            output_dir: Optional override for the output directory. Falls back to
                the ``CONSTELLATION_OUTPUT_DIR`` environment variable or ``<project_root>/outputs``.

        Returns:
            Resolved ``ProjectPaths`` instance.
        """
        resolved_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else get_default_project_root()
        )

        raw_data = data_dir if data_dir is not None else os.environ.get(ENV_DATA_DIR)
        resolved_data = (
            Path(raw_data).expanduser().resolve()
            if raw_data is not None
            else resolved_root / "data"
        )

        raw_output = (
            output_dir if output_dir is not None else os.environ.get(ENV_OUTPUT_DIR)
        )
        resolved_output = (
            Path(raw_output).expanduser().resolve()
            if raw_output is not None
            else resolved_root / "outputs"
        )

        return cls(
            project_root=resolved_root,
            data_dir=resolved_data,
            output_dir=resolved_output,
        )
