"""Environment-agnostic path configuration for Constellation Detection.

Provides OS-agnostic path resolution for project roots, datasets, experiment
outputs, and durable run artifacts without hardcoding local Windows or Google
Colab paths.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_PROJECT_ROOT = "CONSTELLATION_PROJECT_ROOT"
ENV_DATA_DIR = "CONSTELLATION_DATA_DIR"
ENV_OUTPUT_DIR = "CONSTELLATION_OUTPUT_DIR"
ENV_ARTIFACTS_DIR = "CONSTELLATION_ARTIFACTS_DIR"
ENV_CONFIG_FILE = "CONSTELLATION_CONFIG_FILE"

__all__ = [
    "ENV_ARTIFACTS_DIR",
    "ENV_CONFIG_FILE",
    "ENV_DATA_DIR",
    "ENV_OUTPUT_DIR",
    "ENV_PROJECT_ROOT",
    "ProjectPaths",
    "get_default_project_root",
    "is_colab_runtime",
    "to_relative_posix",
]


def get_default_project_root() -> Path:
    """Resolve the default repository root relative to this module."""
    env_root = os.environ.get(ENV_PROJECT_ROOT)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def is_colab_runtime() -> bool:
    """Detect Google Colab without importing ``google.colab``."""
    return bool(
        os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU")
    )


def to_relative_posix(
    path: str | os.PathLike[str],
    base: str | os.PathLike[str],
) -> str:
    """Convert ``path`` to a normalized relative POSIX string under ``base``.

    Args:
        path: Target file or directory path.
        base: Anchor root directory (e.g., ``data_dir`` or ``runs_dir``).

    Returns:
        Forward-slash relative path string safe to store in manifests.

    Raises:
        ValueError: If ``path`` lies outside ``base``.
    """
    resolved_target = Path(path).expanduser().resolve()
    resolved_base = Path(base).expanduser().resolve()
    return resolved_target.relative_to(resolved_base).as_posix()


def _resolve_maybe_relative(
    raw_path: str | os.PathLike[str],
    anchor_root: Path,
) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = anchor_root / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Container for configurable project directory paths.

    Attributes:
        project_root: Root directory of the project repository.
        data_dir: Directory for datasets and reference catalogs.
        output_dir: Directory for working outputs (predictions, visualizations).
        artifacts_dir: Optional directory for durable run bundles and checkpoints.
            Defaults to ``output_dir`` when ``None``.
    """

    project_root: Path
    data_dir: Path
    output_dir: Path
    artifacts_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.artifacts_dir is None:
            object.__setattr__(self, "artifacts_dir", self.output_dir)

    @property
    def raw_data_dir(self) -> Path:
        """Directory containing raw input sky photographs."""
        return self.data_dir / "raw"

    @property
    def annotations_dir(self) -> Path:
        """Directory reserved for independently verified ground truth only."""
        return self.data_dir / "annotations"

    @property
    def splits_dir(self) -> Path:
        """Directory storing image/sky-field dataset split manifests."""
        return self.data_dir / "splits"

    @property
    def catalogs_dir(self) -> Path:
        """Directory storing astronomical reference star catalogues."""
        return self.data_dir / "catalogs"

    @property
    def predictions_dir(self) -> Path:
        """Directory storing unverified prediction CSV exports."""
        return self.output_dir / "predictions"

    @property
    def visualizations_dir(self) -> Path:
        """Directory storing diagnostic plots and rendered overlays."""
        return self.output_dir / "visualizations"

    @property
    def resolved_artifacts_dir(self) -> Path:
        """Return concrete artifacts base directory."""
        return (
            self.artifacts_dir if self.artifacts_dir is not None else self.output_dir
        )

    @property
    def runs_dir(self) -> Path:
        """Directory storing run-scoped checkpoints, metrics, and bundles."""
        return self.resolved_artifacts_dir / "runs"

    @classmethod
    def from_paths(
        cls,
        project_root: str | os.PathLike[str] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        artifacts_dir: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
    ) -> ProjectPaths:
        """Create a ``ProjectPaths`` instance with explicit precedence.

        Precedence order (highest to lowest):
        1. Explicit function arguments.
        2. Environment variables (``CONSTELLATION_*``).
        3. JSON configuration file (``config_path`` or ``CONSTELLATION_CONFIG_FILE``).
        4. Repository-relative defaults (``<project_root>/data``, ``<project_root>/outputs``).
        """
        cfg_data: dict[str, Any] = {}
        raw_cfg = (
            config_path
            if config_path is not None
            else os.environ.get(ENV_CONFIG_FILE)
        )
        if raw_cfg:
            cfg_file = Path(raw_cfg).expanduser().resolve()
            if not cfg_file.is_file():
                raise FileNotFoundError(
                    f"Path configuration file not found: {cfg_file}"
                )
            loaded = json.loads(cfg_file.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Path config JSON must be an object, got {type(loaded).__name__}."
                )
            cfg_data = loaded

        if project_root is not None:
            resolved_root = Path(project_root).expanduser().resolve()
        elif os.environ.get(ENV_PROJECT_ROOT):
            resolved_root = get_default_project_root()
        elif cfg_data.get("project_root"):
            resolved_root = Path(
                str(cfg_data["project_root"])
            ).expanduser().resolve()
        else:
            resolved_root = get_default_project_root()

        if data_dir is not None:
            resolved_data = Path(data_dir).expanduser().resolve()
        elif os.environ.get(ENV_DATA_DIR):
            resolved_data = Path(os.environ[ENV_DATA_DIR]).expanduser().resolve()
        elif cfg_data.get("data_dir"):
            resolved_data = _resolve_maybe_relative(
                str(cfg_data["data_dir"]), resolved_root
            )
        else:
            resolved_data = resolved_root / "data"

        if output_dir is not None:
            resolved_output = Path(output_dir).expanduser().resolve()
        elif os.environ.get(ENV_OUTPUT_DIR):
            resolved_output = Path(os.environ[ENV_OUTPUT_DIR]).expanduser().resolve()
        elif cfg_data.get("output_dir"):
            resolved_output = _resolve_maybe_relative(
                str(cfg_data["output_dir"]), resolved_root
            )
        else:
            resolved_output = resolved_root / "outputs"

        if artifacts_dir is not None:
            resolved_artifacts = Path(artifacts_dir).expanduser().resolve()
        elif os.environ.get(ENV_ARTIFACTS_DIR):
            resolved_artifacts = (
                Path(os.environ[ENV_ARTIFACTS_DIR]).expanduser().resolve()
            )
        elif cfg_data.get("artifacts_dir"):
            resolved_artifacts = _resolve_maybe_relative(
                str(cfg_data["artifacts_dir"]), resolved_root
            )
        else:
            resolved_artifacts = resolved_output

        return cls(
            project_root=resolved_root,
            data_dir=resolved_data,
            output_dir=resolved_output,
            artifacts_dir=resolved_artifacts,
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | os.PathLike[str] | None = None,
        project_root: str | os.PathLike[str] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        artifacts_dir: str | os.PathLike[str] | None = None,
    ) -> ProjectPaths:
        """Resolve ``ProjectPaths`` from a JSON config file and optional overrides."""
        return cls.from_paths(
            project_root=project_root,
            data_dir=data_dir,
            output_dir=output_dir,
            artifacts_dir=artifacts_dir,
            config_path=config_path,
        )

    def validate_storage(self, create_missing: bool = False) -> dict[str, Any]:
        """Inspect configured directories and verify write access without assuming Drive.

        Args:
            create_missing: If ``True``, creates output and runs directories before
                testing write access.

        Returns:
            Structured status dictionary summarizing existence and writability.
        """
        if create_missing:
            for d in (
                self.output_dir,
                self.predictions_dir,
                self.visualizations_dir,
                self.resolved_artifacts_dir,
                self.runs_dir,
            ):
                d.mkdir(parents=True, exist_ok=True)

        def _is_writable(directory: Path) -> bool:
            if not directory.is_dir():
                return False
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=directory,
                    prefix=".write_probe_",
                    delete=False,
                ) as fh:
                    tmp_path = Path(fh.name)
                return True
            except OSError:
                return False
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

        return {
            "is_colab": is_colab_runtime(),
            "project_root_exists": self.project_root.is_dir(),
            "data_dir_exists": self.data_dir.is_dir(),
            "annotations_dir_exists": self.annotations_dir.is_dir(),
            "output_dir_exists": self.output_dir.is_dir(),
            "output_dir_writable": _is_writable(self.output_dir),
            "runs_dir_exists": self.runs_dir.is_dir(),
            "runs_dir_writable": _is_writable(self.runs_dir),
        }
