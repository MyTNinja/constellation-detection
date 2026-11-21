"""Unit tests for portable OS/Colab path configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from constellation_detection.config import (
    ENV_ARTIFACTS_DIR,
    ENV_CONFIG_FILE,
    ENV_DATA_DIR,
    ENV_OUTPUT_DIR,
    ENV_PROJECT_ROOT,
    ProjectPaths,
    is_colab_runtime,
    to_relative_posix,
)


def test_default_project_paths_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        ENV_PROJECT_ROOT,
        ENV_DATA_DIR,
        ENV_OUTPUT_DIR,
        ENV_ARTIFACTS_DIR,
        ENV_CONFIG_FILE,
    ):
        monkeypatch.delenv(var, raising=False)

    paths = ProjectPaths.from_paths()
    assert paths.project_root.is_dir()
    assert paths.data_dir == paths.project_root / "data"
    assert paths.output_dir == paths.project_root / "outputs"
    assert paths.resolved_artifacts_dir == paths.output_dir
    assert paths.raw_data_dir == paths.data_dir / "raw"
    assert paths.annotations_dir == paths.data_dir / "annotations"
    assert paths.predictions_dir == paths.output_dir / "predictions"
    assert paths.runs_dir == paths.output_dir / "runs"
    assert paths.annotations_dir != paths.predictions_dir


def test_precedence_args_over_env_over_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_file = tmp_path / "paths.json"
    cfg_file.write_text(
        json.dumps(
            {
                "data_dir": "cfg_data",
                "output_dir": "cfg_out",
                "artifacts_dir": "cfg_art",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(ENV_PROJECT_ROOT, str(tmp_path))
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "env_data"))
    monkeypatch.setenv(ENV_OUTPUT_DIR, str(tmp_path / "env_out"))
    monkeypatch.delenv(ENV_ARTIFACTS_DIR, raising=False)

    # Config supplies artifacts_dir; env overrides data/output; explicit arg wins last
    resolved = ProjectPaths.from_config(
        config_path=cfg_file,
        data_dir=tmp_path / "arg_data",
    )
    assert resolved.project_root == tmp_path.resolve()
    assert resolved.data_dir == (tmp_path / "arg_data").resolve()
    assert resolved.output_dir == (tmp_path / "env_out").resolve()
    assert resolved.resolved_artifacts_dir == (tmp_path / "cfg_art").resolve()


def test_relative_posix_and_storage_validation(tmp_path: Path) -> None:
    paths = ProjectPaths.from_paths(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
    )
    sample = paths.data_dir / "raw" / "field_01.jpg"
    assert to_relative_posix(sample, paths.data_dir) == "raw/field_01.jpg"

    with pytest.raises(ValueError):
        to_relative_posix(tmp_path / "elsewhere" / "x.jpg", paths.data_dir)

    before = paths.validate_storage(create_missing=False)
    assert before["output_dir_exists"] is False
    assert before["output_dir_writable"] is False

    after = paths.validate_storage(create_missing=True)
    assert after["output_dir_exists"] is True
    assert after["output_dir_writable"] is True
    assert after["runs_dir_exists"] is True
    assert after["runs_dir_writable"] is True


def test_colab_detection_env_without_google_colab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    assert is_colab_runtime() is False

    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-test")
    assert is_colab_runtime() is True
