"""Unit tests for reproducible run artifacts, safe CPU weight I/O, and bundles."""

from __future__ import annotations

import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from constellation_detection.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    RunArtifactManager,
    build_run_id,
    load_weights_artifact,
    verify_checksums,
    verify_imported_bundle,
)


def _tiny_conv_builder(model_name: str, model_kwargs: dict[str, int]) -> object:
    from torch import nn

    if model_name != "tiny_conv":
        raise ValueError(f"Unknown test model: {model_name}")
    channels = int(model_kwargs["channels"])
    return nn.Sequential(
        nn.Conv2d(1, channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, 1, kernel_size=1),
    )


def test_build_run_id_and_json_checksum_roundtrip(tmp_path: Path) -> None:
    run_id = build_run_id(
        phase="P2",
        task="Star Det",
        model_key="UNet_Mini",
        dataset_version="ds01",
        seed=42,
        git_sha="abcdef123456",
        timestamp_utc=datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert run_id == "p2_star-det_unet-mini_ds01_s42_20261121-120000Z_abcdef1"

    mgr = RunArtifactManager(runs_dir=tmp_path / "runs", run_id=run_id)
    mgr.write_json_atomic("run_metadata.json", {"run_id": run_id})
    mgr.write_checksums()

    ok, issues = verify_checksums(mgr.run_dir)
    assert ok is True
    assert issues == []

    # Tamper with metadata to confirm SHA-256 detection catches corruption
    (mgr.run_dir / "run_metadata.json").write_text('{"run_id": "tampered"}\n', encoding="utf-8")
    ok_after, issues_after = verify_checksums(mgr.run_dir)
    assert ok_after is False
    assert len(issues_after) == 1


def test_unsafe_run_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunArtifactManager(runs_dir=tmp_path, run_id="../escape")


def test_weights_parity_and_transfer_bundle_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    run_id = "p2_stardet_tinyconv_ds01_s42_20261121-120000Z_abcdef1"
    colab_runs = tmp_path / "colab_runs"
    mgr = RunArtifactManager(runs_dir=colab_runs, run_id=run_id)
    mgr.write_json_atomic("run_metadata.json", {"run_id": run_id})
    mgr.write_json_atomic("train_config.json", {"seed": 42, "model_name": "tiny_conv"})

    model = _tiny_conv_builder("tiny_conv", {"channels": 4})
    weights_path = mgr.save_weights_artifact(
        model,
        model_name="tiny_conv",
        model_kwargs={"channels": 4},
        tag="best",
    )
    assert weights_path.name == f"{run_id}__weights_best.pt"

    mgr.save_training_state({"epoch": 3, "optimizer": {}}, tag="last")
    mgr.write_parity_reference(model, input_shape=(1, 1, 16, 16), seed=42)

    loaded = load_weights_artifact(weights_path)
    assert loaded["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert all(
        t.device.type == "cpu"
        for t in loaded["state_dict"].values()
        if isinstance(t, torch.Tensor)
    )

    bundle_path = mgr.export_run_bundle(include_trainstate=False)
    assert bundle_path.name == f"{run_id}__inference_bundle.tar.gz"

    local_runs = tmp_path / "local_runs"
    local_runs.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, mode="r:gz") as tf:
        tf.extractall(local_runs, filter="data")

    extracted_run_dir = local_runs / run_id
    assert not any(extracted_run_dir.rglob("*__trainstate_*"))

    report = verify_imported_bundle(
        extracted_run_dir,
        model_builder=_tiny_conv_builder,
    )
    assert report["verified"] is True
    assert report["checksums_valid"] is True
    assert report["weights_loaded"] is True
    assert report["parity"] is not None
    assert report["parity"]["passed"] is True
