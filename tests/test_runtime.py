"""Unit tests for runtime device detection, lazy PyTorch loading, and seeding."""

from __future__ import annotations

import os
import pickle
import random
import subprocess
import sys

import numpy as np
import pytest

from constellation_detection.runtime import (
    collect_env_snapshot,
    make_dataloader_worker_init,
    resolve_device,
    seed_everything,
)


def test_phase1_imports_do_not_eagerly_import_torch() -> None:
    code = (
        "import sys; "
        "import constellation_detection; "
        "import constellation_detection.detection.traditional; "
        "import constellation_detection.evaluation; "
        "print('torch_loaded=' + str('torch' in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "torch_loaded=False" in proc.stdout


def test_collect_env_snapshot_schema() -> None:
    snap = collect_env_snapshot()
    assert snap["python_version"]
    assert "packages" in snap
    assert snap["packages"]["numpy"] is not None
    assert "git" in snap


def test_seed_everything_and_picklable_worker_init() -> None:
    summary = seed_everything(seed=123, deterministic=True)
    assert summary["seed"] == 123
    a1 = random.random()
    b1 = float(np.random.rand())

    seed_everything(seed=123, deterministic=True)
    a2 = random.random()
    b2 = float(np.random.rand())
    assert a1 == a2
    assert b1 == b2
    assert os.environ["PYTHONHASHSEED"] == "123"

    worker_init = make_dataloader_worker_init(100)
    rehydrated = pickle.loads(pickle.dumps(worker_init))
    rehydrated(2)


def test_resolve_device_cpu_and_guardrails() -> None:
    torch = pytest.importorskip("torch")

    device, info = resolve_device(preferred="cpu", require_gpu=False)
    assert device == torch.device("cpu")
    assert info.device_type == "cpu"
    assert info.probe_passed is True
    assert info.amp_supported is False

    with pytest.raises(ValueError):
        resolve_device(preferred="cpu", require_gpu=True)

    with pytest.raises(ValueError):
        resolve_device(preferred="tpu")

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device(preferred="cuda")
        with pytest.raises(RuntimeError):
            resolve_device(preferred="auto", require_gpu=True)
        auto_dev, auto_info = resolve_device(preferred="auto", require_gpu=False)
        assert auto_dev.type == "cpu"
        assert auto_info.fallback_reason is not None
