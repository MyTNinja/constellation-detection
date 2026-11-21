"""Reproducibility seeding, DataLoader worker helpers, and environment snapshots."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from constellation_detection.config import get_default_project_root, is_colab_runtime

__all__ = [
    "WorkerSeedInitializer",
    "collect_env_snapshot",
    "make_dataloader_worker_init",
    "seed_everything",
]


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    return int(seed)


@dataclass(frozen=True)
class WorkerSeedInitializer:
    """Picklable DataLoader ``worker_init_fn`` compatible with Windows ``spawn``."""

    base_seed: int

    def __call__(self, worker_id: int) -> None:
        worker_seed = (self.base_seed + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        try:
            import torch

            torch.manual_seed(worker_seed)
        except ImportError:
            pass


def make_dataloader_worker_init(base_seed: int) -> WorkerSeedInitializer:
    """Build a deterministic, Windows-picklable PyTorch ``worker_init_fn``."""
    valid_seed = _validate_seed(base_seed)
    return WorkerSeedInitializer(base_seed=valid_seed)


def seed_everything(seed: int = 42, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch (when installed) for reproducible runs.

    Args:
        seed: Non-negative integer seed applied across RNG backends.
        deterministic: If ``True``, enables PyTorch deterministic flags when
            PyTorch is installed.

    Returns:
        Structured dictionary summarizing applied seeds and determinism flags.
    """
    valid_seed = _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(valid_seed)
    random.seed(valid_seed)
    np.random.seed(valid_seed)

    torch_seeded = False
    cuda_seeded = False
    deterministic_algorithms_applied = False

    try:
        import torch

        torch.manual_seed(valid_seed)
        torch_seeded = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(valid_seed)
            cuda_seeded = True
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
        try:
            torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
            deterministic_algorithms_applied = True
        except RuntimeError:
            deterministic_algorithms_applied = False
    except ImportError:
        pass

    return {
        "seed": valid_seed,
        "deterministic": bool(deterministic),
        "torch_seeded": torch_seeded,
        "cuda_seeded": cuda_seeded,
        "deterministic_algorithms_applied": deterministic_algorithms_applied,
        "cross_hardware_note": (
            "Exact float32 bit-identity is not guaranteed between CUDA GPUs and "
            "CPU x86 runtimes; verify transferred weights via parity tolerance."
        ),
    }


def _safe_pkg_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_git_state(project_root: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "commit_sha": None,
        "commit_short": None,
        "dirty": None,
    }
    try:
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha_proc.returncode == 0:
            full_sha = sha_proc.stdout.strip()
            state["commit_sha"] = full_sha
            state["commit_short"] = full_sha[:7]
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status_proc.returncode == 0:
            state["dirty"] = bool(status_proc.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return state


def collect_env_snapshot(project_root: Path | None = None) -> dict[str, Any]:
    """Capture a portable environment, Git, package, and hardware snapshot.

    Args:
        project_root: Optional repository path used for Git provenance inspection.

    Returns:
        JSON-serializable dictionary of environment metadata.
    """
    resolved_root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else get_default_project_root()
    )

    packages = {
        pkg: _safe_pkg_version(pkg)
        for pkg in (
            "constellation-detection",
            "numpy",
            "scipy",
            "opencv-python",
            "scikit-image",
            "pandas",
            "matplotlib",
            "torch",
            "torchvision",
            "pytest",
            "ruff",
            "ipykernel",
            "psutil",
        )
    }

    torch_runtime: dict[str, Any] = {"installed": False}
    try:
        import torch

        torch_runtime = {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            ),
        }
    except ImportError:
        pass

    ram_info: dict[str, float | None] = {
        "ram_total_gib": None,
        "ram_available_gib": None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_info = {
            "ram_total_gib": round(float(vm.total) / (1024.0**3), 2),
            "ram_available_gib": round(float(vm.available) / (1024.0**3), 2),
        }
    except ImportError:
        pass

    return {
        "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_colab": is_colab_runtime(),
        "os_platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable_name": Path(sys.executable).name,
        "cpu_logical_count": os.cpu_count(),
        "ram": ram_info,
        "git": _collect_git_state(resolved_root),
        "packages": packages,
        "torch_runtime": torch_runtime,
    }
