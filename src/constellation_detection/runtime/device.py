"""Verified runtime device detection and active PyTorch compute probing.

Imports PyTorch lazily so Phase 1 traditional CV workflows remain usable in
environments without ``torch`` installed.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["DeviceInfo", "resolve_device"]


@dataclass(frozen=True)
class DeviceInfo:
    """Verified telemetry snapshot for the selected PyTorch execution device.

    Attributes:
        device_type: Normalized PyTorch device type (``"cpu"`` or ``"cuda"``).
        device_str: Exact device string (e.g., ``"cpu"`` or ``"cuda:0"``).
        cuda_available: Whether PyTorch reports CUDA support and at least one GPU.
        device_count: Number of visible CUDA devices.
        device_name: Human-readable GPU or CPU device description.
        vram_total_mib: Total GPU VRAM in MiB when CUDA is active, else ``None``.
        vram_free_mib: Free GPU VRAM in MiB when CUDA is active, else ``None``.
        torch_version: Installed ``torch.__version__`` string.
        cuda_version: CUDA toolkit version compiled into PyTorch, or ``None``.
        cudnn_version: cuDNN version integer if available, else ``None``.
        cpu_threads: Active PyTorch intra-op CPU thread count.
        amp_supported: Whether CUDA automatic mixed precision is enabled for use.
        probe_passed: Whether the active tiny-tensor compute probe succeeded.
        fallback_reason: Explanation if ``preferred="auto"`` fell back to CPU.
    """

    device_type: str
    device_str: str
    cuda_available: bool
    device_count: int
    device_name: str
    vram_total_mib: float | None
    vram_free_mib: float | None
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    cpu_threads: int
    amp_supported: bool
    probe_passed: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize device telemetry into a JSON-compatible dictionary."""
        return asdict(self)


def resolve_device(
    preferred: str = "auto",
    require_gpu: bool = False,
) -> tuple[Any, DeviceInfo]:
    """Select and actively verify a PyTorch execution device.

    Never assumes CUDA availability: always checks ``torch.cuda.is_available()``,
    ``torch.cuda.device_count()``, and runs a small tensor allocation + reduction
    probe before returning.

    Args:
        preferred: Requested target (``"auto"``, ``"cpu"``, ``"cuda"``, or
            ``"cuda:<index>"``). Defaults to ``"auto"``.
        require_gpu: If ``True``, raises ``RuntimeError`` when CUDA is unavailable.

    Returns:
        Tuple ``(torch.device, DeviceInfo)``.

    Raises:
        ImportError: If PyTorch is not installed in the active environment.
        ValueError: If ``preferred`` is invalid or conflicts with ``require_gpu``.
        RuntimeError: If a GPU is required/requested without CUDA, or the compute
            probe fails.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for resolve_device() but is not installed. "
            "Install CPU wheels (requirements-local-cpu.txt) or Colab GPU runtime."
        ) from exc

    norm_pref = str(preferred).strip().lower()
    valid_prefix = norm_pref in {"auto", "cpu", "cuda"} or (
        norm_pref.startswith("cuda:") and norm_pref.split(":", 1)[1].isdigit()
    )
    if not valid_prefix:
        raise ValueError(
            f"Invalid preferred device {preferred!r}; "
            "expected 'auto', 'cpu', 'cuda', or 'cuda:<index>'."
        )

    if norm_pref == "cpu" and require_gpu:
        raise ValueError("Cannot request preferred='cpu' together with require_gpu=True.")

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    has_cuda = cuda_available and device_count > 0
    fallback_reason: str | None = None

    if norm_pref == "cpu":
        device = torch.device("cpu")
    elif norm_pref == "cuda" or norm_pref.startswith("cuda:"):
        if not has_cuda:
            raise RuntimeError(
                f"CUDA device {preferred!r} was requested, "
                "but no CUDA GPU is available in this runtime."
            )
        target_str = "cuda:0" if norm_pref == "cuda" else norm_pref
        if target_str.startswith("cuda:"):
            idx = int(target_str.split(":", 1)[1])
            if idx >= device_count:
                raise RuntimeError(
                    f"Requested CUDA index {idx}, but only {device_count} "
                    "CUDA device(s) are visible."
                )
        device = torch.device(target_str)
    else:  # preferred == "auto"
        if has_cuda:
            device = torch.device("cuda:0")
        elif require_gpu:
            raise RuntimeError(
                "require_gpu=True, but no CUDA GPU is available. "
                "Verify runtime accelerator settings."
            )
        else:
            fallback_reason = "CUDA GPU unavailable; falling back to CPU execution."
            warnings.warn(fallback_reason, RuntimeWarning, stacklevel=2)
            device = torch.device("cpu")

    # Active compute probe on the chosen device (never assume allocation works)
    probe = torch.ones((4, 4), dtype=torch.float32, device=device)
    probe_sum = float((probe + probe).sum().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not math.isclose(probe_sum, 32.0, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(
            f"Compute probe failed on {device}: expected 32.0, got {probe_sum}."
        )

    if device.type == "cuda":
        dev_idx = 0 if device.index is None else int(device.index)
        device_name = str(torch.cuda.get_device_name(dev_idx))
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
        vram_free_mib: float | None = round(float(free_bytes) / (1024.0**2), 2)
        vram_total_mib: float | None = round(float(total_bytes) / (1024.0**2), 2)
        device_str = f"cuda:{dev_idx}"
    else:
        device_name = "CPU"
        vram_free_mib = None
        vram_total_mib = None
        device_str = "cpu"

    cudnn_ver: int | None = None
    if torch.backends.cudnn.is_available():
        cudnn_ver = int(torch.backends.cudnn.version())

    info = DeviceInfo(
        device_type=device.type,
        device_str=device_str,
        cuda_available=has_cuda,
        device_count=device_count,
        device_name=device_name,
        vram_total_mib=vram_total_mib,
        vram_free_mib=vram_free_mib,
        torch_version=str(torch.__version__),
        cuda_version=getattr(torch.version, "cuda", None),
        cudnn_version=cudnn_ver,
        cpu_threads=int(torch.get_num_threads()),
        amp_supported=(device.type == "cuda"),
        probe_passed=True,
        fallback_reason=fallback_reason,
    )
    return device, info
