"""Run artifact management, safe CPU weight loading, and Colab-to-local transfer.

Follows ``docs/context/ARCHITECTURE.md`` §20 & §26.3:
- Never saves ambiguous checkpoint filenames.
- Separates lightweight CPU ``state_dict`` inference weights from resumable
  optimizer/scheduler training states.
- Deserializes inference weights strictly with ``map_location='cpu'`` and
  ``weights_only=True``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from constellation_detection.runtime.reproducibility import collect_env_snapshot

ARTIFACT_SCHEMA_VERSION = "1.0.0"

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "RunArtifactManager",
    "build_run_id",
    "compute_sha256",
    "load_weights_artifact",
    "verify_checksums",
    "verify_imported_bundle",
    "verify_parity_reference",
]

_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9-]+")
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")


def _sanitize_token(value: str, fallback: str = "x") -> str:
    cleaned = _SAFE_TOKEN_RE.sub("-", value.strip().lower()).strip("-")
    return cleaned or fallback


def _validate_run_id(run_id: str) -> str:
    if not _SAFE_RUN_ID_RE.match(run_id) or ".." in run_id:
        raise ValueError(
            f"Unsafe or malformed run_id {run_id!r}; expected alphanumerics, '-', '_', '.'."
        )
    return run_id


def build_run_id(
    phase: str,
    task: str,
    model_key: str,
    dataset_version: str,
    seed: int,
    git_sha: str | None = None,
    timestamp_utc: datetime | None = None,
) -> str:
    """Construct a deterministic, self-describing experiment run identifier.

    Format: ``<phase>_<task>_<model_key>_<dataset_version>_s<seed>_<YYYYMMDD-HHMMSSZ>_<git7>``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")

    ts = timestamp_utc if timestamp_utc is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    ts_str = ts.strftime("%Y%m%d-%H%M%SZ")

    if git_sha is None:
        git_short = (
            collect_env_snapshot().get("git", {}).get("commit_short") or "nogit"
        )
    else:
        git_short = _sanitize_token(git_sha, "nogit")[:7]

    parts = (
        _sanitize_token(phase, "p2"),
        _sanitize_token(task, "task"),
        _sanitize_token(model_key, "model"),
        _sanitize_token(dataset_version, "ds0"),
        f"s{seed}",
        ts_str,
        git_short,
    )
    return "_".join(parts)


def compute_sha256(file_path: str | os.PathLike[str]) -> str:
    """Compute the lowercase hexadecimal SHA-256 digest of a file."""
    src = Path(file_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"File not found for hashing: {src}")
    digest = hashlib.sha256()
    with src.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(dest: Path, writer: Callable[[Path], None]) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
        writer(tmp_path)
        os.replace(tmp_path, dest)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return dest


def load_weights_artifact(weights_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load an inference weights artifact safely onto CPU with ``weights_only=True``.

    Args:
        weights_path: Path to ``<run_id>__weights_<tag>.pt``.

    Returns:
        Dictionary containing ``schema_version``, ``model_name``, ``model_kwargs``,
        and CPU ``state_dict``.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load checkpoint weights."
        ) from exc

    src = Path(weights_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Weights artifact not found: {src}")

    payload = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"Invalid weights artifact {src}: expected dict containing 'state_dict'."
        )
    if "schema_version" not in payload or "model_name" not in payload:
        raise ValueError(
            f"Invalid weights artifact {src}: missing schema_version or model_name."
        )
    return payload


def verify_checksums(run_dir: str | os.PathLike[str]) -> tuple[bool, list[str]]:
    """Verify every entry in ``<run_dir>/checksums.sha256`` safely."""
    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "checksums.sha256"
    if not manifest_path.is_file():
        return False, [f"Missing checksums manifest: {manifest_path}"]

    issues: list[str] = []
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return False, ["checksums.sha256 is empty."]

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "  " not in line:
            issues.append(f"Malformed checksum line {line_no}: {line!r}")
            continue
        expected_hash, rel_posix = line.split("  ", 1)
        rel_path = Path(rel_posix)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            issues.append(f"Unsafe relative path in checksums line {line_no}: {rel_posix!r}")
            continue
        target = (root / rel_path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            issues.append(f"Path escapes run directory: {rel_posix!r}")
            continue
        if not target.is_file():
            issues.append(f"Missing artifact file: {rel_posix}")
            continue
        actual_hash = compute_sha256(target)
        if actual_hash != expected_hash.lower():
            issues.append(
                f"Checksum mismatch for {rel_posix}: expected {expected_hash}, got {actual_hash}"
            )
    return len(issues) == 0, issues


def _sample_tensor_values(flat_tensor: Any, max_samples: int = 512) -> tuple[list[int], list[float]]:
    n = int(flat_tensor.numel())
    if n == 0:
        return [], []
    if n <= max_samples:
        indices = list(range(n))
    else:
        step = (n - 1) / float(max_samples - 1)
        indices = sorted({round(i * step) for i in range(max_samples)})
    values = [float(flat_tensor[idx].item()) for idx in indices]
    return indices, values


def _flatten_model_output(output: Any, torch_mod: Any) -> tuple[list[int], Any]:
    """Flatten Tensor or dict[str, Tensor] output deterministically for parity checks."""
    if isinstance(output, torch_mod.Tensor):
        return list(output.shape), output.detach().cpu().to(torch_mod.float32).reshape(-1)
    if isinstance(output, dict) and output:
        parts = []
        shapes: list[int] = []
        for key in sorted(output.keys()):
            val = output[key]
            if not isinstance(val, torch_mod.Tensor):
                raise TypeError(f"Expected Tensor for output[{key!r}], got {type(val).__name__}.")
            if not shapes:
                shapes = list(val.shape)
            parts.append(val.detach().cpu().to(torch_mod.float32).reshape(-1))
        return shapes, torch_mod.cat(parts, dim=0)
    raise TypeError(
        "Parity reference expects model(input) to return a Tensor or dict of Tensors."
    )


def verify_parity_reference(
    model: Any,
    run_dir: str | os.PathLike[str],
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare local CPU model forward pass against recorded Colab parity reference."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for parity verification.") from exc

    ref_path = Path(run_dir).expanduser().resolve() / "metrics" / "parity_reference.json"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Parity reference not found: {ref_path}")

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    seed = int(ref["seed"])
    input_shape = tuple(int(x) for x in ref["input_shape"])
    expected_output_shape = list(ref["output_shape"])
    sample_indices = [int(i) for i in ref["sample_indices"]]
    expected_values = [float(v) for v in ref["sample_values"]]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    dummy_input = torch.randn(input_shape, generator=gen, dtype=torch.float32)

    cpu_model = model.to("cpu")
    cpu_model.eval()
    with torch.inference_mode():
        output = cpu_model(dummy_input)
        actual_shape, flat = _flatten_model_output(output, torch)
        actual_values = [float(flat[idx].item()) for idx in sample_indices]

    shape_match = actual_shape == expected_output_shape
    max_abs_diff = (
        max((abs(a - b) for a, b in zip(actual_values, expected_values)), default=0.0)
        if shape_match
        else float("inf")
    )
    passed = bool(shape_match and max_abs_diff <= float(atol))
    return {
        "passed": passed,
        "shape_match": shape_match,
        "expected_shape": expected_output_shape,
        "actual_shape": actual_shape,
        "max_abs_diff": max_abs_diff,
        "atol": float(atol),
    }


def verify_imported_bundle(
    run_dir: str | os.PathLike[str],
    model_builder: Callable[[str, dict[str, Any]], Any] | None = None,
    weights_tag: str = "best",
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Verify transferred run checksums, safe CPU weight loading, and optional parity."""
    root = Path(run_dir).expanduser().resolve()
    ok, issues = verify_checksums(root)

    run_id = root.name
    meta_path = root / "run_metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = str(meta.get("run_id", run_id))

    weights_path = root / "checkpoints" / f"{run_id}__weights_{weights_tag}.pt"
    weights_loaded = False
    model_name: str | None = None
    parity_report: dict[str, Any] | None = None

    if weights_path.is_file():
        payload = load_weights_artifact(weights_path)
        weights_loaded = True
        model_name = str(payload["model_name"])
        model_kwargs = dict(payload.get("model_kwargs", {}))
        if model_builder is not None:
            model = model_builder(model_name, model_kwargs)
            model.load_state_dict(payload["state_dict"], strict=True)
            ref_file = root / "metrics" / "parity_reference.json"
            if ref_file.is_file():
                parity_report = verify_parity_reference(model, root, atol=atol)

    all_ok = bool(
        ok
        and weights_loaded
        and (parity_report is None or parity_report["passed"])
    )
    return {
        "verified": all_ok,
        "checksums_valid": ok,
        "checksum_issues": issues,
        "weights_loaded": weights_loaded,
        "model_name": model_name,
        "parity": parity_report,
    }


class RunArtifactManager:
    """Manages structured run outputs, atomic checkpoints, and transfer archives."""

    def __init__(
        self,
        runs_dir: str | os.PathLike[str],
        run_id: str,
        create: bool = True,
    ) -> None:
        self.run_id = _validate_run_id(run_id)
        self.runs_dir = Path(runs_dir).expanduser().resolve()
        self.run_dir = self.runs_dir / self.run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        self.visualizations_dir = self.run_dir / "visualizations"

        if create:
            for d in (
                self.run_dir,
                self.checkpoints_dir,
                self.metrics_dir,
                self.logs_dir,
                self.visualizations_dir,
            ):
                d.mkdir(parents=True, exist_ok=True)

    def _resolve_rel(self, rel_path: str) -> Path:
        candidate = (self.run_dir / rel_path).resolve()
        candidate.relative_to(self.run_dir)
        return candidate

    def write_json_atomic(self, rel_path: str, payload: dict[str, Any]) -> Path:
        """Atomically write a formatted UTF-8 JSON file inside ``run_dir``."""
        dest = self._resolve_rel(rel_path)
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        return _atomic_write(dest, lambda p: p.write_bytes(encoded))

    def write_text_atomic(self, rel_path: str, content: str) -> Path:
        """Atomically write a UTF-8 text file inside ``run_dir``."""
        dest = self._resolve_rel(rel_path)
        encoded = content.encode("utf-8")
        return _atomic_write(dest, lambda p: p.write_bytes(encoded))

    def save_weights_artifact(
        self,
        model: Any,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        tag: str = "best",
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save CPU-compatible inference ``state_dict`` and primitive metadata atomically."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to save weights artifacts."
            ) from exc

        safe_tag = _sanitize_token(tag, "best")
        dest = self.checkpoints_dir / f"{self.run_id}__weights_{safe_tag}.pt"
        cpu_state = {
            k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in model.state_dict().items()
        }
        payload: dict[str, Any] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_name": str(model_name),
            "model_kwargs": dict(model_kwargs or {}),
            "extra_metadata": dict(extra_metadata or {}),
            "state_dict": cpu_state,
        }
        return _atomic_write(dest, lambda tmp: torch.save(payload, tmp))

    def save_training_state(
        self,
        state_bundle: dict[str, Any],
        tag: str = "last",
    ) -> Path:
        """Save full resumable training state (optimizer/scheduler/RNG) atomically."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to save training state checkpoints."
            ) from exc

        safe_tag = _sanitize_token(tag, "last")
        dest = self.checkpoints_dir / f"{self.run_id}__trainstate_{safe_tag}.pt"
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": self.run_id,
            **state_bundle,
        }
        return _atomic_write(dest, lambda tmp: torch.save(payload, tmp))

    def write_parity_reference(
        self,
        model: Any,
        input_shape: tuple[int, ...],
        seed: int = 42,
    ) -> Path:
        """Record a deterministic forward-pass reference digest for cross-host checks."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to write a parity reference."
            ) from exc

        params = list(model.parameters())
        original_device = params[0].device if params else torch.device("cpu")
        was_training = bool(model.training)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        dummy_cpu = torch.randn(input_shape, generator=gen, dtype=torch.float32)

        try:
            model.eval()
            with torch.inference_mode():
                output = model(dummy_cpu.to(original_device))
                out_shape, flat = _flatten_model_output(output, torch)
                indices, values = _sample_tensor_values(flat)
                payload = {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "seed": int(seed),
                    "input_shape": [int(x) for x in input_shape],
                    "dtype": "float32",
                    "source_device": str(original_device),
                    "output_shape": out_shape,
                    "sample_indices": indices,
                    "sample_values": values,
                }
        finally:
            if was_training:
                model.train()

        return self.write_json_atomic("metrics/parity_reference.json", payload)

    def _collect_artifact_files(self, include_trainstate: bool = True) -> list[Path]:
        files: list[Path] = []
        for p in sorted(self.run_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name == "checksums.sha256" or p.name.endswith(".tar.gz"):
                continue
            if not include_trainstate and "__trainstate_" in p.name:
                continue
            files.append(p)
        return files

    def write_checksums(self, include_trainstate: bool = True) -> Path:
        """Write ``checksums.sha256`` for artifacts currently in ``run_dir``."""
        lines = [
            f"{compute_sha256(p)}  {p.relative_to(self.run_dir).as_posix()}"
            for p in self._collect_artifact_files(
                include_trainstate=include_trainstate
            )
        ]
        return self.write_text_atomic(
            "checksums.sha256", "\n".join(lines) + ("\n" if lines else "")
        )

    def export_run_bundle(
        self,
        include_trainstate: bool = False,
        dest_dir: str | os.PathLike[str] | None = None,
    ) -> Path:
        """Package run artifacts into a self-verifying ``.tar.gz`` transfer bundle."""
        self.write_checksums(include_trainstate=True)

        target_parent = (
            Path(dest_dir).expanduser().resolve()
            if dest_dir is not None
            else self.runs_dir
        )
        suffix = "full_bundle" if include_trainstate else "inference_bundle"
        archive_path = target_parent / f"{self.run_id}__{suffix}.tar.gz"

        files_to_pack = self._collect_artifact_files(
            include_trainstate=include_trainstate
        )
        checksum_lines = [
            f"{compute_sha256(p)}  {p.relative_to(self.run_dir).as_posix()}"
            for p in files_to_pack
        ]
        checksum_bytes = (
            "\n".join(checksum_lines) + ("\n" if checksum_lines else "")
        ).encode("utf-8")

        def _write_tar(tmp_path: Path) -> None:
            with tarfile.open(tmp_path, mode="w:gz") as tf:
                for file_path in files_to_pack:
                    arcname = f"{self.run_id}/{file_path.relative_to(self.run_dir).as_posix()}"
                    tf.add(file_path, arcname=arcname)
                info = tarfile.TarInfo(name=f"{self.run_id}/checksums.sha256")
                info.size = len(checksum_bytes)
                tf.addfile(info, io.BytesIO(checksum_bytes))

        return _atomic_write(archive_path, _write_tar)
