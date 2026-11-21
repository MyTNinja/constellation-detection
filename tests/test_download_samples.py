"""Unit tests for scripts/download_phase1_samples.py atomic download behavior."""

from __future__ import annotations

import importlib.util
import io
import urllib.error
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


def _load_downloader_module() -> ModuleType:
    """Dynamically load scripts/download_phase1_samples.py without modifying sys.path."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "download_phase1_samples.py"
    )
    spec = importlib.util.spec_from_file_location("download_phase1_samples", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _InterruptedResponse:
    """Response stream that writes partial bytes then fails mid-stream."""

    def __init__(self) -> None:
        self._chunk_sent = False

    def read(self, size: int = -1) -> bytes:
        if not self._chunk_sent:
            self._chunk_sent = True
            return b"partial-image-bytes"
        raise ConnectionResetError("Simulated interrupted download")

    def __enter__(self) -> object:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_successful_download_atomic_replacement_and_user_agent(tmp_path: Path) -> None:
    """Successful download atomically writes final file, preserves User-Agent, leaves no temp files."""
    mod = _load_downloader_module()
    captured_headers: list[str | None] = []

    def fake_urlopen(req: object, timeout: int = 30) -> io.BytesIO:
        assert timeout == 30
        captured_headers.append(req.get_header("User-agent"))  # type: ignore[attr-defined]
        return io.BytesIO(b"complete-jpeg-payload")

    with (
        patch.dict(mod.ESA_HUBBLE_SAMPLES, {"sample.jpg": "https://example.test/sample.jpg"}, clear=True),
        patch.object(mod.urllib.request, "urlopen", side_effect=fake_urlopen),
    ):
        paths = mod.download_phase1_samples(output_dir=tmp_path)

    assert len(paths) == 1
    dest = tmp_path / "sample.jpg"
    assert paths[0] == dest.resolve()
    assert dest.read_bytes() == b"complete-jpeg-payload"
    assert captured_headers == ["constellation-detection-research/0.1 (Python urllib)"]
    # No leftover temporary .part files
    assert list(tmp_path.glob("*.part")) == []


def test_interrupted_download_cleans_temp_file_and_allows_retry(tmp_path: Path) -> None:
    """Interrupted download removes temp file, leaves no partial dest, and retries cleanly."""
    mod = _load_downloader_module()
    dest = tmp_path / "sample.jpg"

    with (
        patch.dict(mod.ESA_HUBBLE_SAMPLES, {"sample.jpg": "https://example.test/sample.jpg"}, clear=True),
        patch.object(mod.urllib.request, "urlopen", return_value=_InterruptedResponse()),
        pytest.raises(ConnectionResetError, match="Simulated interrupted download"),
    ):
        mod.download_phase1_samples(output_dir=tmp_path)

    # Destination must not exist and temp file must be cleaned up
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []

    # Subsequent retry succeeds cleanly
    with (
        patch.dict(mod.ESA_HUBBLE_SAMPLES, {"sample.jpg": "https://example.test/sample.jpg"}, clear=True),
        patch.object(
            mod.urllib.request,
            "urlopen",
            return_value=io.BytesIO(b"recovered-full-payload"),
        ),
    ):
        paths = mod.download_phase1_samples(output_dir=tmp_path)

    assert paths == [dest.resolve()]
    assert dest.read_bytes() == b"recovered-full-payload"
    assert list(tmp_path.glob("*.part")) == []


def test_failed_overwrite_preserves_existing_valid_file(tmp_path: Path) -> None:
    """Failed download with overwrite=True preserves existing destination file intact."""
    mod = _load_downloader_module()
    dest = tmp_path / "sample.jpg"
    dest.write_bytes(b"original-valid-image")

    with (
        patch.dict(mod.ESA_HUBBLE_SAMPLES, {"sample.jpg": "https://example.test/sample.jpg"}, clear=True),
        patch.object(mod.urllib.request, "urlopen", return_value=_InterruptedResponse()),
        pytest.raises(ConnectionResetError),
    ):
        mod.download_phase1_samples(output_dir=tmp_path, overwrite=True)

    assert dest.read_bytes() == b"original-valid-image"
    assert list(tmp_path.glob("*.part")) == []


class _TruncatedResponse(io.BytesIO):
    """Response stream that ends cleanly with fewer bytes than Content-Length."""

    def __init__(self, payload: bytes, advertised_length: int) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(advertised_length)}


def test_truncated_content_length_raises_cleans_up_and_retries(tmp_path: Path) -> None:
    """Truncated HTTP stream (fewer bytes than Content-Length) fails, cleans temp, preserves dest, and retries."""
    mod = _load_downloader_module()
    dest = tmp_path / "sample.jpg"
    dest.write_bytes(b"existing-valid-image")

    truncated = _TruncatedResponse(b"short-body", advertised_length=1024)
    with (
        patch.dict(
            mod.ESA_HUBBLE_SAMPLES,
            {"sample.jpg": "https://example.test/sample.jpg"},
            clear=True,
        ),
        patch.object(mod.urllib.request, "urlopen", return_value=truncated),
        pytest.raises(urllib.error.ContentTooShortError, match="received 10 of 1024 bytes"),
    ):
        mod.download_phase1_samples(output_dir=tmp_path, overwrite=True)

    # Existing destination file must remain intact and no temp files left behind
    assert dest.read_bytes() == b"existing-valid-image"
    assert list(tmp_path.glob("*.part")) == []

    # Next attempt with matching Content-Length succeeds
    full_payload = b"complete-matching-image-payload"
    complete = _TruncatedResponse(full_payload, advertised_length=len(full_payload))
    with (
        patch.dict(
            mod.ESA_HUBBLE_SAMPLES,
            {"sample.jpg": "https://example.test/sample.jpg"},
            clear=True,
        ),
        patch.object(mod.urllib.request, "urlopen", return_value=complete),
    ):
        paths = mod.download_phase1_samples(output_dir=tmp_path, overwrite=True)

    assert paths == [dest.resolve()]
    assert dest.read_bytes() == full_payload
    assert list(tmp_path.glob("*.part")) == []
