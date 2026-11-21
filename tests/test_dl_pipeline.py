"""Unit tests for Phase 2 PyTorch dataset, StarCenterUNetMini, losses, and CPU training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from constellation_detection.artifacts import RunArtifactManager, verify_imported_bundle
from constellation_detection.datasets import (
    ANNOTATION_SCHEMA_VERSION,
    StarHeatmapDataset,
    build_group_split_manifest,
    render_star_targets,
    rotate_points_90_cw_ccw,
    save_annotation_record,
    save_split_manifest,
)


def test_rotate_points_matches_numpy_rot90() -> None:
    h, w = 24, 32
    px, py = 7.0, 5.0
    img = np.zeros((h, w), dtype=np.float32)
    img[int(py), int(px)] = 100.0

    for k in range(4):
        rot_img = np.rot90(img, k=k)
        rot_pts, nh, nw = rotate_points_90_cw_ccw(
            np.array([[px, py]]), height=h, width=w, k_ccw=k
        )
        assert rot_img.shape == (nh, nw)
        rx, ry = round(rot_pts[0, 0]), round(rot_pts[0, 1])
        assert rot_img[ry, rx] == 100.0


def test_render_star_targets_handles_blends_and_ignore() -> None:
    stars = [
        {"x_native_px": 10.3, "y_native_px": 15.8, "ignore": False},
        {"x_native_px": 12.1, "y_native_px": 16.2, "ignore": False},
        {"x_native_px": 30.0, "y_native_px": 30.0, "ignore": True},
    ]
    hm, off, pos, ign = render_star_targets(stars, height=40, width=40, sigma_px=1.5)
    assert hm.shape == (1, 40, 40)
    assert float(hm.max()) <= 1.0
    assert float(pos.sum()) == 2.0
    assert float(ign[0, 30, 30]) == 1.0
    assert off[0, 16, 10] == pytest.approx(0.3, abs=1e-5)
    assert off[1, 16, 10] == pytest.approx(-0.2, abs=1e-5)


def _create_tiny_verified_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    ann_dir = data_dir / "annotations"
    raw_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    entries = []
    rng = np.random.default_rng(99)
    yy, xx = np.mgrid[0:64, 0:64]
    for i in range(6):
        sx, sy = 20.2 + i * 2.0, 22.4 + i * 1.5
        img = 15.0 + rng.normal(0.0, 1.5, (64, 64))
        img = img + 220.0 * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 3.2)
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        img_path = raw_dir / f"s{i}.png"
        cv2.imwrite(str(img_path), u8)

        ann_path = ann_dir / f"s{i}.json"
        save_annotation_record(
            {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "image_id": f"s{i}",
                "image_relpath": f"raw/s{i}.png",
                "coordinate_frame": "native_opencv_top_left_px",
                "native_width": 64,
                "native_height": 64,
                "ground_truth_provenance": "synthetic_wcs_fixture",
                "wcs_metadata": {"solver": "synthetic"},
                "limiting_magnitude": 15.0,
                "ignore_magnitude_upper": 16.0,
                "domain": "synthetic_fixture",
                "stars": [
                    {
                        "x_native_px": sx,
                        "y_native_px": sy,
                        "is_saturated": False,
                        "is_blended": False,
                        "ignore": False,
                    }
                ],
            },
            ann_path,
        )
        entries.append(
            {
                "image_id": f"s{i}",
                "group_id": f"g{i}",
                "image_relpath": f"raw/s{i}.png",
                "annotation_relpath": f"annotations/s{i}.json",
                "image_sha256": hashlib.sha256(img_path.read_bytes()).hexdigest(),
                "annotation_sha256": hashlib.sha256(ann_path.read_bytes()).hexdigest(),
            }
        )

    manifest = build_group_split_manifest(entries, dataset_version="smoke-v1", seed=42)
    m_path = data_dir / "splits" / "manifest.json"
    save_split_manifest(manifest, m_path)
    return data_dir, m_path


def test_dl_model_loss_tiled_inference_and_smoke_training(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from constellation_detection.detection.dl import build_model, detect_stars_dl
    from constellation_detection.runtime import resolve_device
    from constellation_detection.training import (
        StarCenterLoss,
        TrainingConfig,
        run_training,
    )

    data_dir, m_path = _create_tiny_verified_fixture(tmp_path)
    tr_ds = StarHeatmapDataset(
        m_path, data_dir, split="train", crop_size=32, augment=True, seed=42
    )
    va_ds = StarHeatmapDataset(
        m_path, data_dir, split="val", crop_size=32, augment=False, seed=42
    )
    tr_loader = DataLoader(tr_ds, batch_size=2, shuffle=False, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=2, shuffle=False, num_workers=0)

    model = build_model("star_center_unet_mini", {"in_channels": 1, "base_channels": 16})
    n_params = sum(p.numel() for p in model.parameters())
    assert 100_000 < n_params < 600_000

    # Verify tiled CPU inference wrapper conforms to Phase 1 detector dict schema
    sample_raw = cv2.imread(str(data_dir / "raw" / "s0.png"), cv2.IMREAD_UNCHANGED)
    det_res = detect_stars_dl(sample_raw, model, device="cpu", tile_size=32, tile_overlap=8)
    assert det_res["detector"] == "StarCenterUNetMini"
    assert det_res["stars_xyf"].dtype == np.float32
    assert det_res["stars_xyf"].ndim == 2 and det_res["stars_xyf"].shape[1] == 3

    # Verify loss stays finite even with ignore masks active
    criterion = StarCenterLoss()
    sample_batch = next(iter(tr_loader))
    out = model(sample_batch["image"])
    loss, stats = criterion(out, sample_batch)
    assert torch.isfinite(loss) and stats["loss"] >= 0.0

    # 1-epoch CPU pipeline smoke test + artifact parity verification
    _, dev_info = resolve_device(preferred="cpu")
    cfg = TrainingConfig(epochs=1, batch_size=2, crop_size=32, warmup_epochs=0)
    mgr = RunArtifactManager(runs_dir=tmp_path / "runs", run_id="p2_smoke_test_s42")
    summary = run_training(
        model,
        tr_loader,
        va_loader,
        cfg,
        mgr,
        dev_info,
        parity_input_shape=(1, 1, 32, 32),
    )
    assert summary["epochs_completed"] == 1
    mgr.write_checksums()
    bundle_ok = verify_imported_bundle(
        mgr.run_dir,
        model_builder=lambda name, kw: build_model(name, kw),
    )
    assert bundle_ok["verified"] is True
