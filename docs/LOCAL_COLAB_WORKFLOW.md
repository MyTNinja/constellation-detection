# Local-to-Colab GPU Workflow

This guide defines the operational workflow bridging **Antigravity IDE + Windows CPU development** with **Google Colab GPU training** for Phase 2+ of `constellation-detection`.

> **Current Phase Note**: Phase 2 dataset acquisition (`UD-003`) and deep-learning model selection (`UD-004`) are still **pending user approval**. This document covers the Stage A infrastructure modules (`config`, `runtime`, `artifacts`) that keep code, paths, devices, and transferred checkpoints synchronized without starting model training yet.

---

## 1. Architecture Summary

- **Single Shared Codebase**: All models, datasets, transforms, losses, evaluation, and artifact helpers live in reusable `.py` modules under `src/constellation_detection/`. Notebooks only configure paths and orchestrate calls.
- **No `google.colab` in `src/`**: Core modules never import `google.colab`, never assume Google Drive is mounted, and never hardcode local Windows (`D:\...`) or Colab (`/content/...`) paths.
- **Lazy `torch` Isolation**: Top-level package imports and Phase 1 traditional CV modules do not import `torch` eagerly, so Phase 1 stays functional even without PyTorch installed.

---

## 2. Environment-Agnostic Path Management (`config.py`)

`ProjectPaths.from_paths(...)` and `ProjectPaths.from_config(...)` resolve directories using this strict precedence:
1. Explicit Python function arguments (`project_root`, `data_dir`, `output_dir`, `artifacts_dir`)
2. Environment variables (`CONSTELLATION_PROJECT_ROOT`, `CONSTELLATION_DATA_DIR`, `CONSTELLATION_OUTPUT_DIR`, `CONSTELLATION_ARTIFACTS_DIR`, `CONSTELLATION_CONFIG_FILE`)
3. Optional local JSON config file (copy [configs/paths.example.json](../configs/paths.example.json) to a gitignored local file)
4. Repository-relative defaults (`<project_root>/data` and `<project_root>/outputs`)

Derived properties enforce strict separation between inputs and outputs:
- Inputs (never overwritten by inference): `data_dir / raw`, `data_dir / annotations` (verified GT only), `data_dir / splits`, `data_dir / catalogs`
- Outputs (predictions only): `output_dir / predictions`, `output_dir / visualizations`, `artifacts_dir / runs`

Inside manifests, convert paths via `to_relative_posix(path, base)` so no personal absolute path strings leak into saved JSON files.

---

## 3. Colab GPU Execution Procedure

1. **Checkout Exact Code Revision**: Clone or pull the repository branch onto the Colab runtime disk at an explicit Git commit SHA.
2. **Install Package Without Overwriting Colab CUDA PyTorch**:
   ```bash
   pip install -r requirements-colab.txt
   ```
   (`requirements-colab.txt` runs `-e .`, preserving Colab's preinstalled GPU PyTorch/CUDA stack.)
3. **Configure Storage Paths Explicitly**: Point `CONSTELLATION_DATA_DIR` / `CONSTELLATION_ARTIFACTS_DIR` (or pass explicit args to `ProjectPaths.from_paths(...)`) toward runtime storage or user-mounted Drive if desired, then verify via `paths.validate_storage(create_missing=True)`.
4. **Verify Actual GPU Allocation Before Training**:
   ```python
   from constellation_detection.runtime import resolve_device, seed_everything

   device, device_info = resolve_device(preferred="auto", require_gpu=True)
   seed_info = seed_everything(seed=42, deterministic=True)
   ```
   `resolve_device` validates `torch.cuda.is_available()`, checks `device_count() > 0`, and runs a tiny tensor compute probe before training starts.

---

## 4. Artifact Packaging & Colab-to-Local Transfer

Every run creates a self-describing directory under `<artifacts_dir>/runs/<run_id>/`:
```text
<phase>_<task>_<model_key>_<dataset_ver>_s<seed>_<YYYYMMDD-HHMMSSZ>_<git7>
```
Using `RunArtifactManager`:
1. Save inference-only weights separately (`save_weights_artifact` → `<run_id>__weights_best.pt`) with all tensors detached to CPU and primitive metadata only.
2. Save resumable optimizer/scheduler state separately (`save_training_state` → `<run_id>__trainstate_last.pt`).
3. Record a deterministic forward-pass check (`write_parity_reference` → `metrics/parity_reference.json`).
4. Seal & package an inference bundle (`export_run_bundle(include_trainstate=False)` → `<run_id>__inference_bundle.tar.gz` containing matching `checksums.sha256`).
5. Download or sync `<run_id>__inference_bundle.tar.gz` manually into local untracked `outputs/runs/` (never commit `.pt` weights or datasets to Git).

---

## 5. Local Windows CPU Verification

Extract the transferred archive under `outputs/runs/` and validate checksums, `weights_only=True` CPU deserialization (`map_location="cpu"`), and deterministic forward-pass parity:

```python
from constellation_detection.artifacts import verify_imported_bundle

report = verify_imported_bundle(
    run_dir="outputs/runs/<run_id>",
    model_builder=my_model_builder,
    atol=1e-4,
)
assert report["verified"] is True
```

Note: Minor `float32` numerical differences (`~1e-5–1e-4`) between Colab CUDA kernels and local x86 CPU execution are normal; `verify_parity_reference` checks shape identity and `max_abs_diff <= atol` on the exact seeded synthetic probe input.
