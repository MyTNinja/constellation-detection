# Local Python CPU Environment (`venv/`)

This document specifies the reproducible local Windows CPU environment for **Vision-Based Constellation Detection and Recognition Using Hybrid Computer Vision and Deep Learning** (`constellation-detection` v0.1.0).

---

## 1. Hardware & OS Baseline

| Component | Verified Specification (`2026-11-21`) | Operational Consequence |
| :--- | :--- | :--- |
| **OS / Architecture** | Windows 11 (`10.0.26200`), `AMD64` | Keep Windows-specific paths out of `src/` (`AGENTS.md`) |
| **CPU** | Intel Core i3-8130U (`2` physical cores / `4` logical threads, up to `2.20 GHz`) | Suitable for Phase 1 CV, tests, and CPU inference (`torch.get_num_threads() == 2`) |
| **GPU / Accelerator** | Intel UHD Graphics 620 integrated (**no NVIDIA GPU / no CUDA**) | Install PyTorch **CPU-only** wheels locally; run GPU training in Google Colab |
| **System RAM** | `7.91 GiB` (`8.49 GB`) total (`~2.15 GiB` free headroom observed idle) | Use `torch.inference_mode()`, `batch_size=1`, and bounded/tiled inference locally |
| **Disk Footprint** | `venv/` occupies **`1217.8 MiB` (`1.189 GiB`)** alongside legacy Phase 1 `.venv/` (`595.7 MiB`) | `venv/` is gitignored and resides on `D:\` (`61.56 GiB` free) |

---

## 2. Python & Pinned Package Versions

- **Python Version**: `3.13.17` (`C:\Users\shourya\AppData\Local\Programs\Python\Python313\python.exe`)
- **Editable Package**: `constellation-detection 0.1.0`
- **Verified Key Packages** (full lock in [requirements-local-cpu.lock.txt](../requirements-local-cpu.lock.txt)):
  - **PyTorch CPU**: `torch 2.15.0+cpu`, `torchvision 0.29.2+cpu` (`cuda_available == False`)
  - **Traditional CV & Scientific**: `numpy 2.5.4`, `opencv-python 5.1.0.95`, `scipy 1.18.1`, `scikit-image 0.26.0`, `pillow 13.0.0`
  - **Data & Visualization**: `pandas 3.1.1`, `matplotlib 3.11.2`
  - **Testing & Linting**: `pytest 9.2.0`, `ruff 0.17.6`
  - **Notebook & Profiling**: `ipykernel 7.4.0`, `ipython 9.18.1`, `jupyter_client 8.10.0`, `jupyter_core 5.9.1`, `psutil 7.2.2`
- **Preserved Legacy Phase 1 Environment**: Existing `.venv/` (`Python 3.13.17`, no `torch`) remains untouched for Phase 1 isolation checks.

---

## 3. Clean Installation & Reinstallation Procedure

> **Rule:** Never assume CUDA locally and never modify system Python (`Python313` or `Python315`).

### Create Fresh CPU Environment
From the repository root (`d:\projects\constellation-detection`):

```powershell
# 1. Create virtual environment explicitly with Python 3.13 (avoid Python 3.15 on PATH)
C:\Users\shourya\AppData\Local\Programs\Python\Python313\python.exe -m venv venv

# 2. Upgrade pip and install CPU-only PyTorch wheels first
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install project dependencies, dev tools, and ipykernel
venv\Scripts\python.exe -m pip install -e ".[dev]" ipykernel
```

To recreate exact pinned package versions from the lockfile after removing `venv/`:
```powershell
Remove-Item -Recurse -Force venv
C:\Users\shourya\AppData\Local\Programs\Python\Python313\python.exe -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements-local-cpu.lock.txt
```

See also [requirements-local-cpu.txt](../requirements-local-cpu.txt) for constrained local dependencies and [requirements-colab.txt](../requirements-colab.txt) for Colab GPU sessions (`-e .` without overriding Colab's CUDA PyTorch).

---

## 4. Environment Activation & Jupyter Kernel Selection

### Terminal Activation
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# If PowerShell blocks activation scripts in your current session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
In `cmd.exe`:
```cmd
venv\Scripts\activate.bat
```

### Jupyter Kernel Registration & Selection
Registered user kernelspec:
```powershell
venv\Scripts\python.exe -m ipykernel install --user --name constellation-py313-cpu --display-name "Python 3.13 (constellation-cpu)"
```
- **In Antigravity / VS Code Notebooks**: Click the kernel selector (top-right of any `.ipynb`) and select **`Python 3.13 (constellation-cpu)`** (or `.venv` when testing pure Phase 1 without `torch`).
- **Verify active kernel location**: `venv\Scripts\python.exe -m jupyter kernelspec list`.

---

## 5. Measured CPU Verification & Performance Baseline (`2026-11-21`)

All checks ran inside `venv/` (`torch 2.15.0+cpu`, `cuda_available = False`, `1` warmup + timed trials):

| Benchmark Check | Conditions | Measured Median Time | Process RSS RAM |
| :--- | :--- | :---: | :---: |
| **Python Startup** | `venv\Scripts\python.exe -c "pass"` (3 trials) | `191.13 ms` | — |
| **Post-Import Baseline** | CV + SciPy + Matplotlib + Torch + Jupyter imports | — | `353.27 MiB` |
| **OpenCV + Phase 1 Detector** | `detect_stars_adaptive_tophat` on synthetic `640×800` `uint8` (`39` stars, 5 trials) | `42.69 ms` | `361.99 MiB` |
| **PyTorch CPU Matmul** | `512×512` `float32` matrix multiply (5 trials) | `4.75 ms` | `369.00 MiB` |
| **Small 3-Layer CNN Forward** | `5,121` params, `(1, 3, 256, 256)` under `torch.inference_mode()` (5 trials) | `23.64 ms` | `371.91 MiB` |

---

## 6. Troubleshooting & Known Limitations

1. **`Python 3.15.2` on `PATH`**: System `PATH` lists both `Python313` and `Python315`. Always invoke `C:\Users\shourya\AppData\Local\Programs\Python\Python313\python.exe` (or `venv\Scripts\python.exe` directly) when recreating environments, because `Python 3.15` lacks stable PyTorch/SciPy wheels.
2. **PyPI vs. PyTorch CPU Index**: Do **not** run bare `pip install -r` with `--index-url https://download.pytorch.org/whl/cpu` globally alongside PyPI packages (`opencv-python`, `pytest`, etc.); use `--extra-index-url` (as in [requirements-local-cpu.txt](../requirements-local-cpu.txt)) or install `torch`/`torchvision` in Step 2 first.
3. **PowerShell `Activate.ps1` Blocked**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or call `venv\Scripts\python.exe` directly without activating.
4. **Local vs. Colab Python Minor Version**: Local uses Python `3.13.17` while Colab often runs `3.10–3.12`. Write `>=3.10`-compatible code (`requires-python = ">=3.10"`, Ruff `target-version = "py310"`), save weights via CPU `state_dict`, and load checkpoints strictly with `torch.load(..., map_location="cpu", weights_only=True)`.
5. **Known Limitations**: No CUDA GPU locally; `jupyterlab`/`nbformat` web server packages are omitted by default (IDE notebook UI handles `.ipynb` via `ipykernel`); tight `7.91 GiB` RAM ceiling requires small models and tiled CPU inference for multi-megapixel images.
