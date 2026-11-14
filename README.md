# Vision-Based Constellation Detection and Recognition Using Hybrid Computer Vision and Deep Learning

## Research Objective

This project investigates and develops a hybrid pipeline for detecting stars and recognizing astronomical constellations from night-sky images. It combines:

1. **Classical Computer Vision Baseline**: Traditional image processing and morphological/blob-detection techniques for star localization and baseline feature extraction.
2. **Deep Learning Recognition**: PyTorch-based neural architectures for robust constellation pattern detection and classification under varying sky conditions, sensor noise, and field-of-view variations.

## Current Development Status

> **Status: Foundation Stage (v0.1.0)**
>
> The repository is currently initialized with its foundational package structure, build configuration, and portability rules.
> **Note:** The original Phase 1 star-detection notebook has **not** yet been migrated into the modular package or `notebooks/` directory.

## Technology Stack

- **Language**: Python >= 3.10
- **Computer Vision & Image Processing**: OpenCV (`opencv-python`), `scikit-image`, Pillow
- **Scientific Computing & Data**: NumPy, SciPy, Pandas, PyYAML
- **Deep Learning**: PyTorch (`torch`, `torchvision`) *(TensorFlow is intentionally excluded)*
- **Visualization**: Matplotlib
- **Testing & Quality**: `pytest`, `ruff`

## High-Level Architecture

```text
constellation-detection/
├── notebooks/                  # Lightweight notebooks for experiments & visualization
├── src/
│   └── constellation_detection/
│       ├── __init__.py         # Package metadata
│       └── config.py           # Environment-agnostic path configuration
├── tests/                      # Automated unit and integration tests
├── docs/                       # Research notes and project documentation
├── .gitignore                  # Git exclusion rules
├── AGENTS.md                   # Engineering & research rules for AI coding agents
├── README.md                   # Project overview and workflow guide
├── pyproject.toml              # Package build and dependency specification
└── requirements-colab.txt      # Lightweight installation file for Google Colab
```

### Design Principles

- **Core logic in `.py` modules**: All reusable algorithms reside under `src/constellation_detection/`.
- **Lightweight notebooks**: Notebooks in `notebooks/` orchestrate experiments and visualize results by importing the core package.
- **Environment neutrality**: Reusable code contains no `google.colab` imports and no hardcoded `/content`, `/content/drive`, or Windows file paths.

## Installation

### Local Development (Windows / Linux / macOS)

Create and activate a Python >= 3.10 virtual environment, then install the package in editable mode:

```bash
# Core computer vision dependencies only
pip install -e .

# Include development tools (pytest, ruff)
pip install -e ".[dev]"

# Include deep-learning dependencies (PyTorch, torchvision)
pip install -e ".[dl,dev]"
```

*(For specific CUDA or CPU-only builds of PyTorch on Windows, install `torch` and `torchvision` via the official PyTorch index URL prior to running `pip install -e .`.)*

### Google Colab Setup

In Google Colab, PyTorch with GPU support is preinstalled. To preserve Colab's native CUDA environment while installing this package and its core dependencies:

```bash
pip install -r requirements-colab.txt
```

## Development, Execution & Storage Workflow

1. **Local Repository as Source of Truth**:
   - All source code, module refactoring, testing, and Git commits happen in the local repository and are version-controlled via Git and GitHub.
   - Local Windows execution is used for code development, unit testing, classical CV baselines, and CPU-based model inference.

2. **Google Colab for GPU Execution**:
   - Heavy deep-learning training and large-scale evaluation runs are executed in Google Colab notebooks that clone/pull the repository and install the package in editable mode.

3. **Google Drive for Persistent Artifacts**:
   - Datasets, ground-truth catalogs, trained model checkpoints (`.pt`/`.pth`), and experiment outputs are stored in Google Drive (mounted in Colab or synced locally) and excluded from Git via `.gitignore`.
   - Directory locations are passed dynamically to `constellation_detection.config.ProjectPaths` (or configured via `CONSTELLATION_DATA_DIR` and `CONSTELLATION_OUTPUT_DIR` environment variables) so core modules remain completely agnostic to where files are hosted.
