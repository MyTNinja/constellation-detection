# Project Instructions and Engineering Rules (`AGENTS.md`)

This document defines the authoritative workspace rules for **Vision-Based Constellation Detection and Recognition Using Hybrid Computer Vision and Deep Learning** (`constellation-detection` v0.1.0). Specialized procedures and domain standards live in `.agents/rules/` and `.agents/skills/`.

## 1. Project Identity & Research Objectives

- **Identity**: A vision-based astronomical star-detection and constellation-recognition research repository operating exclusively on uploaded night-sky photographs.
- **Sensor Exclusion (`ADR-001`)**: Sensor-based identification using GPS, compass, gyroscope, accelerometer, or device orientation metadata is strictly prohibited.
- **Primary Objectives**:
  1. Detect and localize point-source stars in night-sky images using traditional computer vision baselines (Phase 1) and PyTorch deep learning (Phase 2).
  2. Recognize constellations from detected star geometry and astronomical reference catalogues using vision-only matching (Phase 3).
  3. Combine classical CV and deep learning in a hybrid architecture (Phase 4) and package a local CPU-capable Python application (Phase 5).

## 2. Current Project Phase

| Phase | Scope | Status |
| :--- | :--- | :--- |
| **Phase 0** | Research preparation and legacy evaluation | **COMPLETED** |
| **Phase 1** | Traditional CV star detection baseline (`src/constellation_detection/`) | **COMPLETED** |
| **Phase 2** | PyTorch deep learning star detection | **NOT STARTED** (gated on verified ground truth) |
| **Phase 3** | Vision-based constellation recognition | **PLANNED** |
| **Phase 4** | Hybrid CV + PyTorch architecture | **PLANNED** |
| **Phase 5** | Local Python CPU application | **PLANNED** |

### Completed Phase 1 Baseline (`src/constellation_detection/`)

- **Detectors** (`src/constellation_detection/detection/traditional/`):
  - `global_threshold` (`detect_stars_global_threshold`, Colab identifier `CV1_GlobalThreshold`)
  - `CV2_AdaptiveTopHat` (`detect_stars_adaptive_tophat`, `19x19` elliptical White Top-Hat; **no CLAHE** per `ADR-006`)
  - `CV3_MultiScaleDoG` (`detect_stars_multiscale_dog`, `skimage.feature.blob_dog`)
- **Shared Pipeline**: `resize_to_max_dim(max_dim=800, cv2.INTER_AREA)`, `fast_nms_stars(min_dist_px=10.0, max_candidates=250, pre-cap=500, cKDTree)`, and `compute_star_annulus_snr(top_k=30, edge_margin=11)`.

## 3. Authoritative Documentation & Source Priority

1. `docs/context/ARCHITECTURE.md` — Implemented vs. proposed system architecture, schemas, coordinate conventions, and hardware constraints.
2. `docs/context/DECISIONS.md` — Traceable decision register (`ADR-001` through `ADR-016`, historical `ADR-003-HIST`, open questions `UD-001` through `UD-008`).
3. `research/INDEX.md` — Verified sources vs. unreviewed papers, catalogues, and dataset leads.
4. `README.md`, `pyproject.toml`, and `requirements-colab.txt` — Package setup and runtime entrypoints.
5. **Source Conflict Resolution**: Implemented code takes precedence over historical briefs and notes. Document contradictions explicitly rather than silently rewriting history.

## 4. Technology Stack & Code Quality

- **Core Stack**: Python (`>=3.10`; local `.venv` runs Python `3.13`), NumPy, SciPy, OpenCV (`opencv-python-headless`), scikit-image, Pandas, and Matplotlib.
- **Deep Learning Framework (`ADR-003`)**: **PyTorch** exclusively (`torch>=2.0`, `torchvision>=0.15`). **Do not introduce TensorFlow, Keras, or JAX.**
- **Minimal Dependencies**: Do not add new third-party dependencies or create virtual environments without explicit user approval.
- **Modular Architecture**: Implement all reusable CV, DL, evaluation, and recognition code inside `src/constellation_detection/` using `from __future__ import annotations`, explicit type hints, and concise docstrings covering inputs, outputs, units, coordinate conventions, and assumptions.
- **Thin Notebooks**: Keep `notebooks/` strictly for experimentation, visualization, and orchestration calling `constellation_detection`, never for core algorithm definitions.

## 5. Hardware Constraints & Local vs. Colab Responsibilities

- **Local PC**: Windows 11 AMD64, 4 CPU cores, ~8 GiB RAM, **no NVIDIA GPU, no CUDA, and no local `torch` installed**. Owns Phase 1 CV execution, software engineering, `pytest` and `ruff` checks, CPU-compatible inference (`map_location="cpu"`), and the Phase 5 local Python application.
- **Google Colab GPU**: Owns Phase 2+ PyTorch training and GPU-heavy experiments via `requirements-colab.txt` (`-e .`, preserving Colab's native GPU PyTorch/CUDA build).
- **Environment Portability**: Core modules in `src/constellation_detection/` must never import `google.colab` or hardcode `/content`, `/content/drive`, `C:\`, or `D:\` paths. Accept explicit path parameters or resolve via `constellation_detection.config`.

## 6. Scientific Integrity, Baseline Protection & Coordinate Conventions

- **Preserve Phase 1 Baselines**: Do not alter, simplify, or replace existing algorithms, heuristics, default parameters, or mathematical formulations unless explicitly authorized by the user.
- **Ground-Truth Restrictions (`ADR-008`)**: Zero verified ground truth currently exists in this repository. Never fabricate synthetic star coordinates, bounding boxes, or constellation labels to pass tests or fill gaps. Keep predictions (`outputs/predictions/`, CSV columns `rank,x_px,y_px,flux_score`) strictly separated from future ground-truth annotations (`data/annotations/`). Never report precision, recall, F1, or astrometric error without independently verified ground truth.
- **Coordinate Conventions (`ADR-011`)**:
  - Image pixels follow OpenCV convention: origin `(0, 0)` at top-left, `+x` rightward (`[0, W-1]`), `+y` downward (`[0, H-1]`), NumPy indexing `image[iy, ix]`, sub-pixel centroids in `float32` (`stars_xyf`) or `float64` (CSV).
  - Phase 1 prediction CSVs are recorded in **working resolution (`max_dim=800`), NOT native image resolution**. Always record and convert coordinate frames explicitly.
  - Phase 3 celestial reference coordinates use J2000/ICRS `(RA, Dec)` in degrees.

## 7. Dataset Integrity & Reproducibility

- **Zero Data Leakage**: Partition splits strictly at the image / sky-field / observing-session level. Never compute statistics, normalizations, or thresholds across validation or test samples.
- **Reproducibility**: Fix explicit random seeds (`random`, NumPy, PyTorch, `DataLoader` workers) and log hyperparameters, software versions, dataset split manifests, and Git commit SHAs for every experiment.
- **Licensing & Git Hygiene**: Respect imagery copyrights (do not redistribute copyrighted samples such as Akira Fujii images). Keep raw datasets, credentials, model weights, and generated outputs out of Git commits.

## 8. Operating Modes, Specialized Rules & Reusable Skills

All work operates in one of three explicit modes governed by `.agents/rules/workflow.md`: **Research & Planning**, **Implementation & Experimentation**, or **Independent Technical & Scientific Review**.

| Customization | Type | Activation / Purpose |
| :--- | :--- | :--- |
| `.agents/rules/workflow.md` | Rule (`always_on`) | Operating modes, transition locks, approval gates |
| `.agents/rules/scientific-research.md` | Rule (`model_decision`) | Literature review, citations, uncertainty, scientific claims |
| `.agents/rules/computer-vision.md` | Rule (`model_decision`) | Phase 1 CV baseline protection, coordinates, SNR/localization |
| `.agents/rules/deep-learning.md` | Rule (`model_decision`) | PyTorch-only DL, CPU/Colab portability, training & seeding |
| `.agents/rules/dataset-integrity.md` | Rule (`model_decision`) | Annotation provenance, split leakage, valid augmentations |
| `.agents/rules/code-review.md` | Rule (`model_decision`) | Independent severity-ranked review & regression checks |
| `.agents/skills/literature-review/SKILL.md` | Skill | Cited research specifications & source verification |
| `.agents/skills/dataset-audit/SKILL.md` | Skill | Dataset/catalogue provenance, coordinates & leakage audit |
| `.agents/skills/experiment-design/SKILL.md` | Skill | Hypothesis-driven CV/PyTorch experiment specifications |
| `.agents/skills/experiment-reproduction/SKILL.md` | Skill | Baseline parity & PyTorch experiment reproduction checks |
| `.agents/skills/scientific-code-review/SKILL.md` | Skill | Read-only scientific, numerical & architecture diff review |

## 9. Approval Requirements & Git Workflow

### Explicit User Approval Gates

Explicit user approval is required before: implementing research proposals, resolving open decisions (`UD-001` through `UD-008`), modifying Phase 1 baseline algorithms/defaults, adding dependencies or creating environments, changing data/annotation schemas, starting Phase 2+ model training, or merging branches into `main`.

### Routine, Low-Risk Changes

- Work directly on `main` only after confirming a clean working tree and updating local `main` from `origin/main`.
- Run relevant local checks (`pytest` and `ruff` via the existing `.venv`). Never commit or push a failing change.
- Keep Phase 1 workflows local on the user's PC (no Colab, cloud execution, or GitHub Actions). After checks pass, commit and push routine changes directly to `main`.

### Larger or Higher-Risk Changes

- Use one dedicated branch for multi-step, cross-cutting, algorithmic, dependency, interface, data-format, or Phase 2+ work across prompts; do not create a new branch per prompt.
- Run local checks and inspect the diff before sharing a GitHub compare link (`https://github.com/MyTNinja/constellation-detection/compare/main...<branch>`). Wait for explicit user approval before updating local `main`, fast-forward merging, and pushing `main`.

### Boundaries

- Do not create pull requests. ChatGPT reviews and advises only when asked; Antigravity performs coding and Git operations.
- Never force-push, rewrite Git history, delete branches, change GitHub settings, modify branch protection, install/enable extensions, or add GitHub Actions or PR templates unless explicitly asked.
- Verify only intended files are staged before any commit.
