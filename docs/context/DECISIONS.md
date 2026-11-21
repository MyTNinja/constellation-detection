# Architecture Decision Register (`DECISIONS.md`)

> **Last Updated:** 2026-11-21  
> **Repository Baseline:** `main` @ `71bc41b`  
> **Governance Policy:** Every entry below is traced directly to existing repository files, project briefs (`project-brief.md`, `project-brief_v2.md`), legacy notebooks, or verified Task 01 findings. **No decisions have been fabricated.** Unresolved design choices are isolated in Section 5 and await explicit user approval.

---

## 1. Register Rules & Source Authority

- **Authoritative Hierarchy:** Actual repository code → `project-brief_v2.md` → `project-brief.md` (historical) → `README.md`, `AGENTS.md`, and original notebooks → Task 01 audit → newly approved user decisions.
- **Disagreement Policy:** Where `project-brief_v2.md`, legacy notebooks, and migrated Python code diverge, both the documented brief statement and actual implemented state are recorded explicitly.
- **Status Definitions:**
  - **`Implemented`** — Enforced and verified in tracked repository code/tests/configuration.
  - **`Accepted (Planned)`** — Explicitly agreed in `project-brief_v2.md` and/or `AGENTS.md`, awaiting later phases (`2–5`).
  - **`Superseded`** — Explicitly replaced by a newer decision (`v2` / `AGENTS.md`).
  - **`Rejected`** — Explicitly ruled out by project constraints or Phase 1 design comments.
  - **`Unresolved / Pending Approval`** — Open question surfaced by Task 01 requiring explicit user direction before implementation.

---

## 2. Master Decision Index

| ID | Title | Status | Primary Source(s) |
|---|---|---|---|
| [ADR-001](#adr-001-vision-only-identification-and-sensor-exclusion) | Vision-Only Identification; Exclude Device Sensors | Implemented & Accepted | `project-brief_v2.md` §1, §3; `README.md` |
| [ADR-002](#adr-002-astronomical-catalogues-as-geometric-reference-only) | Astronomical Catalogues Allowed as Geometric Reference Only | Accepted (Planned — Phase 3) | `project-brief_v2.md` §3, §11.5 |
| [ADR-003-HIST](#adr-003-hist-historical-tensorflowkeras-vs-pytorch-framework-comparison) | Compare TensorFlow/Keras and PyTorch via Controlled Prototype | **Superseded** | `project-brief.md` §3, §6, §7, §10 |
| [ADR-003](#adr-003-pytorch-as-sole-deep-learning-framework) | PyTorch Selected as Sole Deep Learning Framework | **Accepted** (Supersedes `ADR-003-HIST`) | `project-brief_v2.md` §3, §10, §11.11; `AGENTS.md` §3; `README.md` |
| [ADR-004](#adr-004-separate-star-detection-from-constellation-recognition) | Separate Star Detection from Constellation Recognition | Implemented (Detection) / Accepted (Recognition) | `project-brief_v2.md` §4, §11.4; `README.md` |
| [ADR-005](#adr-005-preserve-phase-1-traditional-cv-detectors-as-baseline) | Preserve Phase 1 Traditional CV Detectors as Baseline | **Implemented** | `AGENTS.md` §1; `README.md`; `project-brief_v2.md` §7, §8, §11.1 |
| [ADR-006](#adr-006-reject-clahe-in-phase-1-top-hat-preprocessing) | Reject CLAHE in Phase 1 Top-Hat Preprocessing *(Brief Discrepancy)* | **Implemented / Rejected Approach** | `01_Project_Init...ipynb` L163; `tophat.py`; `dog.py` *(vs. `v2` L169)* |
| [ADR-007](#adr-007-reject-fixed-pixel-distance-thresholds-as-sole-recognizer) | Reject Fixed Pixel-Distance Thresholds as Sole Recognizer | Accepted (Planned — Phase 3) / Rejected Approach | `project-brief_v2.md` §11.6; `constellation-extraction.ipynb` |
| [ADR-008](#adr-008-strict-separation-of-predictions-and-verified-ground-truth) | Strict Separation of Unverified Predictions and Ground Truth | **Implemented & Enforced** | `AGENTS.md` §1; `project-brief_v2.md` §7 L196; `export.py`; `compare.py` |
| [ADR-009](#adr-009-modular-environment-independent-package-and-thin-notebooks) | Modular Environment-Independent Package & Lightweight Notebooks | **Implemented** | `AGENTS.md` §2; `config.py`; `phase1_traditional_cv_demo.ipynb` |
| [ADR-010](#adr-010-hybrid-colab-gpu-training-and-local-cpu-execution) | Hybrid Colab GPU Training and Local CPU Execution / App Deployment | Accepted (Local Phase 1 Implemented; Colab Phase 2 Planned) | `project-brief_v2.md` §3, §9; `AGENTS.md` §2; `requirements-colab.txt` |
| [ADR-011](#adr-011-progressive-single-to-multi-and-partial-constellation-scope) | Progressive Single- to Multi- and Partial-Constellation Scope | Accepted (Planned — Phases 3–4) | `project-brief_v2.md` §3, §7 |
| [ADR-012](#adr-012-support-astrophotography-and-smartphone-imagery) | Support Astrophotography and Smartphone Photographs (`RQ5`) | Accepted (Partially Realized — Astrophotography Only So Far) | `project-brief_v2.md` §3, §5 (`RQ5`) |
| [ADR-013](#adr-013-treat-combined-deep-learning-notebook-as-conceptual-reference-only) | Treat `combined-deep-learning-notebook.ipynb` as Conceptual Reference Only | Accepted / Rejected Implementation Source | `project-brief_v2.md` §8; `AGENTS.md` §1–3; Task 01 Audit F |
| [ADR-014](#adr-014-local-only-phase-1-workflow-without-ci-or-pull-requests) | Local-Only Phase 1 Development Without GitHub Actions or PRs | **Implemented** | `AGENTS.md` §4; Git commit `71808a0` |
| [ADR-015](#adr-015-phase-1-working-resolution-nms-and-gt-free-diagnostics) | Standardize `max_dim=800`, Shared `cKDTree` NMS (`10 px`), and `top30_snr` | **Implemented** | `compare.py`; `nms.py`; `diagnostics.py`; `01_Project_Init...ipynb` |
| [ADR-016](#adr-016-phase-1-colab-to-package-migration-adjustments) | Phase 1 Colab-to-`src/` Bug Fixes and Refactoring Adjustments | **Implemented** *(Partially Divergent)* | `threshold.py`; `tophat.py`; `dog.py`; `compare.py`; `export.py`; `overlays.py` |

---

## 3. Detailed Decision Records

### ADR-001: Vision-Only Identification and Sensor Exclusion
- **Status:** Implemented & Accepted (Enforced across all phases)
- **Sources:** `project-brief_v2.md` §1 & §3 (`Sensor-based identification: Excluded`); [README.md](../../README.md#L3).
- **Decision:** Constellations must be detected and recognized solely from visual evidence inside uploaded photographs. GPS, compass/magnetometer, gyroscope, accelerometer, and device-orientation/clock ephemeris lookups are prohibited as identification shortcuts.
- **Rationale:** Ensures core scientific contribution relies on computer vision, pattern matching, and astronomical geometry rather than hardware pose estimation.
- **Consequences:** All Phases 2–5 pipelines must operate purely on RGB/grayscale image inputs plus static reference star catalogues.

### ADR-002: Astronomical Catalogues as Geometric Reference Only
- **Status:** Accepted (Planned — Phase 3)
- **Sources:** `project-brief_v2.md` §3 (`Astronomical catalogue: Allowed as reference information`) & §11.5.
- **Decision:** Standard celestial star catalogues and IAU constellation definitions are permitted as static reference templates/priors for geometric star-pattern matching.
- **Rationale:** Celestial geometry (angular separations, bright-star triangles/quads, magnitude hierarchy, canonical stick figures) is objective astronomical ground knowledge independent of device sensors.
- **Consequences:** Phase 3 must ingest an authoritative local star catalogue (e.g., Hipparcos `I/239` or Yale BSC5 `V/50`) without network sensor queries during inference.

### ADR-003-HIST: Historical TensorFlow/Keras vs. PyTorch Framework Comparison
- **Status:** **Superseded** by [ADR-003](#adr-003-pytorch-as-sole-deep-learning-framework)
- **Sources:** `project-brief.md` §3 (`Deep learning framework: Compare TensorFlow/Keras and PyTorch`), §6, §7 (Phase 2), §10.
- **Historical Decision:** Version 1 of the project brief planned a controlled prototype comparing TensorFlow/Keras against PyTorch before committing to a deep learning framework, largely because earlier university coursework (`combined-deep-learning-notebook.ipynb`) used TensorFlow/Keras.
- **Superseded Reason:** Explicitly dropped in `project-brief_v2.md` to eliminate dual-framework overhead and focus directly on PyTorch.

### ADR-003: PyTorch as Sole Deep Learning Framework
- **Status:** **Accepted** (Supersedes `ADR-003-HIST`; ready for Phase 2)
- **Sources:** `project-brief_v2.md` §3, §7, §10, §11.11, §12; [AGENTS.md](../../AGENTS.md#L27); [README.md](../../README.md#L88); [pyproject.toml](../../pyproject.toml#L25-L28).
- **Decision:** **PyTorch** is the sole deep learning framework for all new neural network code (`torch>=2.0`, `torchvision>=0.15`). TensorFlow, Keras, and JAX are banned from implementation dependencies, and comparative framework experiments are not required.
- **Rationale:** Avoids duplicated training/evaluation engineering, unifies Colab GPU prototyping and local CPU deployment under one ecosystem, and satisfies `AGENTS.md` §3.
- **Consequences:** All Phase 2 datasets, losses, models, checkpoints, and tests must be native PyTorch.

### ADR-004: Separate Star Detection from Constellation Recognition
- **Status:** Implemented (Phase 1 detection) / Accepted (Phases 2–4 recognition)
- **Sources:** `project-brief_v2.md` §4 & §11.4; [README.md](../../README.md#L9), [#L89](../../README.md#L89).
- **Decision:** Decompose the architecture into discrete star coordinate detection (`Phase 1–2`) followed by geometric/graph/learned pattern recognition (`Phase 3–4`), rather than treating raw whole-image classification as sufficient.
- **Rationale:** Enables rigorous component-level benchmarking (`Precision`/`Recall`/`F1`/`Localization Error` for stars; constellation recognition metrics separately) and allows hybrid mixing of CV bright-star detection with DL localization.
- **Consequences:** Phase 1 detectors expose standardized `(x, y, score)` coordinates (`stars_xyf`), and diagnostic line overlays in [overlays.py](../../src/constellation_detection/visualization/overlays.py) must never be described as constellation recognition.

### ADR-005: Preserve Phase 1 Traditional CV Detectors as Baseline
- **Status:** **Implemented**
- **Sources:** [AGENTS.md](../../AGENTS.md#L7); [README.md](../../README.md#L9), [#L84](../../README.md#L84); `project-brief_v2.md` §7, §8, §11.1, §12.
- **Decision:** Keep all three Phase 1 traditional CV algorithms ([threshold.py](../../src/constellation_detection/detection/traditional/threshold.py), [tophat.py](../../src/constellation_detection/detection/traditional/tophat.py), [dog.py](../../src/constellation_detection/detection/traditional/dog.py)) and shared [nms.py](../../src/constellation_detection/detection/nms.py) intact as the permanent experimental baseline, while retaining `constellation-extraction.ipynb` strictly as historical reference.
- **Rationale:** Provides a reproducible traditional baseline required by `RQ1`, `RQ2`, and `RQ4` without biasing future comparisons.
- **Consequences:** Any bug fixes or algorithmic improvements to saturated-star handling or NMS cannot silently overwrite the Phase 1 baseline formulas without explicit user authorization (see [UD-005](#ud-005-phase-1-baseline-preservation-vs-saturated-star-and-nms-fixes)).

### ADR-006: Reject CLAHE in Phase 1 Top-Hat Preprocessing
- **Status:** **Implemented / Rejected Approach** *(Active Brief Discrepancy Documented)*
- **Sources:** `01_Project_Initialization_and_CV_Detectors.ipynb` Cell 2 line 163 (`# Normalize Top-Hat response to full [0, 255] dynamic range without CLAHE noise amplification`); [tophat.py](../../src/constellation_detection/detection/traditional/tophat.py#L142-L143); [dog.py](../../src/constellation_detection/detection/traditional/dog.py#L141-L142).
- **Brief Discrepancy:** Both `project-brief.md` §4 and `project-brief_v2.md` §7 (line 169) mention `"CLAHE enhancement"` as included in Phase 1. **Actual Phase 1 code does NOT implement CLAHE** and intentionally replaced it with `[0, 255]` min-max normalization of the White Top-Hat response.
- **Rationale:** Applying CLAHE amplified background sky/sensor noise in dark sky regions, worsening spurious blob detections.
- **Consequences:** Any future evaluation of CLAHE must be introduced as an explicit ablation experiment rather than assumed present in the Phase 1 baseline.

### ADR-007: Reject Fixed Pixel-Distance Thresholds as Sole Recognizer
- **Status:** Accepted (Planned — Phase 3) / Rejected Approach
- **Sources:** `project-brief_v2.md` §11.6 (`Avoid arbitrary pixel-distance thresholds as the sole recognition mechanism`); `constellation-extraction.ipynb` Cells 9 & 12 (`max_distance = 225`, `140`).
- **Decision:** Reject connecting stars via arbitrary hardcoded pixel thresholds as a constellation recognizer.
- **Rationale:** Pixel separation scales linearly with focal length and image resolution (e.g., `1.00×` vs. `6.08×` downscale across Phase 1 images), cannot distinguish constellations, and cannot handle partial/crowded sky views.
- **Consequences:** The proximity edge plot inside `plot_detector_diagnostics()` (`connect_rad_ratio=0.22`) is strictly a visual aid, not a recognition module. Phase 3 must use scale- and rotation-invariant geometric/graph descriptors matched against astronomical catalogues.

### ADR-008: Strict Separation of Predictions and Verified Ground Truth
- **Status:** **Implemented & Enforced**
- **Sources:** [AGENTS.md](../../AGENTS.md#L8-L10); `project-brief_v2.md` §7 line 196; [README.md](../../README.md#L21), [#L67](../../README.md#L67), [#L85-L86](../../README.md#L85-L86); [export.py](../../src/constellation_detection/evaluation/export.py); [compare.py](../../src/constellation_detection/evaluation/compare.py).
- **Decision:** Unverified detector CSV predictions (`outputs/predictions/*_predictions.csv`) and synthetic demo starfields (`SYNTHETIC_SEED=42`) must remain strictly separated from independently verified ground-truth annotations. Precision, recall, F1, and localization error must never be fabricated or reported without verified annotations.
- **Rationale:** Using detector predictions to train or evaluate supervised models creates circular bias and invalid scientific claims.
- **Consequences:** Supervised Phase 2 training cannot start until a verified, independently grounded annotation workflow is approved (`UD-003`).

### ADR-009: Modular Environment-Independent Package and Thin Notebooks
- **Status:** **Implemented**
- **Sources:** [AGENTS.md](../../AGENTS.md#L13-L20); `project-brief_v2.md` §11.10; [config.py](../../src/constellation_detection/config.py); [phase1_traditional_cv_demo.ipynb](../../notebooks/phase1_traditional_cv_demo.ipynb).
- **Decision:** Implement all reusable CV/DL/evaluation/visualization logic as modular, type-annotated Python under `src/constellation_detection/`, keep notebooks lightweight consumers, prohibit `google.colab` imports in `src/`, and resolve all paths via parameters or `ProjectPaths`.
- **Rationale:** Eliminates Colab notebook lock-in (e.g., `/content/constellation_vision` in the original Phase 1 notebook) and enables automated `pytest` verification.
- **Consequences:** All Phase 2–5 modules must live in `src/constellation_detection/`, covered by unit tests and decoupled from notebook UI state.

### ADR-010: Hybrid Colab GPU Training and Local CPU Execution
- **Status:** Accepted (Local Phase 1 Implemented; Colab Phase 2 Planned)
- **Sources:** `project-brief_v2.md` §3, §9, §12; [AGENTS.md](../../AGENTS.md#L19); [requirements-colab.txt](../../requirements-colab.txt); [README.md](../../README.md#L76-L80).
- **Decision:** Perform neural network training and GPU-heavy experiments in Google Colab (`requirements-colab.txt` preserves Colab's native PyTorch/CUDA build), while keeping software development, automated testing, CV preprocessing, model inference (`map_location="cpu"`), and the final Phase 5 application runnable locally on an `8 GB RAM` / 4-core CPU machine without CUDA.
- **Rationale:** Local workstation has no NVIDIA GPU/CUDA (`7.91 GiB` RAM, 97% utilized in audit).
- **Consequences:** Phase 2 PyTorch architectures must be compact enough for local CPU execution and free of GPU-only dependencies at inference time.

### ADR-011: Progressive Single to Multi and Partial Constellation Scope
- **Status:** Accepted (Planned — Phases 3–4)
- **Sources:** `project-brief_v2.md` §3 (`Initial implementation: Single-constellation pipeline`; `Long-term recognition: Multiple and partially visible constellations`) & §7.
- **Decision:** Build the initial Phase 3 recognizer as a single-constellation pipeline targeting a defined subset of constellations, then generalize in Phase 4 to multi-constellation wide fields and partial visibility.
- **Audit Observation:** Task 01 verified that 2 of the 5 current Phase 1 images (`cygnus_heic0720c.jpg` and `ursa_minor_heic0401g.jpg`) already depict multiple constellations, so Phase 3 benchmarks must account for multi-constellation distractors (`UD-007`).

### ADR-012: Support Astrophotography and Smartphone Imagery
- **Status:** Accepted (Astrophotography samples present; smartphone imagery missing)
- **Sources:** `project-brief_v2.md` §3 & §5 (`RQ5`).
- **Decision:** Target both controlled astrophotography and ordinary handheld smartphone night-sky photographs.
- **Consequences:** Phase 2–4 datasets cannot rely solely on ESA/Hubble outreach JPEGs; smartphone domain samples with realistic high-ISO noise, compression artifacts, light pollution, and foreground clutter must be collected and partitioned strictly by capture/field.

### ADR-013: Treat Combined Deep Learning Notebook as Conceptual Reference Only
- **Status:** Accepted / Rejected Implementation Source
- **Sources:** `project-brief_v2.md` §8; `combined-deep-learning-notebook.ipynb`; [AGENTS.md](../../AGENTS.md#L7-L27).
- **Decision:** Use the 9 university labs consolidated in `combined-deep-learning-notebook.ipynb` strictly for conceptual familiarity; prohibit copying its code into `src/constellation_detection/`.
- **Rationale:** Audited findings confirm 100% TensorFlow/Keras usage (conflicts with `ADR-003`), Colab `colab2pdf()` Linux shell calls, heavy `VGG16` (`138M` params / `528 MB`), code bugs (`compute_higher_order_derivative`, unused `lr`, overwritten `x_val`), and multiple data-leakage anti-patterns (pre-split `StandardScaler`/`MinMaxScaler`, overlapping LSTM train/test windows).

### ADR-014: Local-Only Phase 1 Workflow Without CI or Pull Requests
- **Status:** **Implemented**
- **Sources:** [AGENTS.md](../../AGENTS.md#L36-L52); Git commit `71808a0`.
- **Decision:** Execute Phase 1 engineering, `pytest`, and `ruff` checks locally on the user's PC; prohibit pull requests, GitHub Actions CI, or PR templates unless explicitly requested; use feature branches merged by fast-forward after user approval for larger/Phase 2+ changes.
- **Consequences:** No automated cloud CI exists; running `pytest` and `ruff` locally before reporting changes is mandatory.

### ADR-015: Phase 1 Working-Resolution NMS and GT-Free Diagnostics
- **Status:** **Implemented**
- **Sources:** `01_Project_Initialization_and_CV_Detectors.ipynb` Cells 2–4; [compare.py](../../src/constellation_detection/evaluation/compare.py); [nms.py](../../src/constellation_detection/detection/nms.py); [diagnostics.py](../../src/constellation_detection/evaluation/diagnostics.py).
- **Decision:** Standardize Phase 1 comparison inputs by downscaling to `max_dim=800` via `cv2.INTER_AREA`, suppressing nearby peaks via shared `fast_nms_stars(min_dist_px=10.0, max_candidates=250)` (with `max_candidates * 2 = 500` pre-truncation cap), and comparing detectors via `total_stars`, `top30_snr` (border margin `11 px`, no backfill), and `latency_ms`.
- **Consequences:** Established exact numerical reproducibility across Colab and local runs, while introducing known scale-sensitivity, saturated-star attenuation, and truncation limitations documented in [ARCHITECTURE.md](./ARCHITECTURE.md#28-known-limitations).

### ADR-016: Phase 1 Colab-to-Package Migration Adjustments
- **Status:** **Implemented** *(Minor Divergences from Colab Notebook Documented)*
- **Sources:** [threshold.py](../../src/constellation_detection/detection/traditional/threshold.py); [tophat.py](../../src/constellation_detection/detection/traditional/tophat.py); [dog.py](../../src/constellation_detection/detection/traditional/dog.py); [compare.py](../../src/constellation_detection/evaluation/compare.py); [export.py](../../src/constellation_detection/evaluation/export.py); [overlays.py](../../src/constellation_detection/visualization/overlays.py).
- **Decision:** During Phase 1 extraction from `01_Project_Initialization_and_CV_Detectors.ipynb` to `src/constellation_detection/`, the following changes were implemented:
  1. Corrected `metrics_df["resolution"]` from native dimensions to actual working dimensions.
  2. Exported unrounded `float64` prediction coordinates and wrote header-only CSVs on zero detections.
  3. Hardened centroid edge fallbacks (`np.mean` contour vertices in CV1; unweighted pixel mean in CV2; `round(bx/by)` in CV3).
  4. Changed `CV3` visualization mask circles from hollow (`thickness=1`) to filled (`thickness=-1`) and rendered 1×4 per-detector PNGs instead of 3×4 combined grids.
  5. Renamed CV1 identifier to `"global_threshold"` and hardcoded detector thresholds/kernel parameters as module constants rather than constructor arguments.
- **Consequences:** Zero impact on 5-image `total_stars` / `top30_snr` parity (`100%` match preserved), but CV1 naming inconsistency and removed parameter configurability remain open for review (`UD-005`, `UD-006`).

---

## 4. Compact Reference: Superseded & Rejected Approaches

| Type | Approach / Proposal | Reference | Why Superseded / Rejected |
|---|---|---|---|
| **Superseded** | Compare TensorFlow/Keras and PyTorch via controlled Phase 2 prototype (`ADR-003-HIST`) | `project-brief.md` §3, §6 | Replaced by PyTorch-only policy (`ADR-003`) in `project-brief_v2.md` |
| **Superseded** | Original Sem V inline notebook as active baseline | `constellation-extraction.ipynb` | Replaced by modular Phase 1 CV detectors (`ADR-005`) |
| **Superseded** | Colab `/content/constellation_vision` hardcoded paths & inline cells | `01_Project_Init...ipynb` | Replaced by modular package + `ProjectPaths` (`ADR-009`) |
| **Superseded** | GitHub Actions CI & PR template workflow | Commit `71808a0` | Removed in favor of local-only Phase 1 workflow (`ADR-014`) |
| **Rejected** | GPS, compass, gyroscope, device-orientation constellation lookup | `project-brief_v2.md` §1, §3 | Violates purely vision-based core research objective (`ADR-001`) |
| **Rejected** | CLAHE preprocessing on Top-Hat star maps | `01_Project_Init...ipynb` L163 | Amplified dark-sky background noise (`ADR-006`) |
| **Rejected** | Arbitrary pixel-distance line connection (`max_distance = 225/140`) | `constellation-extraction.ipynb` | Non-scale/FOV-invariant; does not recognize constellations (`ADR-007`) |
| **Rejected** | Treating Phase 1 CSVs or synthetic demo images as ground truth | `project-brief_v2.md` L196; `README.md` | Invalidates supervised evaluation & causes circular bias (`ADR-008`) |
| **Rejected** | Copying TensorFlow/Keras / `VGG16` / pre-split-scaled lab code | `combined-deep-learning-notebook.ipynb` | Wrong framework, data leakage, bugs, CPU memory incompatibility (`ADR-013`) |

---

## 5. Unresolved Questions and New Decisions Requiring User Approval

The following decisions were identified during Task 01 and **must not be improvised without explicit user approval**:

### UD-001 — Archival / Git Ingestion of External Files (`d:\potential captone\`) & Brief Reconciliation
- **Question:** Should `project-brief_v2.md` (and `project-brief.md`), `constellation-extraction.ipynb`, and `01_Project_Initialization_and_CV_Detectors.ipynb` remain external in `d:\potential captone\`, or be archived inside the repository (note the two notebooks total `~13.5 MB` with embedded PNG outputs), and should `project-brief_v2.md` misstatements (CLAHE, omitted `SciPy`/`scikit-image`, `~1.14 TB` storage) be amended?

### UD-002 — Repository State & Stale Artifacts Cleanup
- **Question:** Local artifacts currently contain stale `src/constellation_detection.egg-info/requires.txt` (still listing removed `Pillow`/`PyYAML`), orphaned `outputs/visualizations/phase1_detector_comparison_summary.png` (from pre-`71bc41b` commit `f1ec613`), and 6 historical local branches alongside `main`. Should `.egg-info` be refreshed and orphaned outputs regenerated or removed?

### UD-003 — Verified Ground-Truth Dataset Acquisition & Labeling Protocol *(Pre-Phase 2 Gate)*
- **Question:** Which verified ground-truth strategy should govern Phase 2 dataset construction?
  - *Option A:* Blind astrometric plate solving (e.g., Astrometry.net WCS) projecting verified Hipparcos/Yale BSC5/Gaia stars to native `(x, y)` up to an explicit limiting magnitude, plus manual QA filtering.
  - *Option B:* Manual expert point annotation on a curated subset.
  - *Option C:* Physically realistic synthetic starfield simulation for pretraining/ablation paired strictly with plate-solved real astrophotography/smartphone holdout test sets.

### UD-004 — Phase 2 PyTorch Star-Detection Problem Formulation & First Architecture
- **Question:** Which formulation should be implemented first in PyTorch under `src/constellation_detection/detection/dl/`?
  - *Option A:* Compact encoder-decoder / U-Net style Gaussian heatmap + sub-pixel keypoint detector (supports arbitrary/tiled resolutions and fits CPU RAM constraints).
  - *Option B:* Patch-based CNN classifier/refiner filtering CV candidates (e.g., rejecting watermarks/galaxies/nebulae).
  - *Option C:* Bounding-box object detector (discouraged for 2–6 px point sources and CPU cost).

### UD-005: Phase 1 Baseline Preservation vs. Saturated-Star and NMS Fixes
- **Question:** Should frozen Phase 1 detectors remain strictly untouched (`ADR-005`) while exposing optional parameters and native-to-working scale metadata cleanly, and should saturated-star failures (`Betelgeuse`, `Rigel`) + `500`-candidate pre-NMS truncation be addressed strictly in a **separate** Phase 1.5 / Phase 4 hybrid variant rather than mutating Phase 1 baseline numbers?

### UD-006 — Standardizing `global_threshold` vs. `CV1_GlobalThreshold` Naming
- **Question:** Should `threshold.py`'s `DETECTOR_NAME` remain `"global_threshold"` for backwards compatibility with existing unit tests (`test_threshold.py`, `test_compare_and_export.py`), or be aligned back to `"CV1_GlobalThreshold"` (requiring coordinated test/UI updates)?

### UD-007 — Target Constellation Subset & Multi-Constellation Labeling Scope
- **Question:** Which exact list of constellations constitutes `Initial recognition scope: Multiple selected constellations` (`project-brief_v2.md` §3), and should multi-constellation wide-field frames (`Cygnus` + `Lyra` + `Aquila`; `Ursa Minor` + `Ursa Major`) support multi-label ground truth from Phase 3 onward?

### UD-008 — Sample Image Copyright Verification & Phase 2 Environment Compatibility
- **Question:** Given verified XMP attribution to copyrighted photographer **Akira Fujii** on the 5 ESA/Hubble samples (contradicting `scripts/download_phase1_samples.py`'s `"Public-domain / CC-BY"` docstring), how should the sample docstring and future openly licensed benchmark imagery (strict CC0/CC-BY) be handled? Additionally, confirm target Python version policy between local Python `3.13.17` (`numpy 2.5.4`) and Colab Python `3.10–3.12` before installing `[dl]` (`PyTorch`).
