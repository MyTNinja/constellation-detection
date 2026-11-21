---
trigger: model_decision
description: PyTorch-only deep learning standards, Colab GPU training vs. local CPU inference portability, loss and architecture constraints, seeded reproducibility, and baseline comparison discipline. Activate when planning, implementing, or reviewing Phase 2+ neural networks, datasets, training loops, or PyTorch code.
---

# Deep Learning Engineering and Reproducibility Rules (PyTorch)

## 1. Framework & Legacy Constraints

- **PyTorch Exclusively (`ADR-003`)**: Implement all neural networks, data loaders, losses, and checkpoints in PyTorch (`torch>=2.0`, `torchvision>=0.15`). **TensorFlow, Keras, and JAX are prohibited.**
- **Do Not Port Legacy Classifier**: Do not directly port `combined-deep-learning-notebook.ipynb` (43-constellation single-label whole-image classification trained without star localization or verified ground truth) as the Phase 2 architecture.
- **Core Dependencies**: Avoid adding heavy wrapper frameworks (e.g., Lightning, custom training suites) unless explicitly approved.

## 2. Colab GPU Training vs. Local CPU Deployment

- **Training Target**: Train Phase 2+ models on Google Colab GPU using `requirements-colab.txt` (installing `constellation-detection` editable `-e .` while preserving Colab's native GPU PyTorch/CUDA build).
- **Local CPU Compatibility**: Local development runs on a 4-core Windows machine with ~8 GiB RAM and **no GPU / CUDA**. Every inference pipeline, checkpoint loader, and unit test in `src/` must:
  - Load weights safely with `map_location="cpu"` (never assume CUDA is present).
  - Never import `google.colab` or hardcode `/content`, `/content/drive`, `C:\`, or `D:\` paths.
  - Guard GPU-only code cleanly or skip GPU-requiring tests cleanly when `torch` or CUDA is unavailable.
  - Respect an ~8 GiB local RAM ceiling during full-image CPU inference (e.g., bounded input resolution or tiled inference with deterministic stitching).

## 3. Architecture, Loss & Supervision Discipline

- **Verified Ground Truth Gate (`ADR-008`)**: Do not train or benchmark supervised star detectors on unverified Phase 1 CSV predictions as pseudolabels or fabricated annotations.
- **Class Imbalance**: Point-source star detection exhibits extreme foreground-to-background sparsity; select and justify formulation-appropriate losses (e.g., heatmap regression, focal loss, or weighted BCE) and sub-pixel localization refinement.
- **No Sensor Inputs (`ADR-001`)**: Model inputs must derive solely from image pixels and pure vision pipelines—never GPS, IMU, compass, or EXIF sky-orientation shortcuts.

## 4. Reproducibility & Scientific Evaluation

- **Deterministic Seeding**: Explicitly seed Python `random`, NumPy, `torch.manual_seed`, `torch.cuda.manual_seed_all` (when CUDA active), and `DataLoader` `worker_init_fn` / generators. Log deterministic settings (`torch.use_deterministic_algorithms` / `cudnn.deterministic` trade-offs).
- **Run Provenance**: Record exact config parameters, random seeds, split manifest ID/hash, PyTorch/CUDA versions, hardware device, Git commit SHA, and checkpoint SHA-256 hash outside Git-tracked code paths.
- **Fair Baseline Comparison**: Benchmark every Phase 2+ model against the frozen Phase 1 detectors (`global_threshold`, `CV2_AdaptiveTopHat`, `CV3_MultiScaleDoG`) on identical test splits, image resolutions, and matching tolerances, reporting parameter count, RAM/VRAM usage, and CPU latency alongside accuracy metrics.
