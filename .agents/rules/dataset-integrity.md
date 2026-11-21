---
trigger: model_decision
description: Annotation provenance, strict ground-truth vs. prediction separation, image-level dataset splitting, data leakage prevention, astronomically valid augmentations, and catalogue/dataset versioning. Activate when sourcing, auditing, splitting, augmenting, or loading night-sky datasets, annotations, or star catalogues.
---

# Dataset Integrity, Provenance, and Leakage Prevention Rules

## 1. Ground-Truth Provenance & Schema Discipline (`ADR-008`)

- **Current Status**: Zero verified ground-truth star annotations or constellation labels exist in the repository.
- **Provenance Requirements**: Any future ground-truth dataset accepted for Phase 2+ must document:
  1. Source imagery origin, license/copyright status, and sensor/exposure metadata (if known, without violating `ADR-001` at inference time).
  2. Annotation creation and independent verification method (e.g., plate-solved WCS via Astrometry.net cross-matched with Hipparcos/Gaia/Yale BSC, or documented manual verification).
  3. Exact coordinate frame (`native` vs. `working` pixel resolution and dimension mapping) and celestial epoch (`J2000` / `ICRS`) when astrometric.
  4. Bright-star saturation, faint-magnitude cutoff, and exclusion rules (e.g., hot pixels, aircraft trails, watermarks).

## 2. Strict Separation of Ground Truth and Predictions

- **Predictions Only**: `outputs/predictions/` (`rank,x_px,y_px,flux_score`) holds algorithm/model outputs strictly and must never be used as ground truth or training targets.
- **Future Annotations**: Ground-truth annotations belong in a dedicated untracked path (e.g., `data/annotations/`) with explicit versioned schemas distinct from `outputs/predictions/`.
- **Zero Fabrication**: Never invent synthetic ground-truth labels inside datasets, metrics scripts, or tests. Clearly mark unit-test synthetic Gaussian images as synthetic test fixtures, never astronomical GT benchmarks.

## 3. Split Partitioning & Data Leakage Prevention

- **Group-Level Splitting**: Split datasets strictly at the **image, sky-field, or observing-session level**. Never place overlapping image patches, multi-exposure bursts of the same static shot, or near-duplicate crops across train/validation/test boundaries.
- **Strict Fit-on-Train Rule**: Compute image normalizations, background statistics, threshold priors, class weights, and hyperparameter sweeps strictly on training (or train/val) splits—never across the held-out test set or full dataset.
- **Duplicate Check**: Verify file hashes and perceptual/astrometric overlap across splits before training or evaluation.

## 4. Astronomically Valid Data Augmentation

- **Coordinate Consistency**: Every geometric transformation (crop, resize, rotation, translation) must transform target `(x, y)` coordinates identically and cleanly prune stars falling outside image boundaries or `edge_margin`.
- **Chirality / Handedness Warning**: Horizontal or vertical mirror flips invert sky handedness (chirality), reversing constellation patterns and astrometric parity (`+det(WCS)` vs. `-det(WCS)`). Prohibit mirror flips for Phase 3+ constellation recognition and astrometric matching unless explicitly evaluated and justified for isolated local PSF detection only.
- **Photometric Realism**: Avoid aggressive color jitter, histogram equalization (`CLAHE`), or unnatural sharpening that alters point-spread function morphology relative to real camera noise unless empirically ablation-tested.

## 5. Versioning & Copyright Compliance

- Keep raw imagery, large catalogues, manifests, and model checkpoints untracked in Git (`.gitignore`); commit only lightweight metadata/schemas and deterministic split-generation configs/checksums when appropriate.
- Respect external copyrights: **do not bundle or redistribute copyrighted third-party astrophotography** (including Akira Fujii sample plates referenced in Phase 1).
