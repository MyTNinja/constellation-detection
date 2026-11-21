---
name: experiment-reproduction
description: >-
  Reproduces and verifies Phase 1 traditional CV baselines or Phase 2+ PyTorch
  experiment runs against recorded metrics, validating environment health,
  pytest/ruff status, deterministic parity, and coordinate/schema integrity.
---

# Experiment Reproduction Skill

## Trigger

Use in **Implementation Mode** (post-change verification) or **Review Mode** (independent check) when verifying that Phase 1 baselines or recorded experiment results reproduce accurately on the local environment or Colab.

## Guardrails

- Do NOT modify Phase 1 baseline algorithms or parameters to force numbers to match.
- Report exact observed counts, metrics, command strings, and versions—never extrapolate or hide discrepancies.
- On the local Windows PC (no GPU/PyTorch installed), verify PyTorch CPU availability first and report GPU checks cleanly as `Verification Gap` if `torch`/CUDA is absent.

## Required Inputs

- Target experiment, detector, or baseline to reproduce (default: Phase 1 traditional CV suite).
- Expected reference metrics/outputs (e.g., Phase 1 parity table in `docs/context/ARCHITECTURE.md`).

## Procedure

1. **Inspect Environment & Repo State**:
   - Run `git status -sb` and `git rev-parse --short HEAD` to detect uncommitted modifications or drift.
   - Check `.venv` Python and installed package versions (`numpy`, `scipy`, `cv2`, `skimage`, `pandas`, plus `torch` / CUDA status if Phase 2+).
2. **Run Core Health Suite**:
   - Execute `.venv\Scripts\python.exe -m pytest` and `.venv\Scripts\ruff.exe check .`. Record exact test pass/fail/skip counts.
3. **Execute Target Pipeline** (read-only or writing only to untracked scratch/output paths as intended):
   - For **Phase 1 CV baseline**, run the phase-1 comparison/export pipeline on the 5 downloaded Phase 1 sample images in `data/` using `.venv\Scripts\python.exe`, verifying no crash and capturing per-image counts (`n_stars`), `top30_snr`, `latency_ms`, and CSV outputs (`outputs/predictions/`).
4. **Verify Numerical Parity & Schema**:
   - Compare reproduced results against documented Phase 1 reference benchmarks:
     - `global_threshold`: mean `192.2` stars, mean `top30_snr` `22.61`
     - `CV2_AdaptiveTopHat`: mean `250.0` stars, mean `top30_snr` `25.19`
     - `CV3_MultiScaleDoG`: mean `185.4` stars, mean `top30_snr` `26.78`
   - Verify prediction CSV schema (`rank,x_px,y_px,flux_score`), deterministic ordering, absence of `NaN`/`Inf`, and explicit working-resolution coordinate frame (`max_dim=800`).
5. **Diagnose Divergences**: If numbers differ, inspect library version changes (`opencv`, `scikit-image`, `scipy`), input resolution differences, RNG seeding, or code diffs rather than guessing.

## Expected Output

A **Reproduction Verification Report** containing:
- **1. Environment & Git Snapshot** (commit SHA, dirty state, Python & package versions, CPU/GPU status)
- **2. Test & Lint Results** (exact commands and pass/fail counts)
- **3. Reference vs. Reproduced Comparison Table** (exact numbers and deltas)
- **4. Schema & Coordinate Contract Checks**
- **5. Verdict (`Exact Match` / `Within Tolerance` / `Diverged` / `Blocked`) & Divergence Diagnosis**
