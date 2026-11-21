---
name: experiment-design
description: >-
  Designs reproducible traditional CV, PyTorch deep learning, or hybrid constellation
  recognition experiments in Research Mode, specifying hypotheses, baseline controls,
  leakage-free splits, CPU/Colab budgets, and approval gates before writing code.
---

# Experiment Design Skill

## Trigger

Use in **Research Mode** before creating any Phase 2+ PyTorch training script, Phase 3 constellation matching prototype, Phase 4 hybrid pipeline, or ablation study.

## Guardrails

- Strictly read-only on implementation code until the user approves the design.
- Follow `.agents/rules/deep-learning.md`, `.agents/rules/computer-vision.md`, and `.agents/rules/dataset-integrity.md`.
- Enforce `ADR-001` (vision-only), `ADR-003` (PyTorch-only), `ADR-008` (no unverified accuracy claims without verified GT), and `ADR-011` (explicit coordinate resolution).

## Required Inputs

- Experiment goal / Phase (`Phase 2`, `Phase 3`, `Phase 4`, or targeted CV ablation).
- Approved research specification or open `UD-*` target (if available).
- Available dataset & ground-truth status (audited via `dataset-audit`).

## Procedure

1. **Inspect Baseline & Constraints**: Review relevant modules in `src/constellation_detection/` plus `docs/context/ARCHITECTURE.md` and `docs/context/DECISIONS.md`.
2. **Verify Ground-Truth Gate**: Check whether independently verified GT exists for the experiment. If absent, explicitly restrict the proposal to either: (a) synthetic/smoke pipeline verification (cannot claim real-sky accuracy), or (b) blocked pending GT acquisition/annotation.
3. **Define Experimental Protocol**:
   - **Research Question & Falsifiable Hypothesis**
   - **Frozen Baseline Controls**: Always benchmark against `global_threshold`, `CV2_AdaptiveTopHat`, and `CV3_MultiScaleDoG` on identical test images and coordinate scale.
   - **Data & Split Protocol**: Image/sky-field split manifest, fit-on-train normalizations, chiral-safe augmentations (no unintended mirror flips).
   - **Model, Loss & Architecture**: PyTorch module design, parameter/FLOPs budget, `map_location="cpu"` compatibility, and ~8 GiB local RAM feasibility.
   - **Evaluation Metrics**: Distinguish GT-backed metrics (Hungarian-matched precision/recall/F1 at pixel tolerance `r`, centroid RMSE) from proxy unsupervised diagnostics (`top30_snr`, runtime ms).
   - **Reproducibility & Logging**: Exact seed strategy (`random`, NumPy, Torch, DataLoader workers), artifact logging schema, and runtime parity checks.
4. **Separate Execution Surfaces**: Define what runs on Google Colab GPU (`requirements-colab.txt` training/sweeps) vs. local Windows CPU (`pytest`, unit checks, `map_location="cpu"` inference).
5. **Define Acceptance Criteria & Stop Rules**: Pre-register quantitative success thresholds and failure criteria before seeing test results.

## Expected Output

An **Experiment Design Specification** containing:
- **1. Objective, Hypothesis & GT Gate Status**
- **2. Dataset, Splits & Coordinate Contract**
- **3. Model / Algorithm Architecture & PyTorch Budget**
- **4. Training, Reproducibility & Ablation Plan**
- **5. Evaluation Metrics & Frozen Phase 1 Controls**
- **6. Local PC vs. Google Colab Execution Split**
- **7. Acceptance Criteria & Explicit User Approval Checklist**
