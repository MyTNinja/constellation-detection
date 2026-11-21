---
name: dataset-audit
description: >-
  Audits candidate night-sky image datasets, star annotations, and astronomical
  reference catalogues for provenance, licensing, ground-truth separation,
  coordinate validity, split leakage, and imaging hazards before Phase 2+ use.
---

# Dataset Audit Skill

## Trigger

Use in **Research Mode** or **Review Mode** before accepting any night-sky image collection, star detection annotation set, constellation label set, or reference star catalogue into training, validation, or evaluation workflows.

## Guardrails

- Follow `.agents/rules/dataset-integrity.md` and `ADR-008` / `ADR-011`.
- Never treat files in `outputs/predictions/` as ground truth.
- Never fabricate missing annotations or modify source datasets during an audit.
- Do not commit raw imagery, catalogues, or large data manifests to Git.

## Required Inputs

- Path(s) or description of the candidate dataset, annotation files, split definitions, or astronomical catalogue.
- Stated intended task (e.g., Phase 2 star detection training, Phase 3 constellation matching, or held-out evaluation).

## Procedure

1. **Check Architecture & Contracts**: Read Coordinate Conventions and Dataset Requirements in `docs/context/ARCHITECTURE.md` and `ADR-008`, `ADR-011` in `docs/context/DECISIONS.md`.
2. **Verify Provenance & Licensing**: Inspect READMEs/metadata for data origin, annotation creation method (e.g., WCS plate solve + catalogue match vs. manual labeling), epoch (J2000/ICRS), magnitude limits, and redistribution license (flag copyrighted assets such as Akira Fujii images).
3. **Verify Directory & Schema Separation**: Confirm ground-truth files are strictly separated from `outputs/predictions/` (`rank,x_px,y_px,flux_score`) and use explicit, documented schemas.
4. **Run Read-Only Programmatic Integrity Checks** (using `.venv\Scripts\python.exe`):
   - File readability, image dimensions `(H, W, C)`, dtype/bit depth (8-bit vs. 16-bit), EXIF orientation tags.
   - Annotation coordinate checks: verify `(x, y)` OpenCV bounds `[0, W-1]` / `[0, H-1]`, working (`max_dim=800`) vs. native pixel scale explicitness, absence of `NaN`/`Inf`, duplicate coordinates within NMS radius, and minimum annotation counts.
   - Leakage & split audit: hash image contents (exact SHA-256 and perceptual/resized hash) across proposed train/val/test splits and verify grouping by image / sky-field / session so no sky crop or burst spans splits.
   - Hazard scan: inspect dynamic range, saturation/clipping fraction (`255` or `65535`), borders/watermarks, and non-stellar diffuse structures.
5. **Determine Readiness Verdict**: Issue `PASS`, `CONDITIONAL` (list exact preconditions), or `FAIL` (block Phase 2+ training/evaluation).

## Expected Output

A **Dataset Audit Report** containing:
- **1. Asset & Provenance Summary** (source, license, epoch, schema, path separation)
- **2. Programmatic Verification Results** (exact scripts/checks run, counts, bounds, hash-overlap checks)
- **3. Coordinate & Astronomical Integrity Findings**
- **4. Split & Data Leakage Assessment** (plus augmentation chirality advisories)
- **5. Readiness Verdict (`PASS` / `CONDITIONAL` / `FAIL`) & Required Remediation**
