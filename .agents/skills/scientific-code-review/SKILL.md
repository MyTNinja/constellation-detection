---
name: scientific-code-review
description: >-
  Performs an independent, read-only technical and scientific code review of workspace
  changes, feature branches, or notebooks against AGENTS.md, architecture decisions,
  numerical/coordinate invariants, and test integrity, reporting findings by severity.
---

# Scientific Code Review Skill

## Trigger

Use in **Review Mode** whenever reviewing code changes, a new module, a bugfix, a pull/compare diff against `main`, a notebook update, or an implementation agent's completed work.

## Guardrails

- Strictly read-only on repository code (`src/`, `tests/`, `scripts/`, `notebooks/`, `pyproject.toml`, `requirements-colab.txt`).
- Follow `.agents/rules/code-review.md`: NEVER auto-approve changes, never ignore missing tests, and never accept proxy metrics (`top30_snr`) as detection accuracy.

## Required Inputs

- Target scope: uncommitted working tree, specific commit range (`main..<branch>`), or named files/notebooks.
- Governing specification, ADR, or task prompt (if applicable).

## Procedure

1. **Gather Git Context**: Run `git status -sb` and `git diff` (plus `git diff --stat` against `main` for branch work). Verify no datasets, checkpoints, credentials, or unintended files are staged or modified.
2. **Read Full Context**: Read every changed file in full plus its direct callers, downstream consumers, and `docs/context/ARCHITECTURE.md` / `docs/context/DECISIONS.md` sections touched by the change.
3. **Inspect Tests**: Read matching files under `tests/`. Check whether tests assert real invariants (coordinate transforms, edge cases, no GT fabrication, determinism) or merely mock away failures.
4. **Run Read-Only Verification**: Run `.venv\Scripts\python.exe -m pytest` and `.venv\Scripts\ruff.exe check .`. Note exact pass/fail counts and any checks skipped due to absent GPU/PyTorch/dataset.
5. **Audit Six Verification Axes**:
   - **Scientific Validity & Baseline Integrity**: Phase 1 defaults (`global_threshold`, `CV2_AdaptiveTopHat` without CLAHE, `CV3_MultiScaleDoG`, `fast_nms_stars`, `compute_star_annulus_snr`) preserved unless approved; vision-only (`ADR-001`); no unverified accuracy claims (`ADR-008`).
   - **Coordinate & Numerical Correctness**: OpenCV `(x, y)` vs. `image[iy, ix]`, working (`max_dim=800`) vs. native pixel resolution (`ADR-011`), `float32`/`float64` handling, `m00 == 0` / `MAD == 0` / empty-array / `NaN` safety.
   - **Data Integrity & Leakage**: Strict prediction vs. GT separation, image/field splits, fit-on-train statistics, chirality-safe augmentations.
   - **Architecture & Portability**: PyTorch-only (`ADR-003`), no `google.colab` imports or hardcoded paths in `src/`, `map_location="cpu"` + 8 GiB RAM safety, type hints/docstrings, lightweight notebooks.
   - **Regression Risk**: Backward compatibility of public signatures, schemas (`rank,x_px,y_px,flux_score`), and parity checks.
   - **Git & Policy Compliance**: Branching rules (`main` vs. dedicated branch), no PRs/Actions/history rewrites, minimal unapproved dependencies.
6. **Compile Severity-Ranked Report**: Cite `file:line`, reproducible evidence, risk, and required fix for every issue. Issue `Changes Requested`, `Blocked Pending Evidence`, or `Ready for User Approval`.

## Expected Output

A **Scientific Code Review Report** containing:
- **1. Review Scope & Git Snapshot**
- **2. Verification Commands Executed & Results**
- **3. Findings by Severity** (`Critical / Blocker`, `High`, `Medium`, `Low`)
- **4. Verification Gaps** (untested paths or missing environment capabilities)
- **5. Review Verdict & Next Required Step**
