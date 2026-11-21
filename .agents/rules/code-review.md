---
trigger: model_decision
description: Independent technical and scientific code review protocol covering diff inspection, scientific validity, numerical/coordinate correctness, regression checks, architecture compliance, and severity-ranked findings. Activate whenever Review Mode or independent code review is requested.
---

# Independent Technical and Scientific Code Review Rules

## 1. Reviewer Independence & Scope Lock

- Operate strictly read-only on implementation files (`src/`, `tests/`, `scripts/`, `notebooks/`, `pyproject.toml`, `requirements-colab.txt`).
- Never automatically approve changes produced by an implementation turn or accept "tests pass" at face value without verifying what those tests actually assert.

## 2. Mandatory Review Inspection Sequence

1. Check `git status -sb` and read the full `git diff` (staged and unstaged, or branch diff vs. `main`).
2. Read the complete modified and surrounding files (upstream callers, downstream consumers, shared utilities) plus the governing specification or `docs/context/DECISIONS.md` entry.
3. Read all corresponding tests to verify meaningful assertions rather than tautologies or fabricated ground truth.
4. Execute read-only verification in `.venv`: `.venv\Scripts\python.exe -m pytest` and `.venv\Scripts\ruff.exe check .`.

## 3. Five Review Evaluation Axes

1. **Scientific & Research Correctness**: Phase 1 baseline preservation (no unapproved changes to `global_threshold`, `CV2_AdaptiveTopHat` no-CLAHE rule, `CV3_MultiScaleDoG`, `fast_nms_stars`, or `compute_star_annulus_snr`); no sensor shortcuts (`ADR-001`); no unverified accuracy claims (`ADR-008`); zero train/val/test leakage; valid astronomical chirality/PSF assumptions.
2. **Numerical & Coordinate Correctness**: Verify OpenCV `(x, y)` vs. NumPy `image[iy, ix]` indexing, `max_dim=800` vs. native resolution consistency (`ADR-011`), sub-pixel `float32`/`float64` precision, `m00 == 0` / empty-array / `NaN` / border-margin guards, and deterministic NMS sorting.
3. **Architecture & Portability Compliance**: Adheres to `AGENTS.md`, `docs/context/ARCHITECTURE.md`, and `docs/context/DECISIONS.md`; PyTorch-only (`ADR-003`); no `google.colab` imports or hardcoded paths in `src/`; CPU `map_location="cpu"` compatibility within 8 GiB RAM; lightweight notebooks; no unapproved packages.
4. **Regression & Test Quality**: Tests cover boundary/degenerate cases without mocking away real invariants or inventing ground truth; Phase 1 numerical parity remains intact where relevant.
5. **Git & Scope Discipline**: Only approved changes appear; no datasets, weights, secrets, or generated outputs tracked; Git branch policy respected (no PRs, force-pushes, or CI edits).

## 4. Severity Classification & Report Format

Group every finding under one of these headings, citing exact `file:line`, observed evidence, impact, and required fix:
- **Critical / Blocker**: Scientific invalidity, data leakage, fabricated GT, Phase 1 baseline corruption, prohibited framework/sensor use, or broken core logic.
- **High**: Coordinate scale/axis bug, numerical instability, CPU/Colab portability violation, or missing tests on core paths.
- **Medium**: Maintainability, incomplete edge-case coverage, typing/docstring omissions, minor specification drift.
- **Low**: Readability, naming, or minor formatting improvements.
- **Verification Gaps**: Checks that could not run locally (e.g., GPU training, missing CUDA/PyTorch, unprovided external dataset).

Conclude every review with an explicit verdict and rationale:
- `Changes Requested` — blockers or high-severity findings must be fixed.
- `Blocked Pending Evidence` — verification gaps prevent safe approval.
- `Ready for User Approval` — all verified checks pass with zero Critical/High issues (explicit user approval still required before merge).
