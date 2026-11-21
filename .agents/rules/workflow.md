---
trigger: always_on
description: Core operating procedure for Research, Implementation, and Review modes, plus human approval gates in the constellation-detection repository.
---

# Research Operating System Workflow

Every task in this repository must execute within one of three explicit operating modes. Identify and state the active mode at the start of work. If a user request is ambiguous between modes or spans research into unapproved implementation, ask for clarification or complete Research Mode first and request approval.

## Mode 1: Research and Planning

**Use when**: Surveying literature, analyzing architecture options, designing experiments or Phase 2+ pipelines, auditing datasets in read-only fashion, or resolving open design questions (`UD-001` through `UD-008`).

**Mandatory Procedure**:
1. Inspect authoritative repository documents: `docs/context/ARCHITECTURE.md`, `docs/context/DECISIONS.md`, and `research/INDEX.md`.
2. Inspect relevant source modules in `src/constellation_detection/` and tests in `tests/`.
3. Search academic and technical primary references when evaluating methods, catalogues, or metrics.
4. Compare at least two viable alternatives explicitly against project constraints (vision-only per `ADR-001`, PyTorch-only per `ADR-003`, local 8 GiB CPU inference, Colab GPU training, zero verified GT per `ADR-008`).
5. Explicitly separate verified facts, existing code, proposed design, assumptions, and unresolved uncertainties.
6. Produce a cited research or experiment specification with concrete acceptance criteria and an approval checklist.

**Strict Lock**: Do NOT modify implementation files (`src/`, `tests/`, `scripts/`, `notebooks/`, `pyproject.toml`, `requirements-colab.txt`).

## Mode 2: Implementation and Experimentation

**Use when**: Implementing an explicitly approved specification, bug fix, test addition, or approved experiment.

**Mandatory Procedure**:
1. Read the approved specification or governing decision record (`docs/context/DECISIONS.md`).
2. Inspect all affected files and check `git status` plus branch policy (`main` for routine low-risk edits vs. dedicated branch for substantial/Phase 2+ changes).
3. Identify upstream/downstream dependencies, schema contracts, and Phase 1 baseline invariants.
4. Implement strictly approved changes only; reject scope creep and silent algorithmic rewrites.
5. Add or update unit/regression tests without fabricating ground-truth annotations.
6. Execute verification in the existing `.venv` (`.venv\Scripts\python.exe -m pytest`, `.venv\Scripts\ruff.exe check .`, and parity/reproduction checks where applicable).
7. Record exact commands executed, observed outputs, timings/counts, and any environment gaps.

## Mode 3: Independent Technical and Scientific Review

**Use when**: Reviewing uncommitted changes, a feature branch, a compare diff against `main`, a notebook, or an experiment result.

**Mandatory Procedure**:
1. Inspect `git status` and the full `git diff` (or commit diff).
2. Read the complete surrounding source files—not isolated diff hunks—and corresponding test files.
3. Evaluate scientific and astronomical assumptions (GT provenance, data leakage, chirality/coordinates, sensor exclusion, baseline parity).
4. Evaluate numerical and software correctness (`(x, y)` vs. `img[iy, ix]` indexing, `max_dim=800` vs. native scale, edge cases, CPU portability, typing, docstrings).
5. Check strict compliance with `AGENTS.md`, `docs/context/ARCHITECTURE.md`, and `docs/context/DECISIONS.md`.
6. Report findings ordered by severity (`Critical/Blocker`, `High`, `Medium`, `Low`, `Verification Gap`) with concrete file paths and line numbers.

**Strict Locks**:
- Do NOT modify implementation files during Review Mode.
- NEVER automatically approve the implementation agent's work. Only state `Ready for User Approval`, `Changes Requested`, or `Blocked Pending Evidence` strictly backed by verified checks.

## Human Approval Gates

Stop and obtain explicit user approval before:
- Transitioning from a Research Mode specification to Implementation Mode code changes.
- Resolving any unresolved decision (`UD-001` through `UD-008`).
- Modifying Phase 1 baseline detector logic, thresholds, kernels, NMS, or SNR parameters.
- Adding Python dependencies (`pyproject.toml`, `requirements-colab.txt`) or creating/modifying virtual environments.
- Defining or altering annotation, catalogue, prediction, or dataset-split schemas.
- Launching Phase 2+ deep-learning training runs or creating cloud/Colab execution pipelines.
- Merging any branch into `main` or performing destructive filesystem/Git operations.

## Workspace Skill Routing

| Skill | Allowed Mode(s) | Primary Output |
| :--- | :--- | :--- |
| `literature-review` | Research | Cited research specification & comparison matrix |
| `dataset-audit` | Research / Review | Dataset/catalogue provenance, leakage & coordinate audit report |
| `experiment-design` | Research | Hypothesis-driven reproducible experiment specification |
| `experiment-reproduction` | Implementation / Review | Quantitative parity & verification report |
| `scientific-code-review` | Review | Independent severity-ranked scientific code review |
