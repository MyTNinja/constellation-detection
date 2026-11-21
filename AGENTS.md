# Coding Rules & Guidelines for AI Agents (`AGENTS.md`)

This document defines the mandatory engineering and research standards for AI coding agents working on **Vision-Based Constellation Detection and Recognition Using Hybrid Computer Vision and Deep Learning**.

## 1. Research Integrity & Logic Preservation

- **Preserve existing research logic**: Do not alter, simplify, or replace existing algorithms, heuristics, or mathematical formulations unless explicitly authorized by the user.
- **Do not fabricate ground-truth annotations**: Never invent synthetic ground-truth star coordinates, constellation labels, or bounding boxes to pass tests or fill missing data.
- **Keep predictions separate from ground-truth labels**: Maintain strict separation between ground-truth annotations and model/algorithm predictions in data structures, schemas, and storage paths.
- **Avoid data leakage**: Enforce strict partitioning between training, validation, and test splits. Never compute dataset-wide statistics, thresholds, or normalizations using validation or test samples.
- **Reproducibility**: All experiments, dataset splits, and stochastic operations must use explicit random seeds and recorded configuration parameters.

## 2. Software Architecture & Portability

- **Modular, reusable Python code**: Implement all core computer vision and deep-learning logic in reusable `.py` modules inside `src/constellation_detection/`. Prefer focused functions and well-structured classes with single responsibilities.
- **Keep notebooks lightweight**: Jupyter notebooks in `notebooks/` are strictly for experimentation, interactive analysis, and visualization. Notebooks should import and call functions from `constellation_detection` rather than defining core algorithms inline.
- **Environment independence**: Core Python modules must never import `google.colab` or depend on Colab-specific APIs.
- **No hardcoded environment paths**: Never hardcode `/content`, `/content/drive`, or Windows-specific paths (e.g., `C:\...` or `D:\...`) inside reusable Python modules. Always accept paths as parameters or resolve them via `constellation_detection.config`.
- **Local CPU compatibility**: Ensure all PyTorch models, checkpoints, and inference pipelines remain compatible with local CPU execution (`map_location="cpu"` where needed) alongside Colab GPU training.

## 3. Code Quality & Dependencies

- **Type hints and docstrings**: Use standard Python type annotations (`from __future__ import annotations`) and concise, informative docstrings explaining inputs, outputs, units/coordinate conventions, and assumptions.
- **Minimal dependencies**: Avoid introducing new third-party libraries unless strictly justified. Rely on the core stack (NumPy, SciPy, OpenCV, scikit-image, Pandas, Matplotlib, and PyTorch).
- **Framework restriction**: Use **PyTorch** exclusively for deep learning. **Do not introduce TensorFlow, Keras, or JAX.**

## 4. Git & Review Workflow

### Routine, low-risk changes

- Work directly on `main` for clearly routine, low-risk changes (such as documentation edits, formatting, or isolated changes with narrow impact).
- Before starting, confirm the working tree is clean and update local `main` from `origin/main`.
- Make the change, inspect the diff and staged files, and run relevant local checks. For Python code changes, use the project's existing local environment to run relevant tests (`pytest`) and Ruff checks when available. Do not install dependencies or create a new environment without the user's approval.
- If checks fail, fix the issue or report it. Do not commit or push a failing change.
- After checks pass, commit and push directly to `main`. Routine changes do not require a separate ChatGPT review, branch, merge, or approval cycle (the user may request ChatGPT review when useful).
- Keep this workflow local for Phase 1: run Python and tests on the user's PC. Do not add Colab, cloud execution, or GitHub Actions.

### Larger or higher-risk changes

- Use a dedicated branch for substantial, multi-step, cross-cutting, algorithmic, dependency, interface, data-format, or Phase 2 work.
- Use one branch for the whole effort across multiple prompts; do not create a new branch for every prompt within that effort.
- Run relevant local checks and inspect the diff before sharing a GitHub compare link against `main` (`https://github.com/MyTNinja/constellation-detection/compare/main...<branch>`) at an agreed review point or when the work is ready.
- ChatGPT may review and advise using the compare link. Antigravity applies requested revisions on the same branch and shares an updated compare link.
- Wait for the user's explicit approval before merging the branch into `main` or pushing that merge.
- After approval, update local `main`, fast-forward merge the branch into `main`, and push `main`.

### Boundaries

- Do not create pull requests.
- ChatGPT reviews and advises when asked; it does not commit, push, or merge. Antigravity performs coding and Git operations. The user gives final approval for larger branch merges.
- Never force-push, rewrite Git history, delete branches, change GitHub settings, modify branch protection, install or enable extensions, or add GitHub Actions or PR templates unless the user explicitly asks.
- Verify that only intended files are staged. Keep datasets, credentials, and generated model artifacts out of commits.
