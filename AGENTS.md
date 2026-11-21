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

Follow this simple, human-approved Git workflow for every change:

1. **Sync and branch**: Update local `main` from `origin/main`, then create a dedicated feature branch. Never commit or push directly to `main`.
2. **Implement and test locally**: Implement the requested change and run relevant local checks (`python -m pytest` and `python -m ruff check .` when applicable).
3. **Push branch and share compare link**: Commit and push the feature branch to `origin`, then share its GitHub compare link against `main` (`https://github.com/MyTNinja/constellation-detection/compare/main...<branch>`) so the user can request a review from ChatGPT.
4. **Revise as requested**: Apply any requested revisions on the feature branch, push the updated branch, and share the updated compare link.
5. **Wait for explicit user approval**: Do not merge or push anything to `main` until the user explicitly approves.
6. **Fast-forward merge on approval**: After explicit user approval, update local `main`, fast-forward merge the approved feature branch into `main`, and push `main`.
7. **Roles and boundaries**:
   - Never create a Pull Request or merge automatically.
   - ChatGPT reviews and advises; the user gives final approval; Antigravity performs coding and Git operations.
   - Do not add GitHub Actions workflows, PR templates, branch protection rules, or repository setting changes unless explicitly requested.
   - Do not delete branches or rewrite Git history without explicit user authorization.
