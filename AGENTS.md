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
- **Minimal dependencies**: Avoid introducing new third-party libraries unless strictly justified. Rely on the core stack (NumPy, SciPy, OpenCV, scikit-image, Pandas, Matplotlib, Pillow, PyYAML, and PyTorch).
- **Framework restriction**: Use **PyTorch** exclusively for deep learning. **Do not introduce TensorFlow, Keras, or JAX.**
