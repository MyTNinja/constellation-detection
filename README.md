# Vision-Based Constellation Detection and Recognition

A research project for detecting stars in night-sky images and, in later phases, recognizing constellation patterns from visual evidence. The project does not use GPS, compass, gyroscope, or other device sensors to identify a constellation.

## Project status

**Phase 1 is complete. Phase 2, PyTorch-based star detection, has not started.**

The completed Phase 1 implementation is the traditional computer-vision baseline. It must remain available for comparison with future models. Constellation recognition, hybrid integration, and the local upload application are planned for later phases and are not implemented yet.

## Phase 1 baseline

The reusable Python package provides:

- Image loading, validation, grayscale conversion, and Gaussian smoothing.
- Global-threshold, adaptive top-hat, and multi-scale Difference-of-Gaussians star detectors.
- Non-maximum suppression and star coordinate extraction.
- Detector comparison, signal-to-noise diagnostics, runtime measurements, CSV prediction export, and visualization overlays.
- Unit tests for the image, detection, evaluation, export, and visualization code.

The demo compares detector outputs. The downloaded sample images do not include verified star-level annotations, so the project does **not** report detection precision, recall, or F1 yet. CSV files contain predictions only; they are not ground truth.

## Repository layout

```text
notebooks/
  phase1_traditional_cv_demo.ipynb  # Runs the Phase 1 detector comparison
scripts/
  download_phase1_samples.py        # Optional ESA/Hubble sample downloader
src/constellation_detection/
  config.py                         # Configurable project, data, and output paths
  image/                            # Image loading and preprocessing
  detection/                        # Shared NMS and traditional detectors
  evaluation/                       # Detector comparison and prediction exports
  visualization/                    # Diagnostic overlays
tests/                              # Automated tests
```

Datasets and generated results are local artifacts and are excluded from Git. The optional sample downloader puts images in `data/raw/`; the demo writes predictions and visualizations under `outputs/`.

## Local setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
```

Activate the environment, then install the package and development tools:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

To prepare the optional sample images:

```bash
python scripts/download_phase1_samples.py
```

Open `notebooks/phase1_traditional_cv_demo.ipynb` from the repository root to run the detector comparison. If no sample images are present, the notebook uses a seeded synthetic starfield for a demonstration; this synthetic image is not ground truth and must not be used to claim real-image detection accuracy.

Run the local checks with:

```bash
python -m pytest
python -m ruff check .
```

## Google Colab

`requirements-colab.txt` installs the local package without replacing Colab's preinstalled PyTorch/CUDA stack.

For Phase 2, install the optional PyTorch development dependencies locally with `python -m pip install -e ".[dl,dev]"`, or use the Colab setup above.

## Research rules

- Keep Phase 1 detector behavior and results as the comparison baseline.
- Keep detector/model predictions separate from independently verified annotations.
- Do not fabricate labels or use predictions as ground truth.
- Record experiment settings and random seeds; keep training, validation, and test data separated.
- Use PyTorch for new deep-learning work. TensorFlow/Keras comparisons are out of scope.
- Recognition and constellation matching are future work; detector outputs alone do not identify a constellation.
