---
trigger: model_decision
description: Traditional computer vision standards, Phase 1 star detection baseline protection, OpenCV vs. native coordinate conventions, source localization, astronomical image hazards, and detection evaluation discipline. Activate when reading, designing, implementing, or reviewing traditional CV detectors, image preprocessing, NMS, SNR diagnostics, overlays, notebooks, or CV tests under src/constellation_detection/, notebooks/, or tests/.
---

# Computer Vision and Astronomical Star Detection Rules

## 1. Phase 1 Baseline Protection

Do not modify existing algorithms, mathematical formulations, or default parameters in `src/constellation_detection/` unless explicitly authorized:
- `resize_to_max_dim`: `max_dim=800`, interpolation `cv2.INTER_AREA`, returning image and scale factors.
- `detect_stars_global_threshold` (`global_threshold`, Colab `CV1_GlobalThreshold`): background statistics + global thresholding, contour/component centroiding, shared NMS.
- `detect_stars_adaptive_tophat` (`CV2_AdaptiveTopHat`): `19x19` elliptical White Top-Hat background suppression; **CLAHE is explicitly excluded** (`ADR-006`).
- `detect_stars_multiscale_dog` (`CV3_MultiScaleDoG`): `skimage.feature.blob_dog` multi-scale difference-of-Gaussians blob detection.
- `fast_nms_stars`: `min_dist_px=10.0`, `max_candidates=250`, pre-cap `500`, SciPy `cKDTree` suppression.
- `compute_star_annulus_snr`: local sky-annulus SNR diagnostic with `top_k=30` and `edge_margin=11`.

Preserve exact Phase 1 numerical parity with Colab Phase 1 outputs (15/15 image-detector configurations across the 5 reference sample images).

## 2. Coordinate & Indexing Conventions (`ADR-011`)

- **OpenCV Pixel Coordinates**: Origin `(0, 0)` at top-left pixel corner/center convention as implemented, `+x` rightward in `[0, W-1]`, `+y` downward in `[0, H-1]`.
- **NumPy Array Indexing**: `image[iy, ix]` (`row = y`, `col = x`). Double-check every slice, annulus crop, and `blob_dog` `(row, col)` output before storing or plotting as `(x, y)`.
- **Coordinate Scale Discipline**: Phase 1 CSV exports (`rank,x_px,y_px,flux_score`) use **working resolution (`max_dim=800`), NOT native camera resolution**. Every function or file schema must document whether coordinates live in working or native pixel space and apply exact `(scale_x, scale_y)` conversion without off-by-half-pixel distortion when mapping between frames.
- **Data Types**: Sub-pixel coordinates stored in `float32` (`stars_xyf`) in memory and `float64` in CSV exports.
- **Celestial Coordinates**: Phase 3 catalogue/astrometric projection uses J2000 / ICRS `(RA, Dec)` in decimal degrees.

## 3. Astronomical Image Processing Hazards & Numerical Safety

- **Saturated Stars & Blooming**: Saturated bright-star cores flatten intensity peaks and skew intensity-weighted centroids or DoG peak responses.
- **Extended Non-Stellar Objects**: Nebulae, Milky Way dust bands, clouds, moonlight gradients, horizon silhouettes, lens flare, hot pixels, and text/watermarks must be considered false-positive risks rather than assumed to be point stars.
- **Numerical Edge Guards**: Handle empty candidate arrays `(0, 3)`, zero contour moments (`m00 == 0`), uniform/zero-variance images (`std == 0`, `MAD == 0`), border candidates within `edge_margin`, and deterministic tie-breaking when fluxes are equal. Never emit `NaN` or `Inf` centroids.

## 4. Detection Evaluation Discipline

- `top30_snr`, `n_stars`, and wall-clock `latency_ms` are **unsupervised proxy diagnostics only** (`ADR-008`). They measure candidate count and relative brightness contrast of up to 30 top detections, NOT detection correctness.
- Precision, recall, F1, false-discovery rate, and centroid localization RMSE require independently verified ground truth and explicit one-to-one spatial matching (e.g., Hungarian / linear sum assignment within a stated pixel radius).
- Evaluate traditional CV baselines and future DL models on identical splits and identical coordinate frames.
