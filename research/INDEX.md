# Research Reference Index (`research/INDEX.md`)

> **Last Updated:** 2026-11-21  
> **Integrity Policy:** This index strictly distinguishes artifacts **actually present and inspected** from external papers, documentation, and catalogues **recommended for future review (`NOT YET REVIEWED` / `NOT YET INGESTED`)**. No recommended paper or external API manual is claimed as previously reviewed.

---

## 1. Sources Already Present in the Project

### 1.1 Tracked Repository Sources (`d:\projects\constellation-detection`)
| Source | Type | Verified Role in Project |
|---|---|---|
| [README.md](../README.md) | Repository overview | Documents Phase 1 modular package, setup, Colab deferral, research rules |
| [AGENTS.md](../AGENTS.md) | Policy specification | Enforces research integrity, anti-leakage, PyTorch-only mandate, CPU compatibility, local Git workflow |
| [docs/context/ARCHITECTURE.md](../docs/context/ARCHITECTURE.md) | Architecture KB | Authoritative technical system specification (implemented vs. proposed) |
| [docs/context/DECISIONS.md](../docs/context/DECISIONS.md) | Decision register | Traceable ADRs (`ADR-001`–`ADR-016`) and unresolved approvals (`UD-001`–`UD-008`) |
| [pyproject.toml](../pyproject.toml) | Build & dependency manifest | Core stack (`numpy`, `pandas`, `opencv-python`, `scipy`, `scikit-image`, `matplotlib`), `[dev]`, `[dl]` |
| [requirements-colab.txt](../requirements-colab.txt) | Colab install file | Editable install `-e .` preserving Colab GPU/CUDA PyTorch |
| [src/constellation_detection/](../src/constellation_detection/__init__.py) | Phase 1 Python package | Modular `config`, `image`, `detection` (`nms`, `threshold`, `tophat`, `dog`), `evaluation`, `visualization` |
| [notebooks/phase1_traditional_cv_demo.ipynb](../notebooks/phase1_traditional_cv_demo.ipynb) | Demo notebook | 4-cell executed Phase 1 benchmark (`36.31 / 80.39 / 314.52 ms` local CPU) |
| [scripts/download_phase1_samples.py](../scripts/download_phase1_samples.py) | Utility script | Atomic downloader for the 5 ESA/Hubble astrophotography samples |

### 1.2 External Pre-Migration / Legacy Sources (`d:\potential captone\`, Untracked by Git)
> **Note:** Stored outside the active Git working tree and verified read-only during Task 01.

| External File (`d:\potential captone\`) | Type | Verified Role & Notes |
|---|---|---|
| `project-brief_v2.md` | Latest project brief (`326` lines) | Phase 0 & 1 complete; locks PyTorch (`ADR-003`); plans Phases 2–5 *(contains CLAHE/dependency/disk discrepancies noted in Architecture KB)* |
| `project-brief.md` | Historical brief v1 (`241` lines) | Superseded planning baseline that proposed TF/Keras vs. PyTorch comparison |
| `01_Project_Initialization_and_CV_Detectors.ipynb` | Phase 1 Colab baseline (`12.4 MB`, 5 code cells) | Original Colab CV1/CV2/CV3 implementation (`/content/constellation_vision`); explicitly rejects CLAHE (L163) |
| `notebooks/reference/constellation-extraction.ipynb` | Initial Sem V mini project (`1.08 MB`, 13 cells) | Fixed-threshold (`127`) Cassiopeia/Ursa Minor prototype (`max_distance=225/140`); depends on missing local JPEGs |
| `combined-deep-learning-notebook.ipynb` | Consolidated coursework (`6,594` lines, 9 labs) | Educational TensorFlow/Keras labs only (`ADR-013`); unsuitable for direct copying due to framework lock-in, bugs, `VGG16` CPU weight, and data leakage |

### 1.3 Sample Astrophotography URLs Currently Used (`scripts/download_phase1_samples.py`)
All five are ground-based wide-field photographs by **Akira Fujii** hosted on ESA/Hubble (`photoshop:Credit = "A. Fujii" / "Akira Fujii"`, `photoshop:Source = "ESA/Hubble"`, verified via Task 01 XMP inspection).

| Archive ID | Native Dims (`W×H`) | ESA/Hubble Direct Image URL | Copyright / Attribution Caution |
|---|---|---|---|
| `opo0613c` | `770×631` | `https://cdn.esahubble.org/archives/images/large/opo0613c.jpg` | Third-party copyrighted astrophotography licensed to ESA/Hubble; **not** verified CC-BY/PD (`[DISCREPANCY]` with downloader docstring) |
| `opo0706b` | `1293×1623` | `https://cdn.esahubble.org/archives/images/large/opo0706b.jpg` | Same as above |
| `heic0401g` | `2812×2778` | `https://cdn.esahubble.org/archives/images/large/heic0401g.jpg` | Same as above; depicts both Ursa Minor & Ursa Major |
| `opo0205b` | `1598×2000` | `https://cdn.esahubble.org/archives/images/large/opo0205b.jpg` | Same as above; contains burned-in cyan `"Photo credit: Akira Fujii"` watermark |
| `heic0720c` | `4863×3353` | `https://cdn.esahubble.org/archives/images/large/heic0720c.jpg` | Same as above; wide-field Summer Triangle (Cygnus + Lyra + Aquila) |

---

## 2. Papers Actually Inspected

> **Status: None (`0` peer-reviewed papers formally inspected to date).**

Although `project-brief_v2.md` marks *Phase 0 – Research Preparation ("Review relevant literature")* as `Completed`, neither the repository nor `d:\potential captone\` contains any paper notes, literature review summaries, BibTeX entries, or annotated citations. Therefore, **zero external academic papers are claimed as inspected** in this index.

When papers are reviewed in future work, record them here using the table schema below:

| Citation Key | Full Citation | Stable URL / DOI | Date Inspected | Key Takeaways Applicable to Project |
|---|---|---|---|---|
| *(empty)* | *(Log only papers whose full text has been read and summarized)* | — | — | — |

---

## 3. Official Documentation

### 3.1 Sources Actually Inspected During Audit
- **In-Repository Code, Docstrings & Local Package Metadata:** Directly inspected [pyproject.toml](../pyproject.toml), [requirements-colab.txt](../requirements-colab.txt), all modules in `src/constellation_detection/`, and installed `.venv` package versions (`numpy 2.5.4`, `opencv-python 5.0.0.93`, `pandas 3.1.1`, `scipy 1.18.1`, `scikit-image 0.26.0`, `matplotlib 3.11.2`, `pytest 9.2.0`, `ruff 0.17.5`).
- **ESA/Hubble Akira Fujii Licensing Check:** Verified via online search during Task 02 documentation prep that Akira Fujii images hosted on ESA/Hubble / David Malin Images are copyrighted rather than generic public-domain space-telescope releases (full legal terms at `https://esahubble.org/copyright/` should still be inspected directly before any public dataset release).

### 3.2 Recommended Official Documentation (`NOT YET FORMALLY INSPECTED`)
> **Status:** Canonical documentation URLs listed for upcoming Phase 2–5 engineering; full API manuals have **not** been formally reviewed in this repository yet.

| Component | Phase | Official Stable Documentation URL | Status |
|---|---|---|---|
| **PyTorch** | Phase 2–5 | https://pytorch.org/docs/stable/ | `NOT YET INSPECTED` |
| **TorchVision** | Phase 2 | https://pytorch.org/vision/stable/ | `NOT YET INSPECTED` |
| **OpenCV** | Phase 1+ | https://docs.opencv.org/ | `NOT YET INSPECTED` (only CV calls used in `src/` audited) |
| **scikit-image** (`feature.blob_dog`) | Phase 1+ | https://scikit-image.org/docs/stable/ | `NOT YET INSPECTED` |
| **SciPy** (`spatial.cKDTree`, `optimize.linear_sum_assignment`) | Phase 1–3 | https://docs.scipy.org/doc/scipy/ | `NOT YET INSPECTED` |
| **NumPy** | All | https://numpy.org/doc/stable/ | `NOT YET INSPECTED` |
| **Pandas** | Phase 1–4 | https://pandas.pydata.org/docs/ | `NOT YET INSPECTED` |
| **Matplotlib** | All | https://matplotlib.org/stable/ | `NOT YET INSPECTED` |
| **Astropy** (WCS / SkyCoord, candidate for GT prep & Phase 3) | Phase 2–3 | https://docs.astropy.org/ | `NOT YET INSPECTED` |
| **Photutils** (astronomical source detection/photometry reference) | Evaluation | https://photutils.readthedocs.io/ | `NOT YET INSPECTED` |
| **Astrometry.net** (blind WCS plate solver) | Phase 2–3 | https://github.com/dstndstn/astrometry.net | `NOT YET INSPECTED` |
| **ESA/Hubble Copyright Policy** | Compliance | https://esahubble.org/copyright/ | `NOT YET INSPECTED` in full |
| **Streamlit** (Phase 5 UI candidate) | Phase 5 | https://docs.streamlit.io/ | `NOT YET INSPECTED` |
| **ONNX Runtime** (Phase 5 optional optimization) | Phase 5 | https://onnxruntime.ai/docs/ | `NOT YET INSPECTED` |

---

## 4. Recommended Papers (`NOT YET REVIEWED`)

> **CRITICAL NOTICE:** None of the papers below have been read or reviewed in the repository yet. They are curated strictly as recommended literature directly aligned with **RQ1–RQ5** and the technical gaps identified in Task 01.

### 4.1 Traditional Astronomical Source Extraction & Scale-Space Blob Detection (`RQ1`, Phase 1–4)
| Title & Authors | Venue / Year | Persistent Identifier / Stable URL | Review Status | Why Relevant |
|---|---|---|---|---|
| **SExtractor: Software for source extraction** — E. Bertin & S. Arnouts | *A&A Suppl. Ser.*, 117, 393–404 (1996) | https://doi.org/10.1051/aas:1996164 | `Recommended — NOT YET REVIEWED` | Mesh-based local sky background estimation, deblending, and star/galaxy separation (`RQ1`, extended object FPs) |
| **DAOPHOT: A computer program for crowded-field stellar photometry** — P. B. Stetson | *PASP*, 99, 191–222 (1987) | https://doi.org/10.1086/131977 | `Recommended — NOT YET REVIEWED` | Crowded-field PSF modeling and blended star separation (relevant to Milky Way fields & Mizar/Alcor) |
| **Feature Detection with Automatic Scale Selection** — T. Lindeberg | *IJCV*, 30(2), 79–116 (1998) | https://doi.org/10.1023/A:1008045108935 | `Recommended — NOT YET REVIEWED` | Mathematical foundation of normalized DoG/LoG multi-scale blob detection (`CV3_MultiScaleDoG`) |
| **Distinctive Image Features from Scale-Invariant Keypoints** — D. G. Lowe | *IJCV*, 60(2), 91–110 (2004) | https://doi.org/10.1023/B:VISI.0000029664.99615.94 | `Recommended — NOT YET REVIEWED` | Difference-of-Gaussians sub-pixel localization and scale-invariant keypoint neighborhoods |

### 4.2 Deep Learning Point-Source Detection, Segmentation & Keypoints (`RQ2`, Phase 2)
| Title & Authors | Venue / Year | Persistent Identifier / Stable URL | Review Status | Why Relevant |
|---|---|---|---|---|
| **U-Net: Convolutional Networks for Biomedical Image Segmentation** — O. Ronneberger, P. Fischer, T. Brox | *MICCAI* (2015) | https://arxiv.org/abs/1505.04597 | `Recommended — NOT YET REVIEWED` | Referenced in `project-brief_v2.md` §4 for encoder-decoder segmentation with skip connections |
| **Objects as Points** — X. Zhou, D. Wang, P. Krähenbühl | *arXiv* (2019) | https://arxiv.org/abs/1904.07850 | `Recommended — NOT YET REVIEWED` | Anchor-free CenterNet keypoint heatmap + local offset paradigm suited for point-source stars (`project-brief_v2.md` §4) |
| **DEEPSOURCE: point source detection using deep learning** — A. Vafaei Sadr et al. | *MNRAS*, 484(2), 2793–2806 (2019) | https://doi.org/10.1093/mnras/stz131 \| https://arxiv.org/abs/1807.02701 | `Recommended — NOT YET REVIEWED` | CNN point-source detection directly in noisy astronomical images without relying on natural-image anchor boxes |
| **Deblending and classifying astronomical sources with Mask R-CNN deep learning** — C. J. Burke et al. | *MNRAS*, 490(3), 3952–3965 (2019) | https://doi.org/10.1093/mnras/stz2845 \| https://arxiv.org/abs/1908.02748 | `Recommended — NOT YET REVIEWED` | Distinguishing stars vs. extended galaxies and resolving blended astronomical sources |

### 4.3 Blind Astrometry, Geometric Hashing & Constellation Recognition (`RQ3–RQ4`, Phases 3–4)
| Title & Authors | Venue / Year | Persistent Identifier / Stable URL | Review Status | Why Relevant |
|---|---|---|---|---|
| **Astrometry.net: Blind astrometric calibration of arbitrary astronomical images** — D. Lang, D. W. Hogg, K. Mierle, M. Blanton, S. Roweis | *The Astronomical Journal*, 139(5), 1782–1800 (2010) | https://doi.org/10.1088/0004-6256/139/5/1782 \| https://arxiv.org/abs/0910.2233 | `Recommended — NOT YET REVIEWED` | Blind scale/rotation-invariant quad geometric hashing & Bayesian verification against sky catalogues (both for Phase 2 WCS GT generation and Phase 3 recognition inspiration) |
| **The Pyramid Star Identification Technique** — D. Mortari, M. A. Samaan, C. Bruccoleri, J. L. Junkins | *Navigation*, 51(3), 171–183 (2004) | https://doi.org/10.1002/j.2161-4296.2004.tb00349.x | `Recommended — NOT YET REVIEWED` | Noise-robust polygon/pyramid geometric star matching tolerant to false stars |
| **A Survey on Star Identification Algorithms** — B. B. Spratling IV & D. Mortari | *Algorithms*, 2(1), 93–107 (2009) | https://doi.org/10.3390/a2010093 | `Recommended — NOT YET REVIEWED` | Comparative taxonomy of triangle, grid, and subgraph isomorphism star-pattern matchers (`RQ3`) |

### 4.4 Foundational Star Catalogue Publications (`NOT YET REVIEWED`)
| Title & Authors | Venue / Year | Persistent Identifier / Stable URL | Review Status | Why Relevant |
|---|---|---|---|---|
| **The HIPPARCOS Catalogue** — M. A. C. Perryman et al. | *A&A*, 323, L49–L52 (1997) | https://ui.adsabs.harvard.edu/abs/1997A%26A...323L..49P/abstract | `Recommended — NOT YET REVIEWED` | Authoritative bright-star astrometric & V-magnitude catalogue (Naked-eye + DSLR depth) |
| **Gaia Data Release 3: Summary of the content and survey properties** — Gaia Collaboration (A. Vallenari et al.) | *A&A*, 674, A1 (2023) | https://doi.org/10.1051/0004-6361/202243940 | `Recommended — NOT YET REVIEWED` | High-precision reference catalogue for deeper DSLR/telephoto WCS ground-truth projection |

---

## 5. Recommended Datasets, Catalogues & Calibration Tools (`NOT YET INGESTED`)

| Resource | Category | Access URL / Identifier | Status | Intended Use & Constraints |
|---|---|---|---|---|
| **Yale Bright Star Catalogue, 5th Revised Ed. (`V/50`)** | Star catalogue (~9,110 naked-eye stars) | https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=V/50 | `Recommended — NOT YET INGESTED` | Compact bright-star table ideal for smartphone & wide-field Phase 3 constellation matching (`Vmag <= 6.5`) |
| **Hipparcos and Tycho Catalogues (`I/239`)** | Astrometric star catalogue | https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=I/239 | `Recommended — NOT YET INGESTED` | Intermediate magnitude coverage (`Vmag ~ 7–11`) for astrophotography WCS matching |
| **ESA Gaia Archive (`Gaia DR3`)** | Deep astrometric archive | https://gea.esac.esa.int/archive/ | `Recommended — NOT YET INGESTED` | Query bright/medium subsets (`Gmag <= 12`) for telephoto/deep-sky ground truth |
| **IAU Official Constellations** | Constellation topology & boundaries | https://www.iau.org/public/themes/constellations/ | `Recommended — NOT YET INGESTED` | Authoritative 88 IAU constellation names, abbreviations, boundaries, and sky charts |
| **Astrometry.net (`Nova` / local solver)** | Blind WCS calibration service | https://nova.astrometry.net/ \| https://github.com/dstndstn/astrometry.net | `Recommended — NOT YET INGESTED` | Generates verifiable WCS fits (`RA/Dec <-> x/y`) for uncalibrated night-sky photos (`UD-003`) |
| **Openly Licensed Astrophotography & Smartphone Night-Sky Corpus** | Raw image corpus *(to be curated)* | Must verify explicit `CC0` / `CC-BY` / `NASA-PD` licensing per image | `MISSING — CRITICAL PREREQUISITE` | Phase 2–4 train/val/test sets covering DSLR + smartphone photos (`RQ5`), partitioned strictly by sky field |

---

## 6. Unresolved Research Topics

The following open scientific and engineering questions bridge Phase 1 to Phases 2–5 and map directly to `UD-003`–`UD-008` in [DECISIONS.md](../docs/context/DECISIONS.md):

1. **Verifiable Ground-Truth Protocol & Limiting Magnitude (`UD-003`):** Establishing how sky-dependent limiting magnitudes, saturated guide stars, and blended pairs (`Mizar + Alcor`) are labeled cleanly via blind WCS projection + human QA without circular reliance on Phase 1 heuristics.
2. **Saturated Guide-Star Recovery vs. Background Rejection:** Resolving why Top-Hat hollows bright anchor stars (`Betelgeuse`, `Rigel`) and why `global_threshold` rejects `area > 350`, while rejecting galaxies, nebulae, and text watermarks (`15–22` Orion watermark FPs).
3. **FOV- and Scale-Invariant Suppression (`fast_nms_stars`):** Replacing or adapting fixed `10.0 px` NMS radii and the `500`-candidate pre-NMS truncation cap across `1.0×–6.1×+` downscaled resolutions.
4. **Domain Shift Between Astrophotography and Smartphone Captures (`RQ5`):** Modeling hot-pixel bursts, JPEG compression blocks, high-ISO color noise, lens vignetting, wide-angle optical distortion, light pollution gradients, and terrestrial horizons absent from Hubble outreach JPEGs.
5. **CPU-Feasible PyTorch Formulation (`RQ2`):** Selecting between lightweight U-Net/CenterNet heatmap localization vs. hybrid CV-candidate patch classification under `~1–2 GB` RAM headroom and `Core i3` CPU budgets (`UD-004`).
6. **Projective Distortion & Partial/Multi-Constellation Matching (`RQ3–RQ4`):** Designing geometric/graph matching invariant to wide-field perspective distortion, partial constellation cropping, heavy clutter/false stars, and multi-constellation overlap (`UD-007`).
7. **Deterministic Tie-Breaking & Statistical Benchmarking:** Standardizing sub-pixel precision/recall/localization-error matching (`tau` sensitivity curves) and multi-run CPU latency protocols across all future detector ablations.
