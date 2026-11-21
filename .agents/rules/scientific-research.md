---
trigger: model_decision
description: Academic research standards, source verification, citation integrity, uncertainty recording, and scientific claim discipline. Activate in Research Mode, during literature reviews, or when evaluating papers, catalogues, datasets, or scientific claims.
---

# Scientific Research and Citation Standards

## 1. Source Priority Hierarchy

When project documents conflict, resolve authority in this exact order and explicitly document any discrepancy rather than erasing history:
1. Actual implemented source code (`src/constellation_detection/`) and verified test results (`tests/`).
2. Authoritative context documents (`docs/context/ARCHITECTURE.md`, `docs/context/DECISIONS.md`).
3. Research index (`research/INDEX.md`) and README/packaging metadata (`README.md`, `pyproject.toml`).
4. Legacy project briefs and historical notebooks (external briefs and notebooks in `d:\potential captone\`, `combined-deep-learning-notebook.ipynb`).

## 2. Source Provenance Classification (`research/INDEX.md`)

Always keep these three evidence tiers strictly separate:
- **Project-Present Materials**: Code, briefs, and local notebooks directly inspected in the workspace.
- **Actually Inspected Primary Literature & Documentation**: Peer-reviewed papers, preprints, star-catalogue specs, or official library docs directly read and verified during research.
- **Unreviewed Recommended Leads**: Relevant papers, catalogues, or datasets identified by search or bibliographic reference whose full text or schema has **not yet been inspected**.

**Rule**: Never label or imply that an unreviewed recommended paper, dataset, or benchmark was read or verified.

## 3. Citation & Technical Claim Integrity

- Every literature citation in a research specification must include verified authors/organization, year, title, venue or archive identifier (DOI, arXiv ID, ADS bibcode, or canonical URL).
- Do not fabricate citations, benchmark numbers, sample counts, star magnitudes, or runtime figures from memory.
- Never claim star-detection accuracy (precision, recall, F1, false-discovery rate, localization RMSE) or constellation-recognition accuracy unless measured on independently verified ground truth.
- Reject any proposal or method that relies on device sensors (GPS, compass, gyroscope, IMU, orientation metadata) per `ADR-001`.

## 4. Epistemic Clarity & Uncertainty Recording

Research specifications must explicitly partition content into five labeled sections:
1. **Verified Facts**: Directly observed in repository files, tool output, or inspected primary literature.
2. **Existing Implementation**: Current behavior and exact parameters in `src/constellation_detection/`.
3. **Proposed Design**: Recommended candidate approaches with explicit trade-off comparison.
4. **Assumptions**: Unproven premises required for the design to hold.
5. **Open Uncertainties & Risks**: Edge cases, missing ground truth, compute risks, licensing ambiguity, or open `UD-*` items requiring resolution.
