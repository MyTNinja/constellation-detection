---
name: literature-review
description: >-
  Conducts structured academic and technical literature reviews for star detection,
  astrometric plate solving, constellation recognition, and hybrid CV-DL methods in
  Research Mode, producing a cited research specification without modifying code.
---

# Literature Review Skill

## Trigger

Use in **Research Mode** when evaluating Phase 2+ deep-learning star detectors, Phase 3 vision-based constellation recognition / geometric astrometry, astronomical star catalogues, evaluation benchmarks, or unresolved design questions (`UD-001` through `UD-008`).

## Guardrails

- Strictly read-only on implementation code (`src/`, `tests/`, `scripts/`, `notebooks/`, `pyproject.toml`, `requirements-colab.txt`).
- Follow `.agents/rules/scientific-research.md`: never cite uninspected papers as reviewed, never fabricate bibliographic metadata or metrics, and never consider sensor-based shortcuts (`ADR-001`).

## Required Inputs

- Research topic or question (e.g., Phase 2 detector family, Phase 3 polygon/triangle hashing vs. plate solving, dataset provenance).
- Current scope/constraints from the user (optional if standard project constraints apply).

## Procedure

1. **Ground in Repository State**: Read relevant sections of `docs/context/ARCHITECTURE.md`, `docs/context/DECISIONS.md`, and `research/INDEX.md`, plus existing baseline code if applicable.
2. **Search Primary Sources**: Run targeted `search_web` queries for peer-reviewed papers (CVPR/ICCV/PASP/MNRAS/A&A/AJ), arXiv preprints, official catalogue specifications (Hipparcos, Gaia, Yale BSC), and official PyTorch/OpenCV/astrometry documentation.
3. **Verify Every Citation**: Open/confirm titles, authors/institutions, publication year, venue, and stable identifiers (DOI, arXiv ID, ADS bibcode, or canonical URL). Place uninspected leads in an explicit *Unreviewed Leads* bucket.
4. **Compare Alternatives**: Evaluate at least two candidate approaches side-by-side against project constraints:
   - Vision-only input (no sensors, `ADR-001`)
   - PyTorch compatibility (`ADR-003`)
   - Local 4-core / 8 GiB RAM CPU inference + Colab GPU training
   - Annotation/ground-truth requirements (`ADR-008`)
   - Robustness to bright/saturated stars, nebulae, clouds, light pollution, and wide-field distortion
5. **Surface Uncertainties & Open Decisions**: Map gaps to existing `UD-*` entries or propose new ones.
6. **Draft Cited Research Specification**: Do NOT edit implementation files or overwrite `research/INDEX.md` unless explicitly approved; provide any proposed `research/INDEX.md` additions as a patch preview.

## Expected Output

A structured **Cited Research Specification** containing:
- **1. Research Objective & Scope**
- **2. Baseline Context (Verified Repo State)**
- **3. Inspected Primary Sources** (verified metadata & key findings)
- **4. Unreviewed Leads Requiring Inspection** (explicitly separated)
- **5. Comparative Trade-Off Matrix** (constraints, pros, cons, CPU/RAM cost, GT needs)
- **6. Recommended Direction & Open Uncertainties**
- **7. Next Approval Gate** (what the user must approve before `experiment-design` or Implementation Mode)
