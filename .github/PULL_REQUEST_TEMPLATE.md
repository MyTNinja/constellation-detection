## Scope

<!-- What problem does this PR solve? What is included and intentionally out of scope? -->

## Implementation notes

<!-- Summarize the approach, relevant files, assumptions, and any migration or compatibility impact. -->

## Test evidence

- [ ] Tests added or updated where behavior changed.
- [ ] `python -m pytest` — paste result or explain why it could not be run.
- [ ] `python -m ruff check .` — paste result or explain why it could not be run.
- [ ] CI checks pass, or failures are explained below.

Evidence / limitations:

## Research and scientific validity

- [ ] No ground-truth labels, coordinates, or measurements were fabricated.
- [ ] Predictions remain separate from ground-truth annotations.
- [ ] Train, validation, and test data remain separated; no evaluation data leakage was introduced.
- [ ] Any thresholds, units, coordinate conventions, and assumptions are documented.
- [ ] Randomness is seeded and experiment configuration is recorded where relevant.
- [ ] Claims are supported by appropriate data or citations; limitations are stated.
- [ ] Datasets, secrets, credentials, and model weights are not included.

Notes:

## Reviewer checklist

- [ ] Change matches the stated scope and preserves existing research logic unless explicitly approved.
- [ ] Tests and edge cases cover the changed behavior.
- [ ] API, packaging, and portability conventions are respected.
- [ ] Results and scientific claims are reproducible and appropriately qualified.
- [ ] No unrelated files or generated artifacts are included.
- [ ] Human reviewer approves before merge; no automatic merge or next-task progression.
