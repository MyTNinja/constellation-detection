# Antigravity repository workflow

For every user-approved task:

1. Confirm the task scope and acceptance criteria with the user before implementation. Work on one task at a time.
2. Start from an up-to-date `main` and create a task-specific branch (for example, `feature/<short-task>`, `fix/<short-task>`, or `docs/<short-task>`). Never commit or push directly to `main`.
3. Inspect the working tree first. Preserve unrelated user changes; do not overwrite or discard them. Keep datasets, credentials, secrets, generated outputs, and model weights out of Git.
4. Follow `AGENTS.md`. Preserve research logic unless the user explicitly approves a change. Do not fabricate labels or results, introduce data leakage, or overstate scientific conclusions.
5. Add or update relevant tests. Run `python -m pytest` and `python -m ruff check .` when available; report exact results and any limitations.
6. Commit only task-related changes with a clear message, push only the task branch, and open a pull request to `main`. Include scope, implementation notes, test evidence, and scientific-validity details using the PR template.
7. Stop after opening the PR. Do not merge, enable auto-merge, start another task, or claim approval unless the user explicitly directs that action. Wait for the user's review and next instruction.
