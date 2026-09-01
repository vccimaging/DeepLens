# Repository Guidelines

## Python Formatting and Linting

Ruff is the canonical Python formatter and linter. Its configuration and pinned
development dependency in `pyproject.toml` are authoritative.

For Python files changed by the current task, run safe lint fixes before
formatting:

```bash
ruff check --fix path/to/changed.py test/test_changed.py
ruff format path/to/changed.py test/test_changed.py
```

Inspect the diff. Do not use `--unsafe-fixes`, and do not run repository-wide
mutating Ruff commands as part of an unrelated change.

Before handoff, run the repository-wide read-only gates:

```bash
ruff check .
ruff format --check .
```
