# Repository Guidelines

## Key Files

- `agent/architecture.md` — code structure, module layout, design patterns
- `agent/goal.md` — project goals and non-goals
- `agent/findings.md` — hard-won insights, gotchas, workarounds
- `agent/todo.md` — current task backlog

## Before Any Task

Read `agent/architecture.md` to understand the codebase structure.

## Project Entry Points

- `deeplens/` — core library package
- `deeplens/lens.py` — base lens interface
- `deeplens/geolens.py` — geometric lens (ray tracing)
- `deeplens/diffraclens.py` — diffractive lens (wave optics)
- `deeplens/hybridlens.py` — hybrid ray-wave lens
- `deeplens/psfnetlens.py` — neural PSF surrogate lens
- `test/` — test suite
- `configs/` — plain YAML settings for the numbered example scripts, loaded
  with `yaml.safe_load`

## Environment

Activate the project environment before running anything (`conda activate
deeplens`, or the equivalent venv). A bare system Python will not have torch,
and `ruff` ships only in the `dev` extra.

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

## Docstrings

Google style, rendered by mkdocstrings, so use `$$` for math and fenced
` ```python ` blocks rather than RST markup.

Every module opens with a docstring, placed after the copyright block. A header
that lists a module's functions is a contract: when you add, rename, or remove
one, update the list in the same change. Do not describe behavior the file does
not implement.

## Workflow

- Plans go in `agent/plan/plan_<feature>.md`
- Research notes go in `agent/research/`
- Log non-obvious discoveries in `agent/findings.md`

## Commands

```bash
# Install
pip install -e .

# Install with linting and notebook tooling
pip install -e ".[dev]"

# Run tests
pytest test/
```

Documentation lives in a separate repository; there is no `mkdocs.yml` here.
