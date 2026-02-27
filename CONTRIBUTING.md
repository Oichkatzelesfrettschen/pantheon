# Contributing to Spandrel

Thank you for your interest in contributing. This document explains how to set
up a development environment, run the tests, and submit changes.

---

## Table of contents

1. [Development setup](#development-setup)
2. [Running the tests](#running-the-tests)
3. [Code standards](#code-standards)
4. [Submitting changes](#submitting-changes)
5. [Scientific contributions](#scientific-contributions)
6. [Reporting bugs](#reporting-bugs)

---

## Development setup

### Prerequisites

- Python 3.9 or later (3.12 recommended)
- Git
- A virtual environment tool (venv, conda, etc.)

### Steps

```bash
# Clone the repository
git clone https://github.com/Oichkatzelesfrettschen/pantheon.git
cd pantheon

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with development extras
pip install -e ".[dev]"

# Install pre-commit hooks (runs ruff and mypy on every commit)
pre-commit install
```

The `[dev]` extra installs pytest, pytest-cov, ruff, mypy, and pre-commit.

---

## Running the tests

```bash
# Full suite with coverage report
pytest tests/ -v --cov=spandrel --cov-report=term-missing

# Single module
pytest tests/test_flux.py -v

# Non-interactive (headless) mode -- used in CI
MPLBACKEND=Agg pytest tests/ -v
```

All 120+ tests must pass before a PR is merged. Coverage must not regress.

---

## Code standards

### Style and linting

We use **ruff** for linting and formatting. The configuration lives in
`pyproject.toml` under `[tool.ruff]`.

```bash
# Check for violations
ruff check src/ tests/

# Auto-fix safe violations
ruff check --fix src/ tests/
```

### Type checking

We use **mypy** in strict mode for `src/spandrel/core/` and
`src/spandrel/ddt/`. Run it with:

```bash
mypy src/spandrel/core src/spandrel/ddt
```

New modules added to these packages must include type annotations.

### Pre-commit hooks

The pre-commit configuration (`.pre-commit-config.yaml`) runs ruff, mypy,
and several file-hygiene checks automatically on `git commit`. If a hook
fails, fix the reported issue and re-stage your changes before committing.

### Documentation style

- Write comments in plain English. Avoid Unicode in source files.
- Document **WHY** before WHAT and HOW. See `CLAUDE.md` for the full
  rationale.
- Every public function or class must have a docstring explaining its
  purpose, arguments, and return values.

### Physics constants and parameters

All shared physical constants live in `src/spandrel/core/constants.py`.
Do not hard-code constants in individual modules; import from core instead.
When adding a new constant, include the literature reference in a comment.

---

## Submitting changes

### Branch naming

Use short-lived feature branches:

```
feature/<description>
fix/<description>
docs/<description>
```

### Commit messages

Follow Conventional Commits:

```
feat: add Chandrasekhar-mass EOS correction
fix: prevent T^(-3/2) overflow in screening_factor
docs: update SCOPE.md with phase 8 completion
refactor: remove sys.path hacks from synthesis modules
```

Keep messages imperative and under 72 characters for the subject line.

### Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your changes, including tests for any new behavior.
3. Run the full test suite locally: `pytest tests/ -v`.
4. Run `pre-commit run --all-files` and fix any violations.
5. Open a PR against `main` with a clear description of:
   - **WHY**: what problem this solves or risk it reduces.
   - **WHAT**: files and components affected.
   - **HOW**: commands to reproduce and verify the change.
6. Ensure CI is green before requesting review.

Reviewers will check that:
- Tests pass and coverage does not regress.
- No new `sys.path.insert` or global `warnings.filterwarnings('ignore')`.
- `plt.show()` is replaced with `show_or_close(fig)` from `spandrel.visuals.utils`.
- No secrets or personal data in the diff.

---

## Scientific contributions

### Adding a new reaction rate or EOS

1. Place the module in the appropriate package (`ddt/`, `synthesis/`, etc.).
2. Cite the paper source in the module docstring and in constant comments.
3. Add a unit test in `tests/` that validates the rate at a known reference
   temperature and density.
4. Numerical overflow risks (division by zero, exponential blowup) must be
   handled with a floor and a **WHY** comment explaining the physical justification.

### Adding a cosmological model

1. Subclass or extend `SpandrelCosmology` in `src/spandrel/cosmology/`.
2. Update `src/spandrel/cosmology/__init__.py` to export the new class.
3. Add tests that reproduce at least one published result from the model's
   reference paper.

### Adding a light-curve or spectral model

Place it in `src/spandrel/elevated/` and document any fixed integration
resolutions with a RESOLUTION NOTE comment (see `light_curve_synthesis.py`
for the pattern).

---

## Reporting bugs

Open an issue on GitHub with:
- A minimal reproducible example.
- The exact error message and traceback.
- Python version (`python --version`) and platform.
- Spandrel version (`pip show spandrel`).

For numerical or scientific discrepancies, include the reference result you
expected and the actual output, along with the input parameters.

---

## License

By contributing, you agree that your contributions will be licensed under the
same GPL-2.0-only license as the rest of the project.
