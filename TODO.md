# Project Spandrel: Harmonization and Enhancement Roadmap

## 1. Structural Reorganization
- [x] Create standard directory structure (`src/`, `docs/`, `data/`, `tests/`, `scripts/`).
- [x] Move source code into `src/spandrel/`.
- [x] Move documentation into `docs/`.
- [x] Move data files into `data/`.
- [x] Move images and figures into `results/figures`.
- [x] Archive obsolete files into `_archive/`.

## 2. Codebase Audit & Refactoring
- [x] Update `pyproject.toml` to reflect new structure (`src` layout).
- [x] Fix imports in all Python files to match the new package structure.
- [x] Create `requirements.md` for reproducible builds.
- [x] Ensure all scripts are executable (verified via tests).
- [x] Remove all `sys.path.insert` hacks from src/ and tests/ (12 removed).
- [x] Fix TypeError in CLI: `run_analysis()` now accepts keyword arguments.
- [x] Remove sys.argv mutation in CLI: `elevated/run_all.main()` now accepts kwargs.
- [x] Remove import-time `print()` side effects from `spandrel_cosmology_hpc.py`.
- [x] Replace global `warnings.filterwarnings('ignore')` with targeted filters (8 files).
- [x] Move OMP/MKL/OPENBLAS env var mutations from module-level into `main()`.
- [x] Guard all `plt.show()` calls with `show_or_close()` (16 call sites).
- [x] Define `__all__` in previously-empty `__init__.py` files.
- [x] Fix CI import check paths (pre-refactor paths -> current paths).

## 3. Documentation & Verification
- [x] Update `README.md` to point to new locations.
- [x] Verify `pytest` discovery works with new structure.
- [x] Run static analysis for common errors.
- [x] Update `SCOPE.md` and `DOCUMENTATION.md` with current dates/status.
- [x] Create `CHANGELOG.md` from git history.
- [x] Reconcile `requirements.txt` vs `requirements.md` version discrepancies.

## 4. Synthesis & Expansion
- [x] Harmonize module interfaces (removed `sys.path` hacks).
- [x] Address `numpy.trapz` deprecation.
- [x] Add CLI test suite (`tests/test_cli.py`).
- [x] Add elevated module smoke tests (`tests/test_elevated.py`).
- [x] Add visualization smoke tests (`tests/test_visuals.py`).
- [x] Enable pytest-cov in addopts.

## 5. Build & Packaging
- [x] Fix Source URL in `pyproject.toml`.
- [x] Pin `spandrel-core` to commit SHA for reproducibility.
- [x] Add platform marker to `mlx` (macOS only).
- [x] Narrow bare `except Exception` to `except ImportError` in `__init__.py`.

## 6. Linting & Tooling
- [x] Configure `ruff` in `pyproject.toml`.
- [x] Expand `mypy` coverage to `ddt/` module.
- [x] Create `.pre-commit-config.yaml`.

## 7. Scientific Debt (Outstanding)
- [ ] Add temperature floor guard in `reaction_carbon.py` to prevent T^(-20/3) overflow.
- [ ] Document periodic BC assumption in `flux_hllc.py` MUSCL reconstruction.
- [ ] Document fixed integration resolution in `light_curve_synthesis.py`.
- [ ] Add Sod shock tube convergence test to `tests/test_flux.py`.

## 8. Distribution (Outstanding)
- [ ] Publish to PyPI (add publish workflow).
- [ ] Add Dockerfile for containerized execution.
- [ ] Add CONTRIBUTING.md.
- [ ] Add SPDX license headers to all source files.
