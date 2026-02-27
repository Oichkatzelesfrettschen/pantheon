# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] - 2026-02-26

### Fixed
- **CLI TypeError**: `run_analysis()` signature changed from `(args)` to keyword-only
  arguments so `cli.py` no longer crashes on `spandrel cosmology`.
- **CI broken imports**: `.github/workflows/tests.yml` import checks updated from
  pre-refactor module paths (`spandrel_cosmology`, `ddt_solver`, `constants`) to
  current paths (`spandrel.cosmology.spandrel_cosmology`, etc.).
- **sys.argv mutation**: `elevated/run_all.main()` now accepts `quick`/`full` kwargs;
  `cli.py` no longer mutates `sys.argv` to communicate flags.
- **Import-time stdout**: Removed `print()` calls at module level in
  `spandrel_cosmology_hpc.py` (CPU core count and MLX status).
- **Source URL**: `pyproject.toml` Source pointed at upstream data release; corrected
  to `https://github.com/Oichkatzelesfrettschen/pantheon`.

### Changed
- **sys.path hacks removed**: All 12 `sys.path.insert` calls removed from `src/`
  (ddt, synthesis, elevated modules) and from test files. The one intentional fallback
  in `__init__.py` is now documented.
- **Warning filters**: Replaced 8 global `warnings.filterwarnings('ignore')` calls with
  targeted `category=RuntimeWarning, module='scipy'/'numpy'` filters.
- **Environment variables**: OMP/MKL/OPENBLAS thread count env vars moved from
  module-level in `run_analysis.py` to inside `main()` to avoid import-time side effects.
- **plt.show() guarded**: All 16 bare `plt.show()` calls replaced with `show_or_close(fig)`
  from the new `spandrel.visuals.utils` module. Non-interactive backends (Agg, CI) now
  close figures instead of blocking.
- **Exception narrowing**: Bare `except Exception` in `__init__.py` narrowed to
  `except ImportError`.
- **spandrel-core pinned**: Dependency pinned to commit SHA
  `52c2c9c6c9f60ce9b5e2e42c6bbb7312a8e144cc` for reproducible installs.
- **mlx platform-guarded**: `mlx` optional dependency now carries
  `sys_platform == 'darwin'` marker.
- **Python 3.13** added to CI test matrix.
- **pytest-cov** enabled in `addopts`.

### Added
- `src/spandrel/visuals/utils.py`: `show_or_close(fig)` utility for non-blocking plot display.
- `tests/test_cli.py`: CLI routing tests (mocked, fast).
- `tests/test_elevated.py`: Elevated module smoke tests.
- `tests/test_visuals.py`: Visualization utility tests (Agg backend).
- `__all__` exports defined in previously-empty `__init__.py` files
  (`core/`, `cosmology/`, `analysis/`, `synthesis/`, `visuals/`).
- `[tool.ruff]` configuration in `pyproject.toml`.
- `.pre-commit-config.yaml` with ruff, mypy, and pre-commit-hooks.
- `CHANGELOG.md` (this file).

---

## [1.0.0] - 2025-12-14

### Added
- `src/` layout with `spandrel` package.
- Centralized constants via `spandrel-core` git dependency.
- `PantheonData` data interface delegated to `spandrel-core`.
- 120 unit and integration tests.

### Changed
- All modules import from `spandrel.core.constants` instead of local copies.

---

## [0.2.0] - 2025-11-xx

### Added
- Scientific paper with pgfplots/tikz graphics.

---

## [0.1.0] - 2025-11-xx (503d6c4)

### Added
- Initial commit: Pantheon+SH0ES cosmology analysis framework.
- DDT solver (HLLC Riemann, EOS, reaction network).
- Cosmology pipeline (MLE, MCMC, nested sampling).
- Synthesis modules (turbulent flame theory, Phillips relation).
- Elevated modules (alpha-chain network, light curve, model comparison).
