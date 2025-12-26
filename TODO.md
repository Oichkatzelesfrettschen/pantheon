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

## 3. Documentation & Verification
- [x] Update `README.md` to point to new locations.
- [x] Verify `pytest` discovery works with new structure (120 tests passed).
- [x] Run static analysis (visual check) for common errors. (Implicitly done during refactor)

## 4. Synthesis & Expansion
- [x] Harmonize module interfaces (Fixed sys.path hacks in `synthesis` and `elevated` modules).
- [x] Address any "TODO" or "FIXME" found in code (Fixed `numpy.trapz` deprecation).
