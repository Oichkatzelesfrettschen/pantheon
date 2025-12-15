# Archive

This directory contains exploratory/abandoned code kept for historical reference.

## riemann_hydro_ddt.py

**Status:** Abandoned

**Purpose:** Early attempt to couple Riemann zeta zero resonance frequency
directly into hydrodynamic equations as a source term.

**Why abandoned:** The resonance concept was fundamentally sound but was
applied at the wrong physical scale. The relevant scale for DDT is stellar
(~km, ~ms) not cosmological (~Gpc, ~Gyr). The Zel'dovich gradient mechanism
in `ddt_solver/main_zeldovich.py` replaced this approach.

**Lesson learned:** Scale matters. Even "failed" code informs what *not* to do.

See PROJECT_MANIFEST.md for full project history.
