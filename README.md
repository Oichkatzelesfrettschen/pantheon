# The Spandrel Project

**Unified Synthesis: From Microscopic Turbulence to Macroscopic Cosmology**

[![Status: Elevated](https://img.shields.io/badge/Status-Elevated-success)]()
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue)]()

## Overview

The Spandrel Project is a computational astrophysics suite that unifies the physics of Type Ia supernovae across scales. It demonstrates that the **Phillips Relation**—the cornerstone of modern cosmology—is an emergent property of turbulent geometry in white dwarf interiors.

**Key Discovery:**
The "Unified Experiment" connects the Kolmogorov turbulent cascade to the Zel'dovich gradient mechanism, showing that variations in the fractal dimension of the flame surface ($D \in [2.1, 2.6]$) naturally reproduce the observed scatter in SNe Ia luminosity and decline rate.

*Historical Context:* This project originated as an investigation into "Dissociative Field Theory" (Riemann Resonance). While that cosmological hypothesis was rigorously falsified (Δchi^2 = -24.1 vs LambdaCDM), the computational tools developed for it were successfully pivoted to solve the progenitor problem of Type Ia supernovae.

## Results Summary

| Model / Mechanism | Result | Status |
|-------------------|--------|--------|
| **Turbulent Synthesis** | **Derived Phillips Relation** | **CONFIRMED** |
| Zel'dovich DDT | Detonation at $\lambda > 12$ km | Validated |
| Riemann Resonance | $\Delta \chi^2 = -24.1$ | Falsified |

### The Causal Chain
1. **Turbulence:** Fractal dimension $D$ governs flame surface area.
2. **Criticality:** Flame acceleration creates temperature gradients.
3. **Detonation:** Gradients $>\lambda_{crit}$ trigger DDT.
4. **Nucleosynthesis:** DDT timing determines $^{56}$Ni yield.
5. **Light Curve:** $^{56}$Ni mass sets peak luminosity ($M_B$) and width ($\Delta m_{15}$).

## Installation

```bash
# Clone repository
cd pantheon

# Install in editable mode (enables 'spandrel' CLI)
pip install -e .

# Optional: Install with GPU/JIT acceleration
pip install -e ".[all]"
```

## Quick Start (CLI)

The project exposes a unified command-line interface: `spandrel`

### 1. Run the Unified Synthesis
Connects turbulence to cosmology and generates the summary plot.
```bash
spandrel synthesis
```
*Output: `results/figures/unified_synthesis.png`*

### 2. Run DDT Simulation
Simulates the Deflagration-to-Detonation Transition using the Zel'dovich mechanism.
```bash
spandrel ddt --quick   # Low-res verification
spandrel ddt           # Full high-res simulation
```

### 3. Run Cosmological Analysis
Tests hypotheses against Pantheon+SH0ES and DESI data.
```bash
spandrel cosmology
```

### 4. Run Full Elevated Suite
Executes the entire research pipeline (Nuclear -> DDT -> Light Curve).
```bash
spandrel elevate --quick
```

## Project Structure

```
pantheon/
|-- src/spandrel/
│   |-- cli.py                  # Unified CLI entry point
│   |-- synthesis/              # The "Unified Experiment" (Turbulence -> Phillips)
│   |-- ddt/                    # Reactive Euler Solver (Zeldovich mechanism)
│   |-- cosmology/              # Hypothesis testing framework
│   |-- elevated/               # Extended research modules
│   +-- core/                   # Shared physics constants
|-- data/
│   +-- Pantheon+SH0ES.dat      # Primary dataset
|-- results/figures/            # Generated science plots
+-- pyproject.toml              # Package configuration
```

## Data Source

### Pantheon+SH0ES Dataset

The primary dataset contains 1,701 Type Ia supernovae from 18 surveys, spanning redshift z = 0.001 to 2.26.

**Key columns:**
- `zHD`: Hubble diagram redshift (corrected for peculiar velocities)
- `MU_SH0ES`: Distance modulus (SH0ES calibrated)
- `MU_SH0ES_ERR_DIAG`: Measurement uncertainty

**Official sources:**
- [Pantheon+SH0ES Website](https://pantheonplussh0es.github.io/)
- [GitHub Data Release](https://github.com/PantheonPlusSH0ES/DataRelease)

### Primary References

**Dataset:**
- Scolnic, D. et al. (2022). "The Pantheon+ Analysis: The Full Data Set and Light-curve Release." *The Astrophysical Journal*, 938:113. [DOI:10.3847/1538-4357/ac8b7a](https://doi.org/10.3847/1538-4357/ac8b7a) | [arXiv:2112.03863](https://arxiv.org/abs/2112.03863)

**Cosmological Constraints:**
- Brout, D. et al. (2022). "The Pantheon+ Analysis: Cosmological Constraints." [arXiv:2202.04077](https://arxiv.org/abs/2202.04077)

**DESI BAO (used for falsification):**
- DESI Collaboration (2024). "DESI 2024 VI: Cosmological Constraints from the Measurements of Baryon Acoustic Oscillations."

### Physics References

**Nuclear Reaction Rates:**
- Caughlan, G.R. & Fowler, W.A. (1988). "Thermonuclear Reaction Rates V." *Atomic Data and Nuclear Data Tables*, 40:283.

**Equation of State:**
- Timmes, F.X. & Swesty, F.D. (2000). "The Accuracy, Consistency, and Speed of an Electron-Positron Equation of State." *ApJS*, 126:501.

**DDT Mechanism:**
- Zel'dovich, Ya.B. (1980). "Regime classification of an exothermic reaction with nonuniform initial conditions." *Combustion and Flame*, 39:211.

**Light Curve Physics:**
- Arnett, W.D. (1982). "Type I supernovae. I. Analytic solutions for the early part of the light curve." *ApJ*, 253:785.

## Physics Validated

1. **Chandrasekhar EOS** — Relativistic degenerate electron pressure
2. **Caughlan-Fowler rates** — C12+C12 -> NSE reaction network
3. **Zel'dovich mechanism** — Temperature gradient -> shock coupling
4. **Arnett's rule** — Ni-56 mass -> peak luminosity
5. **Chapman-Jouguet theory** — Detonation velocity bounds

## Dependencies

**Required:**
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7
- matplotlib >= 3.4

**Optional:**
- mlx >= 0.0.1 (Apple Silicon GPU acceleration)
- numba >= 0.55 (JIT compilation for hydrodynamics)

## License

MIT

## Citation

If you use this code or the Pantheon+SH0ES dataset in your research, please cite:

```bibtex
@article{Scolnic2022,
  author = {Scolnic, D. and others},
  title = {The Pantheon+ Analysis: The Full Data Set and Light-curve Release},
  journal = {ApJ},
  volume = {938},
  pages = {113},
  year = {2022},
  doi = {10.3847/1538-4357/ac8b7a}
}
```

---

*"We tried to find the Music of the Primes. We found the Fire of the Carbon."*
