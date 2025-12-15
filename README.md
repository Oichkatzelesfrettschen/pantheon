# The Spandrel Project

**Cosmological hypothesis testing and Type Ia supernova DDT simulation**

[![Status: Complete](https://img.shields.io/badge/Status-Complete-success)]()
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue)]()

## Overview

This project began as an investigation into "Dissociative Field Theory" — a speculative framework proposing that dark energy oscillates at the frequency of the first Riemann zeta zero (γ₁ = 14.134725). Through rigorous testing against Pantheon+SH0ES supernova data and DESI 2024 BAO constraints, the cosmological hypothesis was **falsified** (Δχ² = -24.1).

The project successfully pivoted to computational astrophysics, producing a validated 1D reactive Euler solver for Type Ia supernova deflagration-to-detonation transition (DDT) via the Zel'dovich gradient mechanism.

## Results Summary

| Model | Δχ² vs ΛCDM | Status |
|-------|-------------|--------|
| Linear Spandrel (ε) | ~0 | Inconclusive |
| Riemann Resonance (γ₁) | -24.1 | **RULED OUT** |

| DDT Solver Metric | Simulated | Observed (SN Ia) |
|-------------------|-----------|------------------|
| Detonation time | 0.76 ms | 0.1–1 ms |
| Shock velocity | 5.4×10⁸ cm/s | 5–10×10⁸ cm/s |
| Ni-56 mass | 1.04 M☉ | 0.4–0.9 M☉ |

## Installation

```bash
# Clone or navigate to repository
cd pantheon

# Install (minimal)
pip install -e .

# Install with GPU acceleration (Apple Silicon)
pip install -e ".[gpu]"

# Install with JIT compilation
pip install -e ".[jit]"

# Install all optional dependencies
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Load and explore the data

```python
from data_interface import PantheonData

# Load Pantheon+SH0ES dataset
data = PantheonData()
print(data)  # PantheonData(n=1701, z=[0.0010, 2.2600], surveys=18)

# Get arrays for cosmological fitting
z, mu, mu_err = data.get_cosmology_data()

# Validation report
data.validate()
```

### Run cosmological analysis

```bash
python run_analysis.py --quick  # Fast verification
python run_analysis.py          # Full MCMC analysis
```

### Run DDT simulation

```bash
python -m ddt_solver.main_zeldovich
```

### Run tests

```bash
pytest tests/ -v
```

## Project Structure

```
pantheon/
├── constants.py              # Shared physical constants (CGS)
├── data_interface.py         # Clean Pantheon+ data loader
├── Pantheon+SH0ES.dat        # Primary dataset (1,701 SNe Ia)
│
├── spandrel_cosmology.py     # Spandrel hypothesis framework
├── spandrel_cosmology_hpc.py # High-performance MCMC analysis
├── spandrel_joint_analysis.py # Joint SNe + BAO constraints
├── spandrel_visualization.py # Publication-quality plots
├── spandrel_visuals.py       # Summary figures
├── riemann_resonance_cosmology.py # Riemann oscillation model
├── desi_riemann_synthesis.py # Multi-dataset comparison
├── run_analysis.py           # Main analysis entry point
│
├── ddt_solver/               # Type Ia supernova DDT solver
│   ├── eos_white_dwarf.py    # Chandrasekhar degenerate EOS
│   ├── flux_hllc.py          # HLLC Riemann solver
│   ├── reaction_carbon.py    # C12+C12 nuclear network
│   ├── main_zeldovich.py     # DDT simulation driver
│   └── nickel_yield.py       # Nucleosynthesis analysis
│
├── elevated/                 # Extended research modules
│   ├── model_comparison.py   # Bayesian evidence calculation
│   ├── alpha_chain_network.py # 13-isotope nuclear network
│   ├── light_curve_synthesis.py # SNe Ia light curves
│   ├── ddt_parameter_study.py # DDT phase space exploration
│   └── run_all.py            # Elevated analysis runner
│
├── synthesis/                # Theoretical framework
│   ├── turbulent_flame_theory.py # Kolmogorov cascade model
│   ├── phillips_from_turbulence.py # Phillips relation derivation
│   ├── unified_experiment.py # Complete synthesis chain
│   └── future_physics.py     # 3D LES roadmap
│
├── figures/                  # Generated output figures
├── results/                  # Analysis results
├── tests/                    # Test suite
├── _archive/                 # Archived exploratory code
│
├── DOCUMENTATION.md          # Complete technical documentation
├── pyproject.toml            # Package configuration
└── README.md                 # This file
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
2. **Caughlan-Fowler rates** — C12+C12 → NSE reaction network
3. **Zel'dovich mechanism** — Temperature gradient → shock coupling
4. **Arnett's rule** — Ni-56 mass → peak luminosity
5. **Chapman-Jouguet theory** — Detonation velocity bounds

## Dependencies

**Required:**
- numpy ≥ 1.20
- pandas ≥ 1.3
- scipy ≥ 1.7
- matplotlib ≥ 3.4

**Optional:**
- mlx ≥ 0.0.1 (Apple Silicon GPU acceleration)
- numba ≥ 0.55 (JIT compilation for hydrodynamics)

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
