# The Spandrel Project: Paper

Scientific paper synthesizing the cosmological hypothesis testing and DDT astrophysics validation.

## Building the Paper

### Requirements

- LuaLaTeX (TeX Live 2023+ recommended)
- Bibtex
- PGFPlots 1.18+
- TikZ

### Quick Build

```bash
cd paper
make
```

### Continuous Compilation

```bash
make watch
```

### Clean Build

```bash
make distclean
make pdf
```

## Structure

```
paper/
|-- spandrel_paper.tex    # Main document
|-- references.bib        # Bibliography
|-- Makefile              # Build system
|-- .latexmkrc            # latexmk configuration
|-- figures/              # TikZ/PGFPlots figures
│   |-- hubble_diagram.tex
│   |-- spandrel_correction.tex
│   |-- riemann_snake.tex
│   |-- ddt_structure.tex
│   |-- ddt_evolution.tex
│   |-- eos_gamma.tex
│   |-- nuclear_network.tex
│   |-- contour_H0_Om.tex
│   +-- kolmogorov_cascade.tex
+-- README.md             # This file
```

## Figures

All figures are created with TikZ and PGFPlots for vector graphics quality:

| Figure | Description |
|--------|-------------|
| `hubble_diagram.tex` | Pantheon+SH0ES Hubble diagram with residuals |
| `spandrel_correction.tex` | Stiffness parameter effect on distance modulus |
| `riemann_snake.tex` | Falsified Riemann resonance in w_0-wₐ plane |
| `ddt_structure.tex` | DDT wave structure schematic |
| `ddt_evolution.tex` | Time evolution of DDT simulation |
| `eos_gamma.tex` | Chandrasekhar EOS effective gamma |
| `nuclear_network.tex` | C12+C12 burning network diagram |
| `contour_H0_Om.tex` | H_0-Omegaₘ posterior contours |
| `kolmogorov_cascade.tex` | Turbulent cascade and fractal flames |

## Citation

If using this paper or code:

```bibtex
@article{SpandrelProject2025,
    title = {The Spandrel Project: From Cosmological Hypothesis to Astrophysical Validation},
    year = {2025},
    note = {GitHub: https://github.com/Oichkatzelesfrettschen/pantheon}
}
```
