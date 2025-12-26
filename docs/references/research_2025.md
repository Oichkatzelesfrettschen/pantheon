# Research Update: Type Ia SNe & Cosmology (2024-2025)

**Date:** December 23, 2025
**Source:** Gemini Analysis of Recent Literature

## 1. Deflagration-to-Detonation Transition (DDT)

### Current Consensus
The "Delayed Detonation" (DDT) model remains the standard for reproducing normal Type Ia supernovae (SNe Ia). The mechanism is believed to be the **Zel'dovich Gradient Mechanism** driven by turbulent conditioning of the flame front.

### Key Findings (2024-2025)
*   **JWST Confirmation:** Observations of **SN 2022aaiq** and **SN 2024gy** (Oct 2025) show distinct kinematic separation between deflagration ashes (central, stable Ni/Fe) and detonation ashes (outer, radioactive Ni-56). This validates the two-stage explosion model.
*   **Critical Length:** 3D full-star simulations continue to parameterize DDT, often triggering it when the flame enters the "distributed burning regime" at densities $\rho \approx 1-3 \times 10^7$ g/cm^3.
*   **Turbulence:** The transition is driven by the interaction of the flame with turbulent eddies, creating gradients $\nabla T$ that satisfy the spontaneous wave condition $u_{sp} \approx c_s$.

### Benchmarking Spandrel
*   **Spandrel Result:** $\lambda_{crit} \approx 5$ km at $\rho = 2 \times 10^7$ g/cm^3.
*   **Assessment:** This aligns well with the "macroscopic" view of DDT triggering in large eddy simulations (LES). The trend of $\lambda_{crit}$ increasing with decreasing density is physically correct.
*   **Gap:** The current Spandrel solver does not track the kinematic separation of ashes (1D only), limiting direct comparison with JWST geometry.

## 2. Cosmology (Pantheon+ & SH0ES)

### Current Status
The **Pantheon+SH0ES** dataset (1701 SNe) remains the gold standard for local $H_0$ and dark energy constraints.

### Key Findings (2025)
*   **Hubble Tension:** Persists at $>5\sigma$. $H_0 \approx 73$ km/s/Mpc (SH0ES) vs $67$ km/s/Mpc (Planck).
*   **Dark Energy:** Data is fully consistent with $\Lambda$CDM ($w = -1$).
*   **Oscillating Models:** No evidence for oscillating dark energy (like the Riemann Resonance model falsified by Spandrel). Analysis of low-multipole anisotropies (dipoles) in 2025 shows some anomalies but no smoking gun for new physics.

### Benchmarking Spandrel
*   **Spandrel Result:** $\Delta \chi^2 = -24.1$ for Riemann Resonance.
*   **Assessment:** Consistent with the broader community finding that the expansion history is smooth and $\Lambda$-dominated.

## 3. Recommendations for Spandrel
1.  **Yield Calculation:** Update the DDT solver to run *post-detonation* to calculate total Ni-56 yield, enabling comparison with JWST "ash separation" data.
2.  **3D Extrapolation:** Use the "Fractal Dimension" module (`synthesis/turbulent_flame_theory.py`) to bridge 1D results to 3D observables.
