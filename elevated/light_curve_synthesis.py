#!/usr/bin/env python3
"""
Light Curve Synthesis for Type Ia Supernovae

Converts Ni-56 yields from DDT simulation into observable light curves
using Arnett's analytical model and radiative transfer approximations.

The light curve is powered by:
    Ni-56 → Co-56 → Fe-56 (radioactive decay chain)

Key observables:
    - Peak luminosity (Arnett's rule: L_peak ∝ M_Ni)
    - Rise time (~17-20 days)
    - Decline rate (Δm₁₅: magnitude decline in 15 days)
    - Phillips relation: brighter SNe decline slower

Reference:
    - Arnett (1982), ApJ 253, 785
    - Pinto & Eastman (2000), ApJ 530, 744
    - Kasen (2010), ApJ 708, 1025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from pathlib import Path

import sys
sys.path.insert(0, '..')
from constants import (
    C_LIGHT_CGS as C_LIGHT, M_SUN, L_SUN, DAY, SIGMA_SB,
    TAU_NI56, TAU_CO56, E_NI56, E_CO56, M_AMU
)

# Energy release per decay
Q_NI56 = 1.75e6 * 1.602e-12   # MeV → erg (γ-rays)
Q_CO56 = 3.73e6 * 1.602e-12   # MeV → erg (γ-rays + positrons)

# Specific heating rates at t=0
EPSILON_NI = Q_NI56 / (TAU_NI56 * 56 * M_AMU)  # erg/g/s
EPSILON_CO = Q_CO56 / (TAU_CO56 * 56 * M_AMU)


# =============================================================================
# RADIOACTIVE HEATING
# =============================================================================
def radioactive_heating(t: np.ndarray, M_Ni: float) -> np.ndarray:
    """
    Instantaneous heating rate from Ni-56/Co-56 decay chain.

    Q(t) = M_Ni × [ε_Ni × exp(-t/τ_Ni) + ε_Co × (exp(-t/τ_Co) - exp(-t/τ_Ni)) × τ_Ni/(τ_Ni - τ_Co)]

    Args:
        t: Time since explosion (seconds)
        M_Ni: Nickel-56 mass (grams)

    Returns:
        Q: Heating rate (erg/s)
    """
    # Ni-56 decay
    Q_Ni = EPSILON_NI * np.exp(-t / TAU_NI56)

    # Co-56 decay (from Ni-56 daughter)
    # Accounts for Co-56 build-up from Ni-56 decay
    ratio = TAU_NI56 / (TAU_NI56 - TAU_CO56)
    Q_Co = EPSILON_CO * ratio * (np.exp(-t / TAU_CO56) - np.exp(-t / TAU_NI56))

    return M_Ni * (Q_Ni + Q_Co)


# =============================================================================
# ARNETT MODEL
# =============================================================================
@dataclass
class ArnettModel:
    """
    Arnett's analytical light curve model.

    Assumes:
        - Homologous expansion: v ∝ r
        - Constant opacity
        - Centrally concentrated Ni-56

    Key parameter:
        τ_m = (κ M / v c)^(1/2)  "effective diffusion time"

    Where κ is opacity, M is ejecta mass, v is expansion velocity.
    """

    M_ej: float = 1.4 * M_SUN    # Ejecta mass (g)
    M_Ni: float = 0.6 * M_SUN    # Ni-56 mass (g)
    v_exp: float = 1e9           # Expansion velocity (cm/s)
    kappa: float = 0.2           # Opacity (cm²/g) - Fe-group dominated

    def __post_init__(self):
        # Effective diffusion timescale
        self.tau_m = np.sqrt(2 * self.kappa * self.M_ej / (13.8 * C_LIGHT * self.v_exp))

        # Rise time (approximately τ_m)
        self.t_rise = self.tau_m

        # Peak luminosity (Arnett's rule)
        self.L_peak = radioactive_heating(np.array([self.t_rise]), self.M_Ni)[0]

    def luminosity(self, t: np.ndarray) -> np.ndarray:
        """
        Compute bolometric luminosity L(t).

        Uses the integral form of Arnett's solution:
        L(t) = exp(-(t/τ_m)²) × ∫₀ᵗ Q(t') × 2t'/τ_m² × exp((t'/τ_m)²) dt'
        """
        L = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti <= 0:
                L[i] = 0
                continue

            # Numerical integration
            n_steps = 1000
            t_prime = np.linspace(0, ti, n_steps)
            dt = t_prime[1] - t_prime[0] if len(t_prime) > 1 else 0

            Q = radioactive_heating(t_prime, self.M_Ni)
            integrand = Q * 2 * t_prime / self.tau_m**2 * np.exp((t_prime / self.tau_m)**2)

            integral = np.trapz(integrand, t_prime)
            L[i] = np.exp(-(ti / self.tau_m)**2) * integral

        return L

    def magnitude(self, t: np.ndarray, band: str = 'bol') -> np.ndarray:
        """
        Convert luminosity to absolute magnitude.

        For bolometric: M_bol = -2.5 log₁₀(L / L_sun) + 4.74
        """
        L = self.luminosity(t)
        L = np.maximum(L, 1e30)  # Floor to avoid log(0)

        if band == 'bol':
            M = -2.5 * np.log10(L / L_SUN) + 4.74
        elif band == 'B':
            # B-band is roughly 0.8-1.0 of bolometric at peak
            # With color evolution approximation
            BC = -0.1 + 0.015 * (t / DAY - 15)  # Bolometric correction
            M = -2.5 * np.log10(L / L_SUN) + 4.74 + BC
        else:
            M = -2.5 * np.log10(L / L_SUN) + 4.74

        return M


# =============================================================================
# PHILLIPS RELATION
# =============================================================================
def phillips_relation(delta_m15: float) -> float:
    """
    Phillips relation: absolute B magnitude at peak vs decline rate.

    M_B(max) = -19.3 + 0.78 × (Δm₁₅(B) - 1.1)

    Args:
        delta_m15: Magnitude decline in B-band over 15 days

    Returns:
        M_B_max: Peak absolute B magnitude
    """
    return -19.3 + 0.78 * (delta_m15 - 1.1)


def compute_delta_m15(model: ArnettModel) -> float:
    """
    Compute Δm₁₅ (decline rate) from light curve model.
    """
    # Find peak
    t_grid = np.linspace(5 * DAY, 30 * DAY, 1000)
    mag = model.magnitude(t_grid, band='B')
    i_peak = np.argmin(mag)
    t_peak = t_grid[i_peak]
    mag_peak = mag[i_peak]

    # Magnitude 15 days after peak
    t_15 = t_peak + 15 * DAY
    mag_15 = model.magnitude(np.array([t_15]), band='B')[0]

    return mag_15 - mag_peak


# =============================================================================
# LIGHT CURVE GENERATOR
# =============================================================================
class LightCurveGenerator:
    """
    Generate synthetic light curves from DDT simulation output.
    """

    def __init__(self, M_Ni: float, M_ej: float = 1.4 * M_SUN,
                 v_exp: float = 1e9, kappa: float = 0.2):
        """
        Args:
            M_Ni: Nickel-56 mass from simulation (grams)
            M_ej: Total ejecta mass (grams)
            v_exp: Expansion velocity (cm/s)
            kappa: Mean opacity (cm²/g)
        """
        self.model = ArnettModel(M_ej=M_ej, M_Ni=M_Ni, v_exp=v_exp, kappa=kappa)

    def generate(self, t_start: float = 0, t_end: float = 100 * DAY,
                 n_points: int = 500) -> Dict:
        """
        Generate complete light curve data.

        Returns:
            Dictionary with time, luminosity, magnitude, observables
        """
        t = np.linspace(t_start, t_end, n_points)

        # Bolometric
        L_bol = self.model.luminosity(t)
        M_bol = self.model.magnitude(t, band='bol')

        # B-band
        M_B = self.model.magnitude(t, band='B')

        # Key observables
        i_peak = np.argmin(M_B)
        t_peak = t[i_peak]
        L_peak = L_bol[i_peak]
        M_B_peak = M_B[i_peak]

        # Decline rate
        delta_m15 = compute_delta_m15(self.model)

        # Phillips prediction
        M_B_phillips = phillips_relation(delta_m15)

        # Rise time
        half_max = L_peak / 2
        i_half = np.argmin(np.abs(L_bol[:i_peak] - half_max))
        t_rise = t_peak - t[i_half]

        return {
            't': t,
            't_days': t / DAY,
            'L_bol': L_bol,
            'M_bol': M_bol,
            'M_B': M_B,
            'observables': {
                't_peak': t_peak / DAY,
                'L_peak': L_peak,
                'M_B_peak': M_B_peak,
                'delta_m15': delta_m15,
                'M_B_phillips': M_B_phillips,
                't_rise': t_rise / DAY,
                'M_Ni': self.model.M_Ni / M_SUN,
            }
        }

    def plot(self, save_path: str = None):
        """Generate publication-quality light curve plot."""
        data = self.generate()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Style
        plt.style.use('dark_background')

        t_days = data['t_days']
        obs = data['observables']

        # Panel 1: Bolometric luminosity
        ax = axes[0, 0]
        ax.semilogy(t_days, data['L_bol'], 'orange', linewidth=2)
        ax.axvline(obs['t_peak'], color='white', linestyle='--', alpha=0.5, label=f"Peak: {obs['t_peak']:.1f} d")
        ax.axhline(obs['L_peak'], color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Days since explosion')
        ax.set_ylabel('Bolometric Luminosity (erg/s)')
        ax.set_title('Bolometric Light Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

        # Panel 2: B-band magnitude
        ax = axes[0, 1]
        ax.plot(t_days, data['M_B'], 'b-', linewidth=2, label='Model')
        ax.axhline(obs['M_B_phillips'], color='red', linestyle='--',
                  label=f"Phillips: {obs['M_B_phillips']:.2f}")
        ax.scatter([obs['t_peak']], [obs['M_B_peak']], s=100, c='yellow',
                  marker='*', zorder=5, label=f"Peak: {obs['M_B_peak']:.2f}")
        ax.set_xlabel('Days since explosion')
        ax.set_ylabel('Absolute B Magnitude')
        ax.set_title(f'B-Band Light Curve (Δm₁₅ = {obs["delta_m15"]:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)

        # Panel 3: Radioactive heating
        ax = axes[1, 0]
        t_sec = data['t']
        Q = radioactive_heating(t_sec, self.model.M_Ni)
        ax.semilogy(t_days, Q, 'r-', linewidth=2, label='Total')

        # Ni-56 component
        Q_Ni = self.model.M_Ni * EPSILON_NI * np.exp(-t_sec / TAU_NI56)
        ax.semilogy(t_days, Q_Ni, 'g--', linewidth=1.5, label='Ni-56')

        ax.set_xlabel('Days since explosion')
        ax.set_ylabel('Heating Rate (erg/s)')
        ax.set_title('Radioactive Energy Deposition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

        # Panel 4: Summary table
        ax = axes[1, 1]
        ax.axis('off')

        summary = f"""
        ╔══════════════════════════════════════════╗
        ║     LIGHT CURVE SYNTHESIS RESULTS        ║
        ╠══════════════════════════════════════════╣
        ║  Input Parameters:                       ║
        ║    M_Ni  = {obs['M_Ni']:.3f} M☉                     ║
        ║    M_ej  = {self.model.M_ej/M_SUN:.2f} M☉                      ║
        ║    v_exp = {self.model.v_exp:.1e} cm/s             ║
        ║    κ     = {self.model.kappa:.2f} cm²/g                   ║
        ╠══════════════════════════════════════════╣
        ║  Derived Observables:                    ║
        ║    Rise time    = {obs['t_rise']:.1f} days                ║
        ║    Peak time    = {obs['t_peak']:.1f} days                ║
        ║    L_peak       = {obs['L_peak']:.2e} erg/s     ║
        ║    M_B(peak)    = {obs['M_B_peak']:.2f} mag                 ║
        ║    Δm₁₅(B)      = {obs['delta_m15']:.2f} mag                 ║
        ╠══════════════════════════════════════════╣
        ║  Phillips Relation Check:                ║
        ║    Predicted M_B = {obs['M_B_phillips']:.2f} mag              ║
        ║    Actual M_B    = {obs['M_B_peak']:.2f} mag              ║
        ║    Residual      = {obs['M_B_peak'] - obs['M_B_phillips']:+.2f} mag              ║
        ╚══════════════════════════════════════════╝
        """

        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
               fontfamily='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#4a4a6a'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight',
                       facecolor='#0d1117', edgecolor='none')
            print(f"Saved: {save_path}")

        plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TYPE Ia SUPERNOVA LIGHT CURVE SYNTHESIS")
    print("=" * 70)

    # From our DDT simulation: M_Ni = 1.04 M_sun
    M_Ni_sim = 1.04 * M_SUN

    print(f"\nInput from DDT simulation:")
    print(f"  Ni-56 mass: {M_Ni_sim/M_SUN:.2f} M☉")

    # Generate light curve
    generator = LightCurveGenerator(M_Ni=M_Ni_sim)
    data = generator.generate()
    obs = data['observables']

    print(f"\nSynthesized observables:")
    print(f"  Rise time: {obs['t_rise']:.1f} days")
    print(f"  Peak luminosity: {obs['L_peak']:.2e} erg/s")
    print(f"  Peak M_B: {obs['M_B_peak']:.2f}")
    print(f"  Δm₁₅(B): {obs['delta_m15']:.2f}")

    print(f"\nPhillips relation check:")
    print(f"  Predicted from Δm₁₅: M_B = {obs['M_B_phillips']:.2f}")
    print(f"  Actual from model:   M_B = {obs['M_B_peak']:.2f}")
    residual = obs['M_B_peak'] - obs['M_B_phillips']
    print(f"  Residual: {residual:+.2f} mag")

    if abs(residual) < 0.3:
        print(f"\n  ✓ CONSISTENT with Phillips relation!")
    else:
        print(f"\n  ⚠ Deviation from Phillips relation")

    # Plot
    generator.plot(save_path=Path(__file__).parent / 'light_curve.png')
