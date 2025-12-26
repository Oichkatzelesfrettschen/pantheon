#!/usr/bin/env python3
"""
Nickel-56 Yield Analysis for Type Ia Supernova Simulation

Post-processes the DDT simulation to calculate:
    1. Mass that reached Nuclear Statistical Equilibrium (NSE)
    2. Estimated Ni-56 yield
    3. Comparison to observed Type Ia luminosities

Physics:
    - T > 5×10⁹ K: Complete silicon burning -> NSE -> Iron-group (mostly Ni-56)
    - T ~ 4-5×10⁹ K: Incomplete Si burning -> mixture of Si, S, Ar, Ca
    - T ~ 2-4×10⁹ K: Oxygen/Neon burning -> Intermediate mass elements
    - T < 2×10⁹ K: Carbon burning only -> O, Ne, Mg

The Ni-56 mass directly determines peak luminosity via:
    L_peak ~ 2×10^4^3 (M_Ni / M_sun) erg/s

Reference:
    - Arnett (1982), ApJ 253, 785 (Arnett's Rule)
    - Mazzali et al. (2007), Science 315, 825
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spandrel.core.constants import M_SUN, DAY, TAU_NI56, TAU_CO56, Q_BURN
from spandrel.ddt.eos_white_dwarf import eos_from_rho_T
from spandrel.ddt.flux_hllc import primitive_to_conserved, conserved_to_primitive
from spandrel.ddt.reaction_carbon import chapman_jouguet_velocity
from spandrel.ddt.main_zeldovich import SimulationConfig, ZeldovichDDTSolver

# Temperature thresholds for nucleosynthesis
T_NSE = 5.0e9      # K - Nuclear Statistical Equilibrium (-> Ni-56)
T_SI_BURN = 4.0e9  # K - Silicon burning (-> Si-group)
T_O_BURN = 2.0e9   # K - Oxygen burning (-> IME)
T_C_BURN = 1.0e9   # K - Carbon burning threshold

# NSE composition (mass fractions at T > T_NSE)
# At NSE, composition is determined by density and electron fraction
# For Y_e = 0.5 (C/O WD), predominantly Ni-56
X_NI56_NSE = 0.85   # ~85% Ni-56 by mass in NSE
X_HE4_NSE = 0.10    # ~10% alpha particles
X_OTHER_NSE = 0.05  # ~5% other iron-group


@dataclass
class NucleosynthesisResult:
    """Results of nucleosynthesis analysis."""
    # Masses by burning regime (grams)
    M_NSE: float          # Reached NSE (-> Ni-56)
    M_Si_burn: float      # Silicon burning (-> Si-group)
    M_O_burn: float       # Oxygen burning (-> IME)
    M_C_burn: float       # Carbon burning only
    M_unburned: float     # Never ignited

    # Derived quantities
    M_Ni56: float         # Estimated Ni-56 mass
    M_total_burned: float # Total mass processed

    # Observables
    L_peak: float         # Peak luminosity (erg/s)
    M_B_peak: float       # Absolute B magnitude at peak


class NickelYieldAnalyzer:
    """
    Analyzes DDT simulation output to compute nucleosynthesis yields.

    Tracks the maximum temperature reached by each fluid element
    throughout the simulation.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.dx = config.domain_size / config.n_cells
        self.x = np.linspace(0.5*self.dx, config.domain_size - 0.5*self.dx, config.n_cells)

        # Track maximum temperature history
        self.T_max = np.zeros(config.n_cells)

        # Track burned mass fractions
        self.X_C12_final = np.ones(config.n_cells) * config.X_C12_initial

    def update_temperature_history(self, T: np.ndarray):
        """Update the maximum temperature record."""
        self.T_max = np.maximum(self.T_max, T)

    def compute_yields(self, rho: np.ndarray, T_max: np.ndarray = None) -> NucleosynthesisResult:
        """
        Compute nucleosynthesis yields based on peak temperatures.

        Args:
            rho: Final density profile
            T_max: Maximum temperature reached (if None, uses stored history)

        Returns:
            NucleosynthesisResult with mass breakdown
        """
        if T_max is None:
            T_max = self.T_max

        # Mass per cell
        dm = rho * self.dx  # g/cm^2 (1D) - need to scale for 3D

        # For 1D simulation, we compute mass per unit area
        # To get total mass, we'd need to assume spherical geometry
        # Here we report "specific" quantities that can be scaled

        # Classification by peak temperature
        mask_NSE = T_max >= T_NSE
        mask_Si = (T_max >= T_SI_BURN) & (T_max < T_NSE)
        mask_O = (T_max >= T_O_BURN) & (T_max < T_SI_BURN)
        mask_C = (T_max >= T_C_BURN) & (T_max < T_O_BURN)
        mask_unburned = T_max < T_C_BURN

        # Masses in each regime (g/cm^2)
        M_NSE = np.sum(dm[mask_NSE])
        M_Si = np.sum(dm[mask_Si])
        M_O = np.sum(dm[mask_O])
        M_C = np.sum(dm[mask_C])
        M_unburned = np.sum(dm[mask_unburned])
        M_total = np.sum(dm)

        # Ni-56 production
        # NSE: ~85% goes to Ni-56
        # Si-burning: ~30% goes to Ni-56 (rest is Si-group)
        # O-burning: ~5% goes to Ni-56
        M_Ni56 = X_NI56_NSE * M_NSE + 0.30 * M_Si + 0.05 * M_O

        # Peak luminosity from Arnett's rule
        # L_peak ~ epsilon_Ni * M_Ni / tau_Ni
        # where epsilon_Ni = 3.9×10¹⁰ erg/g/s and tau_Ni = 8.8 days
        # Simplified: L_peak ~ 2×10^4^3 * (M_Ni / M_sun) erg/s
        L_peak = 2e43 * (M_Ni56 / M_SUN)

        # Absolute B magnitude
        # M_B ~ -19.3 for M_Ni = 0.6 M_sun (Phillips relation baseline)
        # M_B = -19.3 - 2.5 * log10(M_Ni / 0.6 M_sun)
        if M_Ni56 > 0:
            M_B_peak = -19.3 - 2.5 * np.log10(M_Ni56 / (0.6 * M_SUN))
        else:
            M_B_peak = 0.0

        return NucleosynthesisResult(
            M_NSE=M_NSE,
            M_Si_burn=M_Si,
            M_O_burn=M_O,
            M_C_burn=M_C,
            M_unburned=M_unburned,
            M_Ni56=M_Ni56,
            M_total_burned=M_NSE + M_Si + M_O + M_C,
            L_peak=L_peak,
            M_B_peak=M_B_peak
        )

    def plot_yields(self, rho: np.ndarray, T_max: np.ndarray = None,
                    save_path: str = None):
        """
        Visualize the nucleosynthesis zones.
        """
        if T_max is None:
            T_max = self.T_max

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        x_km = self.x / 1e5

        # Panel 1: Temperature history
        ax = axes[0, 0]
        ax.semilogy(x_km, T_max, 'r-', linewidth=2, label='T_max')
        ax.axhline(T_NSE, color='purple', linestyle='--', label=f'NSE ({T_NSE:.0e} K)')
        ax.axhline(T_SI_BURN, color='blue', linestyle='--', label=f'Si-burn ({T_SI_BURN:.0e} K)')
        ax.axhline(T_O_BURN, color='green', linestyle='--', label=f'O-burn ({T_O_BURN:.0e} K)')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('T_max (K)')
        ax.set_title('Maximum Temperature Reached')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 2: Burning zones
        ax = axes[0, 1]
        zones = np.zeros(len(T_max))
        zones[T_max >= T_NSE] = 4
        zones[(T_max >= T_SI_BURN) & (T_max < T_NSE)] = 3
        zones[(T_max >= T_O_BURN) & (T_max < T_SI_BURN)] = 2
        zones[(T_max >= T_C_BURN) & (T_max < T_O_BURN)] = 1

        colors = ['white', 'yellow', 'orange', 'red', 'purple']
        labels = ['Unburned', 'C-burn', 'O-burn', 'Si-burn', 'NSE (Ni-56)']

        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = zones == i
            if np.any(mask):
                ax.fill_between(x_km, 0, 1, where=mask, color=color,
                               alpha=0.7, label=label)

        ax.set_xlabel('x (km)')
        ax.set_ylabel('Zone')
        ax.set_title('Nucleosynthesis Zones')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.2)

        # Panel 3: Mass distribution
        ax = axes[1, 0]
        dm = rho * self.dx
        ax.plot(x_km, dm / 1e5, 'b-', linewidth=2)
        ax.fill_between(x_km, 0, dm / 1e5, where=T_max >= T_NSE,
                       color='purple', alpha=0.5, label='NSE')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('dm/dx (10⁵ g/cm)')
        ax.set_title('Mass Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 4: Yield summary
        ax = axes[1, 1]
        result = self.compute_yields(rho, T_max)

        # Normalize to total mass
        M_total = result.M_NSE + result.M_Si_burn + result.M_O_burn + result.M_C_burn + result.M_unburned

        if M_total > 0:
            fractions = [
                result.M_NSE / M_total,
                result.M_Si_burn / M_total,
                result.M_O_burn / M_total,
                result.M_C_burn / M_total,
                result.M_unburned / M_total
            ]
        else:
            fractions = [0, 0, 0, 0, 1]

        wedges, texts, autotexts = ax.pie(
            fractions,
            labels=['NSE\n(Ni-56)', 'Si-burn', 'O-burn', 'C-burn', 'Unburned'],
            colors=['purple', 'red', 'orange', 'yellow', 'lightgray'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title('Mass Fraction by Burning Regime')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved yield plot to {save_path}")

        plt.show()


class DDTWithYieldTracking(ZeldovichDDTSolver):
    """
    Extended DDT solver that tracks temperature history for yield analysis.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.analyzer = NickelYieldAnalyzer(config)
        self.T_history = []

    def _update_thermodynamics(self):
        """Override to track temperature history."""
        super()._update_thermodynamics()
        self.analyzer.update_temperature_history(self.T)

    def compute_final_yields(self) -> NucleosynthesisResult:
        """Compute yields after simulation completes."""
        rho, _, _ = conserved_to_primitive(self.U, self.gamma_eff)
        return self.analyzer.compute_yields(rho)

    def plot_yields(self, save_path: str = None):
        """Plot nucleosynthesis results."""
        rho, _, _ = conserved_to_primitive(self.U, self.gamma_eff)
        self.analyzer.plot_yields(rho, save_path=save_path)


def run_yield_analysis():
    """Run DDT simulation with yield tracking."""
    print("=" * 70)
    print("Type Ia Supernova DDT Simulation with Nickel Yield Analysis")
    print("=" * 70)
    print()

    # Configuration
    config = SimulationConfig(
        n_cells=1024,
        domain_size=1e7,
        rho_ambient=2e7,
        T_ambient=5e8,
        T_hotspot=3e9,
        hotspot_width=5e5,
        gradient_width=2e6,
        X_C12_initial=0.5,
        cfl=0.3,
        t_end=0.02,
        max_steps=50000,
        plot_interval=500,  # Less frequent plotting
        verbose=True
    )

    # Run simulation with yield tracking
    solver = DDTWithYieldTracking(config)
    solver.run(show_plots=False)  # Suppress main plots for speed

    # Compute yields
    print("\n" + "=" * 70)
    print("NUCLEOSYNTHESIS ANALYSIS")
    print("=" * 70)

    result = solver.compute_final_yields()

    # Get total mass in simulation (g/cm^2) and scale to 3D estimate
    rho, _, _ = conserved_to_primitive(solver.U, solver.gamma_eff)
    M_1D = np.sum(rho * solver.dx)  # g/cm^2 (column mass)

    # For a rough 3D estimate, assume this 1D slice represents
    # a spherical shell of radius R ~ domain_size / 2
    R_shell = config.domain_size / 2
    M_3D_estimate = 4 * np.pi * R_shell**2 * M_1D  # Very rough!

    print(f"\n1D Column Masses (g/cm^2):")
    print(f"  NSE (-> Ni-56):     {result.M_NSE:.3e} g/cm^2")
    print(f"  Si-burning:        {result.M_Si_burn:.3e} g/cm^2")
    print(f"  O-burning:         {result.M_O_burn:.3e} g/cm^2")
    print(f"  C-burning:         {result.M_C_burn:.3e} g/cm^2")
    print(f"  Unburned:          {result.M_unburned:.3e} g/cm^2")
    print(f"  Total:             {result.M_NSE + result.M_Si_burn + result.M_O_burn + result.M_C_burn + result.M_unburned:.3e} g/cm^2")

    # Mass fractions
    M_total = result.M_NSE + result.M_Si_burn + result.M_O_burn + result.M_C_burn + result.M_unburned
    if M_total > 0:
        print(f"\nMass Fractions:")
        print(f"  NSE (-> Ni-56):     {100*result.M_NSE/M_total:.1f}%")
        print(f"  Si-burning:        {100*result.M_Si_burn/M_total:.1f}%")
        print(f"  O-burning:         {100*result.M_O_burn/M_total:.1f}%")
        print(f"  C-burning:         {100*result.M_C_burn/M_total:.1f}%")
        print(f"  Unburned:          {100*result.M_unburned/M_total:.1f}%")

    # Estimated Ni-56 fraction
    f_Ni56 = result.M_Ni56 / M_total if M_total > 0 else 0
    print(f"\nEstimated Ni-56 mass fraction: {100*f_Ni56:.1f}%")

    # Scale to typical WD mass
    M_WD = 1.4 * M_SUN  # Chandrasekhar mass
    M_Ni56_scaled = f_Ni56 * M_WD

    print(f"\n" + "-" * 70)
    print("SCALED TO CHANDRASEKHAR MASS WHITE DWARF (1.4 M_sun):")
    print("-" * 70)
    print(f"  Estimated Ni-56 mass: {M_Ni56_scaled/M_SUN:.3f} M_sun")
    print(f"  Expected for normal SN Ia: 0.4-0.8 M_sun")

    # Peak luminosity
    L_peak_scaled = 2e43 * (M_Ni56_scaled / M_SUN)
    print(f"\n  Peak luminosity: {L_peak_scaled:.2e} erg/s")
    print(f"  Expected for normal SN Ia: ~1-2 × 10^4^3 erg/s")

    # Absolute magnitude
    if M_Ni56_scaled > 0:
        M_B = -19.3 - 2.5 * np.log10(M_Ni56_scaled / (0.6 * M_SUN))
        print(f"\n  Peak absolute B magnitude: M_B = {M_B:.2f}")
        print(f"  Expected for normal SN Ia: M_B ~ -19.3 +/- 0.3")

    # Classification
    print(f"\n" + "=" * 70)
    print("SUPERNOVA CLASSIFICATION:")
    print("=" * 70)

    f_Ni56_percent = 100 * f_Ni56
    if f_Ni56_percent > 50:
        print("  -> NORMAL Type Ia Supernova (high Ni-56 yield)")
        print("     This would be a valid 'standard candle' for cosmology!")
    elif f_Ni56_percent > 30:
        print("  -> SUBLUMINOUS Type Ia (1991bg-like)")
        print("     Lower Ni-56 -> dimmer, faster decline")
    elif f_Ni56_percent > 10:
        print("  -> TRANSITIONAL object")
        print("     Intermediate between normal and subluminous")
    else:
        print("  -> FAILED DETONATION or peculiar transient")
        print("     Insufficient Ni-56 for standard Type Ia")

    # Plot
    print("\nGenerating nucleosynthesis plot...")
    solver.plot_yields(save_path=Path(__file__).parent / 'nickel_yield.png')

    return solver, result


if __name__ == "__main__":
    solver, result = run_yield_analysis()
