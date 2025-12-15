#!/usr/bin/env python3
"""
Turbulent Flame Theory for Type Ia Supernovae

Connects the microscopic (turbulent flame structure) to the macroscopic
(supernova luminosity) through the Fractal Dimension framework.

Key theoretical relations:
    1. Turbulent flame speed: S_T = S_L × (L/λ_k)^(D-2)
    2. Zel'dovich criticality: u_sp = c_s → detonation
    3. Burn efficiency: η = f(D, ρ, gradient)
    4. Phillips relation: Δm₁₅ ∝ 1/D

This module provides the theoretical framework connecting:
    Kolmogorov Cascade → Flame Wrinkling → DDT → Ni-56 → Light Curve

Reference:
    - Kolmogorov (1941), Dokl. Akad. Nauk SSSR
    - Peters (2000), "Turbulent Combustion"
    - Röpke (2007), ApJ 668, 1103
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Callable
from scipy.integrate import odeint
from scipy.optimize import brentq
import sys
sys.path.insert(0, '..')

from constants import (
    C_LIGHT_CGS as C_LIGHT,
    K_BOLTZMANN,
    M_PROTON,
    M_SUN,
    DAY,
    RHO_DDT,
    R_WD,
    TAU_NI56,
    TAU_CO56
)

# White dwarf parameters (module-specific)
M_WD = 1.4 * M_SUN       # Chandrasekhar mass
RHO_CENTRAL = 2e9        # g/cm³ (central density)


# =============================================================================
# KOLMOGOROV TURBULENCE
# =============================================================================
@dataclass
class KolmogorovCascade:
    """
    Kolmogorov turbulence model for white dwarf convection.

    Energy cascades from integral scale L to Kolmogorov scale λ_k:
        ε = u³/L  (energy dissipation rate)
        λ_k = (ν³/ε)^(1/4)  (dissipation scale)

    Velocity at scale r:
        u(r) = (ε × r)^(1/3)  (Kolmogorov scaling)
    """

    L_integral: float = 1e7      # cm (100 km) - convective cell size
    u_rms: float = 1e7           # cm/s (100 km/s) - turbulent velocity
    nu: float = 1e3              # cm²/s - kinematic viscosity (degenerate)

    def __post_init__(self):
        # Energy dissipation rate
        self.epsilon = self.u_rms**3 / self.L_integral

        # Kolmogorov scale
        self.lambda_k = (self.nu**3 / self.epsilon)**0.25

        # Reynolds number
        self.Re = self.u_rms * self.L_integral / self.nu

    def velocity_at_scale(self, r: np.ndarray) -> np.ndarray:
        """Turbulent velocity at scale r (Kolmogorov scaling)."""
        return (self.epsilon * r)**(1/3)

    def eddy_turnover_time(self, r: np.ndarray) -> np.ndarray:
        """Eddy turnover time at scale r."""
        u_r = self.velocity_at_scale(r)
        return r / u_r

    def gradient_probability(self, L_gradient: float) -> float:
        """
        Probability of finding a gradient of length L_gradient.

        Based on intermittency in turbulent flows - gradients are
        not uniformly distributed but cluster at certain scales.
        """
        # Log-normal distribution of gradient lengths
        # (Characteristic of turbulent intermittency)
        sigma = 0.5  # Log-standard deviation
        L_mean = self.L_integral / 10  # Mean gradient is 1/10 of integral scale

        log_L = np.log(L_gradient / L_mean)
        return np.exp(-0.5 * (log_L / sigma)**2) / (L_gradient * sigma * np.sqrt(2 * np.pi))


# =============================================================================
# FRACTAL FLAME MODEL
# =============================================================================
@dataclass
class FractalFlame:
    """
    Fractal model of turbulent premixed flames.

    The flame surface is wrinkled by turbulence, increasing its
    effective area and thus the burning rate.

    Key parameter: Fractal dimension D (2 < D < 3)
        D = 2: Smooth (laminar) flame
        D = 7/3 ≈ 2.33: Kolmogorov cascade (self-similar)
        D = 3: Space-filling (hypothetical limit)

    Flame speed enhancement:
        S_T/S_L = (L_outer/L_inner)^(D-2)

    Where:
        L_outer = integral scale (largest wrinkle)
        L_inner = Gibson scale (smallest wrinkle before flame smooths it)
    """

    D_fractal: float = 2.35      # Fractal dimension
    S_laminar: float = 1e6       # cm/s - laminar flame speed
    L_outer: float = 1e7         # cm - outer cutoff (integral scale)
    L_inner: float = 1e4         # cm - inner cutoff (Gibson scale)

    def __post_init__(self):
        # Turbulent flame speed
        self.S_turbulent = self.S_laminar * (self.L_outer / self.L_inner)**(self.D_fractal - 2)

        # Flame surface area enhancement
        self.area_ratio = (self.L_outer / self.L_inner)**(self.D_fractal - 2)

    @classmethod
    def from_turbulence(cls, cascade: KolmogorovCascade, S_laminar: float = 1e6):
        """
        Derive fractal flame parameters from turbulence properties.

        The Gibson scale is where flame speed = turbulent velocity:
            L_Gibson: S_L = u(L_Gibson)
        """
        # Gibson scale: S_L = (ε × L_G)^(1/3)
        L_Gibson = (S_laminar**3 / cascade.epsilon)

        # Fractal dimension from turbulence (Peters formula)
        # D = 2 + (D_3 - 2) × min(1, Da^(-1/2))
        # Da = Damköhler number = τ_turb / τ_chem
        tau_turb = cascade.L_integral / cascade.u_rms
        tau_chem = 1e-6  # Chemical timescale (approximate)
        Da = tau_turb / tau_chem

        D_fractal = 2 + 1/3 * min(1, Da**(-0.5))  # Simplified Peters formula

        return cls(
            D_fractal=D_fractal,
            S_laminar=S_laminar,
            L_outer=cascade.L_integral,
            L_inner=max(L_Gibson, cascade.lambda_k)
        )

    def burning_rate(self, rho: float, T: float) -> float:
        """
        Mass burning rate per unit volume.

        ω = ρ × S_T × A_flame / V
        """
        # Flame area per unit volume scales with fractal dimension
        A_per_V = self.area_ratio / self.L_outer

        return rho * self.S_turbulent * A_per_V

    def burn_efficiency(self, t_available: float) -> float:
        """
        Fraction of fuel burned in available time.

        η = 1 - exp(-S_T × t / L_burn)
        """
        L_burn = self.L_outer  # Characteristic burn length
        return 1 - np.exp(-self.S_turbulent * t_available / L_burn)


# =============================================================================
# ZEL'DOVICH CRITICALITY
# =============================================================================
@dataclass
class ZeldovichCriticality:
    """
    Zel'dovich gradient mechanism for DDT.

    The spontaneous wave velocity is:
        u_sp = |dT/dx|^(-1) × |dT/dt|_burn

    Criticality: u_sp ≈ c_s (sound speed)

    This defines a critical gradient length:
        λ_crit = c_s × τ_burn
    """

    rho: float = 2e7             # g/cm³
    T_hot: float = 3e9           # K
    T_cold: float = 5e8          # K
    c_s: float = 5e8             # cm/s (sound speed)

    def __post_init__(self):
        # Burning timescale (from C12+C12 rates)
        # τ_burn ∝ T^(-20) at high T
        T9 = self.T_hot / 1e9
        self.tau_burn = 1e-6 * T9**(-20) * (self.rho / 1e7)**(-1)

        # Critical gradient length
        self.lambda_crit = self.c_s * self.tau_burn

        # Temperature gradient for criticality
        self.dT_dx_crit = (self.T_hot - self.T_cold) / self.lambda_crit

    def spontaneous_velocity(self, L_gradient: float) -> float:
        """
        Spontaneous wave velocity for given gradient length.
        """
        dT_dx = (self.T_hot - self.T_cold) / L_gradient
        dT_dt = (self.T_hot - self.T_cold) / self.tau_burn

        return dT_dt / dT_dx if dT_dx > 0 else np.inf

    def is_critical(self, L_gradient: float, tolerance: float = 0.2) -> bool:
        """
        Check if gradient produces near-critical conditions.

        Returns True if u_sp is within tolerance of c_s.
        """
        u_sp = self.spontaneous_velocity(L_gradient)
        return abs(u_sp - self.c_s) / self.c_s < tolerance

    def detonation_probability(self, L_gradient: float) -> float:
        """
        Probability of detonation given gradient length.

        Uses a sigmoid function centered at λ_crit.
        """
        x = (L_gradient - self.lambda_crit) / (0.3 * self.lambda_crit)
        return 1 / (1 + np.exp(-x))


# =============================================================================
# UNIFIED MODEL: TURBULENCE → PHILLIPS RELATION
# =============================================================================
class TurbulentSupernovaModel:
    """
    Complete model connecting turbulence to observables.

    Chain of causation:
        1. Convection → Kolmogorov cascade
        2. Cascade → Fractal flame (D)
        3. Flame + Gradient → DDT probability
        4. DDT → Ni-56 mass
        5. Ni-56 → Light curve (Phillips relation)
    """

    def __init__(self, L_integral: float = 1e7, u_rms: float = 1e7):
        """
        Initialize model with turbulence parameters.

        Args:
            L_integral: Integral scale (cm) - convective cell size
            u_rms: RMS turbulent velocity (cm/s)
        """
        # Build cascade
        self.cascade = KolmogorovCascade(L_integral=L_integral, u_rms=u_rms)

        # Derive flame properties
        self.flame = FractalFlame.from_turbulence(self.cascade)

        # Zel'dovich criticality
        self.zeldovich = ZeldovichCriticality()

        # Store key parameters
        self.D_fractal = self.flame.D_fractal
        self.lambda_crit = self.zeldovich.lambda_crit

    def compute_ni56_yield(self, n_gradients: int = 1000) -> float:
        """
        Monte Carlo estimate of Ni-56 yield from turbulent gradient distribution.

        Samples gradient lengths from turbulent probability distribution,
        computes DDT probability for each, and averages the yields.
        """
        # Sample gradient lengths (log-uniform)
        L_gradients = np.exp(np.random.uniform(
            np.log(1e5), np.log(1e7), n_gradients
        ))

        # Probability weight for each gradient
        weights = np.array([
            self.cascade.gradient_probability(L) for L in L_gradients
        ])
        weights /= np.sum(weights)

        # DDT probability for each
        P_ddt = np.array([
            self.zeldovich.detonation_probability(L) for L in L_gradients
        ])

        # Ni-56 yield:
        # - DDT → ~100% burn (NSE) → ~0.85 goes to Ni-56
        # - No DDT → ~30% burn (deflagration) → ~0.5 goes to Ni-56
        Ni56_ddt = 0.85
        Ni56_deflag = 0.15

        yield_per_gradient = P_ddt * Ni56_ddt + (1 - P_ddt) * Ni56_deflag

        # Weighted average
        f_Ni56 = np.sum(weights * yield_per_gradient)

        return f_Ni56 * 1.4  # Scale to Chandrasekhar mass (M_sun)

    def phillips_relation(self) -> Tuple[float, float]:
        """
        Derive Phillips relation parameters from fractal dimension.

        Returns (Δm₁₅, M_B_peak) prediction.
        """
        # Empirical fit: Δm₁₅ scales inversely with D
        # Higher D → more efficient burn → more Ni-56 → broader light curve
        delta_m15 = 1.1 * (2.35 / self.D_fractal)

        # Peak magnitude from Ni-56 yield
        M_Ni = self.compute_ni56_yield()
        M_B_peak = -19.3 - 2.5 * np.log10(M_Ni / 0.6)

        return delta_m15, M_B_peak

    def critical_gradient_analysis(self, L_range: np.ndarray = None) -> Dict:
        """
        Analyze DDT probability vs gradient length.

        Returns dictionary with:
            - L_gradients: Gradient lengths
            - P_ddt: DDT probability
            - u_sp: Spontaneous velocity
            - lambda_crit: Critical gradient
        """
        if L_range is None:
            L_range = np.logspace(5, 7, 100)  # 1 km to 100 km

        P_ddt = np.array([self.zeldovich.detonation_probability(L) for L in L_range])
        u_sp = np.array([self.zeldovich.spontaneous_velocity(L) for L in L_range])

        return {
            'L_gradients': L_range,
            'L_km': L_range / 1e5,
            'P_ddt': P_ddt,
            'u_sp': u_sp,
            'c_s': self.zeldovich.c_s,
            'lambda_crit': self.zeldovich.lambda_crit,
            'lambda_crit_km': self.zeldovich.lambda_crit / 1e5
        }

    def turbulence_sensitivity(self, L_integral_range: np.ndarray,
                               u_rms_range: np.ndarray) -> np.ndarray:
        """
        2D sensitivity analysis: how do turbulence params affect Ni-56 yield?

        Returns 2D array of Ni-56 yields.
        """
        yields = np.zeros((len(L_integral_range), len(u_rms_range)))

        for i, L in enumerate(L_integral_range):
            for j, u in enumerate(u_rms_range):
                model = TurbulentSupernovaModel(L_integral=L, u_rms=u)
                yields[i, j] = model.compute_ni56_yield()

        return yields


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TURBULENT FLAME THEORY: Type Ia Supernova Model")
    print("=" * 70)

    # Initialize model
    model = TurbulentSupernovaModel(L_integral=1e7, u_rms=1e7)

    print(f"\nTurbulence Parameters:")
    print(f"  Integral scale: {model.cascade.L_integral/1e5:.0f} km")
    print(f"  RMS velocity: {model.cascade.u_rms/1e5:.0f} km/s")
    print(f"  Kolmogorov scale: {model.cascade.lambda_k:.2e} cm")
    print(f"  Reynolds number: {model.cascade.Re:.2e}")

    print(f"\nFractal Flame:")
    print(f"  Fractal dimension D: {model.flame.D_fractal:.3f}")
    print(f"  S_turbulent / S_laminar: {model.flame.S_turbulent/model.flame.S_laminar:.1f}")
    print(f"  Gibson scale: {model.flame.L_inner:.2e} cm")

    print(f"\nZel'dovich Criticality:")
    print(f"  τ_burn: {model.zeldovich.tau_burn:.2e} s")
    print(f"  λ_crit: {model.zeldovich.lambda_crit/1e5:.1f} km")
    print(f"  c_s: {model.zeldovich.c_s:.2e} cm/s")

    print(f"\nNi-56 Yield (Monte Carlo):")
    M_Ni = model.compute_ni56_yield()
    print(f"  M_Ni = {M_Ni:.2f} M☉")

    print(f"\nPhillips Relation Prediction:")
    delta_m15, M_B = model.phillips_relation()
    print(f"  Δm₁₅(B) = {delta_m15:.2f} mag")
    print(f"  M_B(peak) = {M_B:.2f} mag")

    # Critical gradient analysis
    print(f"\nCritical Gradient Analysis:")
    analysis = model.critical_gradient_analysis()
    print(f"  λ_crit = {analysis['lambda_crit_km']:.1f} km")

    # Find 50% DDT probability point
    idx_50 = np.argmin(np.abs(analysis['P_ddt'] - 0.5))
    print(f"  50% DDT probability at L = {analysis['L_km'][idx_50]:.1f} km")
