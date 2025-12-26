"""
Equation of State for White Dwarf Matter

Implements the degenerate electron EOS appropriate for Type Ia supernova
conditions (rho ~ 10^7 g/cm^3, T ~ 10^9 K).

The total pressure is:
    P = P_deg(rho) + P_ion(rho, T) + P_rad(T)

Where:
    P_deg: Relativistic degenerate electron pressure (Fermi-Dirac)
    P_ion: Ideal gas contribution from ions (C/O nuclei)
    P_rad: Radiation pressure (usually negligible)

Reference: Timmes & Swesty (2000), ApJS 126, 501
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import sys
sys.path.insert(0, '..')
from spandrel.core.constants import (
    C_LIGHT_CGS as C_LIGHT,
    K_BOLTZMANN,
    M_ELECTRON,
    M_PROTON,
    A_RAD,
    HBAR,
    A_BAR,
    Z_BAR,
    Y_E
)


@dataclass
class EOSState:
    """Container for thermodynamic state variables."""
    rho: np.ndarray      # Density (g/cm^3)
    T: np.ndarray        # Temperature (K)
    P: np.ndarray        # Total pressure (erg/cm^3 = dyne/cm^2)
    e_int: np.ndarray    # Specific internal energy (erg/g)
    cs: np.ndarray       # Sound speed (cm/s)
    gamma_eff: np.ndarray  # Effective adiabatic index


def electron_density(rho: np.ndarray) -> np.ndarray:
    """Number density of electrons from charge neutrality."""
    return Y_E * rho / M_PROTON


def fermi_momentum(n_e: np.ndarray) -> np.ndarray:
    """Fermi momentum p_F = hbar(3pi^2n_e)^(1/3)."""
    return HBAR * (3.0 * np.pi**2 * n_e)**(1.0/3.0)


def relativity_parameter(p_F: np.ndarray) -> np.ndarray:
    """x = p_F / (m_e * c) - dimensionless relativity parameter."""
    return p_F / (M_ELECTRON * C_LIGHT)


def pressure_degenerate(rho: np.ndarray) -> np.ndarray:
    """
    Relativistic degenerate electron pressure.

    Uses the Chandrasekhar formula:
    P_deg = (pi m_e^4 c⁵)/(3 h^3) * f(x)

    where f(x) = x(2x^2-3)√(x^2+1) + 3 sinh⁻¹(x)

    Limits:
        x << 1 (non-relativistic): P ∝ rho^(5/3)
        x >> 1 (ultra-relativistic): P ∝ rho^(4/3)
    """
    n_e = electron_density(rho)
    p_F = fermi_momentum(n_e)
    x = relativity_parameter(p_F)

    # Chandrasekhar function f(x)
    # f(x) = x(2x^2-3)√(x^2+1) + 3 arcsinh(x)
    sqrt_term = np.sqrt(x**2 + 1.0)
    f_x = x * (2.0 * x**2 - 3.0) * sqrt_term + 3.0 * np.arcsinh(x)

    # Prefactor: (pi m_e^4 c⁵) / (3 h^3) = (m_e c^2)/(8pi^2) * (m_e c / hbar)^3
    # Using: (m_e c / hbar) = 1/(Compton wavelength / 2pi)
    prefactor = (np.pi * M_ELECTRON**4 * C_LIGHT**5) / (3.0 * (2.0 * np.pi * HBAR)**3)

    return prefactor * f_x


def pressure_ions(rho: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Ideal gas pressure from ions (C/O nuclei).
    P_ion = n_ion * k_B * T = (rho / A_bar m_p) * k_B * T
    """
    n_ion = rho / (A_BAR * M_PROTON)
    return n_ion * K_BOLTZMANN * T


def pressure_radiation(T: np.ndarray) -> np.ndarray:
    """
    Radiation pressure P_rad = (1/3) a T^4
    Usually negligible for DDT conditions.
    """
    return (1.0/3.0) * A_RAD * T**4


def energy_degenerate(rho: np.ndarray) -> np.ndarray:
    """
    Specific internal energy of degenerate electrons.

    e_deg = (m_e c^2 / rho) * n_e * [√(1+x^2) - 1]

    (Energy above rest mass, per unit mass)
    """
    n_e = electron_density(rho)
    p_F = fermi_momentum(n_e)
    x = relativity_parameter(p_F)

    # Average energy per electron (above rest mass)
    # For T=0 Fermi gas: <E> = (3/4) * E_F for non-rel, different for rel
    # Using exact integral: e = (m_e c^2 n_e / rho) * g(x)
    # where g(x) involves hypergeometric-like terms

    # Simplified: use thermodynamic relation e = integralP/rho^2 drho
    # For relativistic case, approximate:
    sqrt_term = np.sqrt(x**2 + 1.0)
    g_x = 3.0 * (x * sqrt_term * (1.0 + 2.0*x**2) - np.arcsinh(x)) / (8.0 * x**3 + 1e-30)

    e_fermi = M_ELECTRON * C_LIGHT**2 * (sqrt_term - 1.0)
    return n_e * e_fermi / rho


def energy_ions(T: np.ndarray) -> np.ndarray:
    """
    Specific internal energy of ions (ideal gas).
    e_ion = (3/2) * (k_B T) / (A_bar m_p)
    """
    return 1.5 * K_BOLTZMANN * T / (A_BAR * M_PROTON)


def energy_radiation(rho: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Specific radiation energy e_rad = a T^4 / rho."""
    return A_RAD * T**4 / rho


def sound_speed(rho: np.ndarray, P: np.ndarray, gamma_eff: np.ndarray) -> np.ndarray:
    """
    Adiabatic sound speed c_s = √(gamma_eff P / rho).
    """
    return np.sqrt(gamma_eff * P / rho)


def effective_gamma(rho: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Effective adiabatic index gamma_eff.

    Interpolates between:
        gamma = 5/3 (non-relativistic degenerate)
        gamma = 4/3 (ultra-relativistic degenerate)
        gamma = 5/3 (ideal ion gas contribution)
    """
    n_e = electron_density(rho)
    p_F = fermi_momentum(n_e)
    x = relativity_parameter(p_F)

    # Degenerate electron contribution
    # gamma_deg interpolates from 5/3 to 4/3
    gamma_deg = (5.0/3.0) - (1.0/3.0) * x**2 / (1.0 + x**2)

    # Weight by pressure contributions
    P_deg = pressure_degenerate(rho)
    P_ion = pressure_ions(rho, T)
    P_total = P_deg + P_ion + 1e-30

    gamma_ion = 5.0 / 3.0

    gamma_eff = (P_deg * gamma_deg + P_ion * gamma_ion) / P_total

    return gamma_eff


def eos_from_rho_T(rho: np.ndarray, T: np.ndarray) -> EOSState:
    """
    Complete EOS: given (rho, T), compute all thermodynamic quantities.

    This is the "forward" EOS call used during initialization.
    """
    rho = np.atleast_1d(rho).astype(np.float64)
    T = np.atleast_1d(T).astype(np.float64)

    # Pressures
    P_deg = pressure_degenerate(rho)
    P_ion = pressure_ions(rho, T)
    P_rad = pressure_radiation(T)
    P_total = P_deg + P_ion + P_rad

    # Internal energies
    e_deg = energy_degenerate(rho)
    e_ion = energy_ions(T)
    e_rad = energy_radiation(rho, T)
    e_total = e_deg + e_ion + e_rad

    # Thermodynamic derivatives
    gamma_eff = effective_gamma(rho, T)
    cs = sound_speed(rho, P_total, gamma_eff)

    return EOSState(
        rho=rho,
        T=T,
        P=P_total,
        e_int=e_total,
        cs=cs,
        gamma_eff=gamma_eff
    )


def temperature_from_rho_e(rho: np.ndarray, e_int: np.ndarray,
                           T_guess: np.ndarray = None,
                           tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """
    Invert EOS: given (rho, e), solve for T using Newton-Raphson.

    This is the "backward" EOS call needed during hydro evolution
    when we know conserved variables (rho, rhoe) but need T for reactions.
    """
    rho = np.atleast_1d(rho).astype(np.float64)
    e_int = np.atleast_1d(e_int).astype(np.float64)

    # Initial guess
    if T_guess is None:
        # Estimate from ideal gas: e_ion = (3/2) k T / (A m_p)
        T = (2.0/3.0) * e_int * A_BAR * M_PROTON / K_BOLTZMANN
        T = np.clip(T, 1e7, 1e11)
    else:
        T = T_guess.copy()

    # Subtract degenerate electron energy (T-independent at T << T_Fermi)
    e_deg = energy_degenerate(rho)
    e_thermal = e_int - e_deg
    e_thermal = np.maximum(e_thermal, 1e10)  # Floor to prevent negative

    # Newton iteration for thermal component
    for _ in range(max_iter):
        e_ion = energy_ions(T)
        e_rad = energy_radiation(rho, T)
        e_calc = e_ion + e_rad

        # Derivative: de/dT = (3/2) k/(A m_p) + 4 a T^3/rho
        de_dT = 1.5 * K_BOLTZMANN / (A_BAR * M_PROTON) + 4.0 * A_RAD * T**3 / rho

        # Newton step
        dT = (e_thermal - e_calc) / de_dT
        T = T + dT
        T = np.clip(T, 1e6, 1e12)

        if np.all(np.abs(dT / T) < tol):
            break

    return T


def eos_from_rho_e(rho: np.ndarray, e_int: np.ndarray,
                   T_guess: np.ndarray = None) -> EOSState:
    """
    Complete EOS from conserved variables: given (rho, e), compute all quantities.

    This is what the hydro solver calls at each timestep.
    """
    T = temperature_from_rho_e(rho, e_int, T_guess)
    return eos_from_rho_T(rho, T)


# =============================================================================
# Test / Validation
# =============================================================================
if __name__ == "__main__":
    print("White Dwarf EOS Test")
    print("=" * 60)

    # Test at DDT-relevant conditions
    rho_test = np.array([1e6, 1e7, 3e7, 1e8])  # g/cm^3
    T_test = np.array([1e8, 5e8, 1e9, 3e9])    # K

    for rho, T in zip(rho_test, T_test):
        state = eos_from_rho_T(rho, T)

        # Relativity parameter
        n_e = electron_density(rho)
        p_F = fermi_momentum(n_e)
        x = relativity_parameter(p_F)

        print(f"\nrho = {rho:.1e} g/cm^3, T = {T:.1e} K")
        print(f"  Relativity param x = {x:.3f}")
        print(f"  P_total = {state.P[0]:.3e} dyne/cm^2")
        print(f"  P_deg/P = {pressure_degenerate(np.array([rho]))[0]/state.P[0]:.3f}")
        print(f"  e_int = {state.e_int[0]:.3e} erg/g")
        print(f"  c_s = {state.cs[0]:.3e} cm/s = {state.cs[0]/1e8:.2f} × 10⁸ cm/s")
        print(f"  gamma_eff = {state.gamma_eff[0]:.3f}")

    # Test EOS inversion
    print("\n" + "=" * 60)
    print("EOS Inversion Test")
    rho_inv = 2e7
    T_orig = 1e9
    state_orig = eos_from_rho_T(rho_inv, T_orig)
    T_recovered = temperature_from_rho_e(np.array([rho_inv]), state_orig.e_int)
    print(f"Original T = {T_orig:.3e} K")
    print(f"Recovered T = {T_recovered[0]:.3e} K")
    print(f"Relative error = {abs(T_recovered[0] - T_orig)/T_orig:.2e}")
