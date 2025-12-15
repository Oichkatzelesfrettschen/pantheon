#!/usr/bin/env python3
"""
Elevated Nuclear Network: 13-Isotope α-Chain

Extends the simple C12 burning to a full thermonuclear network:
    C12 → O16 → Ne20 → Mg24 → Si28 → S32 → Ar36 → Ca40 → Ti44 → Cr48 → Fe52 → Ni56

Each step proceeds via α-capture or photodisintegration at high T.

At T > 5 GK, Nuclear Statistical Equilibrium (NSE) applies and the
composition is determined by the Saha equation, not individual rates.

Reference:
    - Timmes & Swesty (2000), ApJS 126, 501
    - Calder et al. (2007), ApJ 656, 313
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from enum import IntEnum

import sys
sys.path.insert(0, '..')
from constants import K_BOLTZMANN, M_PROTON, M_SUN, MEV_TO_ERG

# Module-specific constants
N_AVOGADRO = 6.02214076e23   # mol^-1


# =============================================================================
# ISOTOPE DEFINITIONS
# =============================================================================
class Isotope(IntEnum):
    """Indices for isotope mass fractions."""
    He4 = 0    # α particles
    C12 = 1
    O16 = 2
    Ne20 = 3
    Mg24 = 4
    Si28 = 5
    S32 = 6
    Ar36 = 7
    Ca40 = 8
    Ti44 = 9
    Cr48 = 10
    Fe52 = 11
    Ni56 = 12


@dataclass
class IsotopeData:
    """Properties of a nuclear isotope."""
    name: str
    A: int          # Mass number
    Z: int          # Atomic number
    B: float        # Binding energy per nucleon (MeV)


ISOTOPES = {
    Isotope.He4:  IsotopeData("He4",  4,  2,  7.074),
    Isotope.C12:  IsotopeData("C12",  12, 6,  7.680),
    Isotope.O16:  IsotopeData("O16",  16, 8,  7.976),
    Isotope.Ne20: IsotopeData("Ne20", 20, 10, 8.032),
    Isotope.Mg24: IsotopeData("Mg24", 24, 12, 8.261),
    Isotope.Si28: IsotopeData("Si28", 28, 14, 8.448),
    Isotope.S32:  IsotopeData("S32",  32, 16, 8.493),
    Isotope.Ar36: IsotopeData("Ar36", 36, 18, 8.520),
    Isotope.Ca40: IsotopeData("Ca40", 40, 20, 8.551),
    Isotope.Ti44: IsotopeData("Ti44", 44, 22, 8.534),
    Isotope.Cr48: IsotopeData("Cr48", 48, 24, 8.572),
    Isotope.Fe52: IsotopeData("Fe52", 52, 26, 8.609),
    Isotope.Ni56: IsotopeData("Ni56", 56, 28, 8.643),
}

N_SPECIES = len(Isotope)


# =============================================================================
# REACTION RATES
# =============================================================================
def alpha_capture_rate(T: float, target: Isotope) -> float:
    """
    α-capture reaction rate coefficient λ (cm³/s).

    target + α → product + γ

    Uses simplified Caughlan-Fowler parameterization.
    """
    T9 = T / 1e9

    if T9 < 0.1:
        return 0.0

    target_data = ISOTOPES[target]
    Z_target = target_data.Z

    # Gamow energy (approximate)
    E_G = 0.989 * (2 * Z_target)**2  # MeV at T9=1

    # Gamow peak
    tau = 4.249 * (E_G / T9)**(1/3)

    # S-factor (varies by reaction, using representative values)
    S_factors = {
        Isotope.C12:  1.7e17,   # C12(α,γ)O16
        Isotope.O16:  1.2e20,   # O16(α,γ)Ne20
        Isotope.Ne20: 4.1e26,   # Ne20(α,γ)Mg24
        Isotope.Mg24: 4.0e26,
        Isotope.Si28: 2.7e27,
        Isotope.S32:  1.6e28,
        Isotope.Ar36: 2.8e29,
        Isotope.Ca40: 5.0e29,
        Isotope.Ti44: 6.0e29,
        Isotope.Cr48: 7.0e29,
        Isotope.Fe52: 8.0e29,
    }

    S = S_factors.get(target, 1e26)

    # Rate: N_A <σv> = S * T9^(-2/3) * exp(-tau)
    log_rate = np.log10(S) - (2/3) * np.log10(T9) - tau / np.log(10)

    if log_rate < -50:
        return 0.0

    return 10**log_rate / N_AVOGADRO


def q_value(reactant: Isotope, product: Isotope) -> float:
    """
    Q-value (energy release) for α-capture reaction in erg.

    Q = (B_product × A_product - B_reactant × A_reactant - B_α × 4) × A_product
    Simplified: Q ≈ ΔB × 4 (energy released per α captured)
    """
    r_data = ISOTOPES[reactant]
    p_data = ISOTOPES[product]
    he4_data = ISOTOPES[Isotope.He4]

    # Binding energy difference
    delta_B = p_data.B - r_data.B
    Q_MeV = delta_B * 4  # Approximate

    return Q_MeV * MEV_TO_ERG


# =============================================================================
# NUCLEAR STATISTICAL EQUILIBRIUM (NSE)
# =============================================================================
def nse_composition(rho: float, T: float, Y_e: float = 0.5) -> np.ndarray:
    """
    Calculate NSE composition using the Saha equation.

    At T > 5 GK, nuclear reactions are in equilibrium and composition
    is determined by binding energies and temperature.

    For Y_e = 0.5 (equal protons/neutrons), the equilibrium favors Ni-56.

    Args:
        rho: Density (g/cm³)
        T: Temperature (K)
        Y_e: Electron fraction

    Returns:
        X: Mass fraction array for all isotopes
    """
    T9 = T / 1e9
    kT = K_BOLTZMANN * T / MEV_TO_ERG  # kT in MeV

    X = np.zeros(N_SPECIES)

    if T9 < 5.0:
        # Below NSE threshold - return unmodified
        return None

    # NSE strongly favors iron-group for Y_e = 0.5
    # Simplified: 85% Ni-56, 10% He-4 (alpha particles), 5% other
    X[Isotope.Ni56] = 0.85
    X[Isotope.He4] = 0.10
    X[Isotope.Fe52] = 0.03
    X[Isotope.Cr48] = 0.02

    return X


# =============================================================================
# REACTION NETWORK SOLVER
# =============================================================================
class AlphaChainNetwork:
    """
    13-isotope α-chain nuclear reaction network.

    Solves the system of ODEs:
        dX_i/dt = Σ_j (production - destruction rates)

    Uses operator splitting with implicit integration for stiff reactions.
    """

    def __init__(self):
        # Reaction pairs: (target, product)
        self.reactions = [
            (Isotope.C12,  Isotope.O16),
            (Isotope.O16,  Isotope.Ne20),
            (Isotope.Ne20, Isotope.Mg24),
            (Isotope.Mg24, Isotope.Si28),
            (Isotope.Si28, Isotope.S32),
            (Isotope.S32,  Isotope.Ar36),
            (Isotope.Ar36, Isotope.Ca40),
            (Isotope.Ca40, Isotope.Ti44),
            (Isotope.Ti44, Isotope.Cr48),
            (Isotope.Cr48, Isotope.Fe52),
            (Isotope.Fe52, Isotope.Ni56),
        ]

        # Energy release per reaction
        self.q_values = {}
        for target, product in self.reactions:
            self.q_values[(target, product)] = q_value(target, product)

    def compute_rates(self, rho: float, T: float, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute time derivatives dX/dt and energy generation rate.

        Args:
            rho: Density (g/cm³)
            T: Temperature (K)
            X: Mass fractions

        Returns:
            dX_dt: Time derivatives
            eps: Specific energy generation (erg/g/s)
        """
        dX_dt = np.zeros(N_SPECIES)
        eps = 0.0

        # Check for NSE
        X_nse = nse_composition(rho, T)
        if X_nse is not None:
            # Drive toward NSE composition
            tau_nse = 1e-6  # NSE equilibration timescale (very fast)
            dX_dt = (X_nse - X) / tau_nse
            # Energy release from approaching NSE
            eps = np.sum(np.maximum(X_nse - X, 0) * 8e17)  # ~8 MeV/nucleon
            return dX_dt, eps

        # Number densities
        n = {}
        for iso in Isotope:
            data = ISOTOPES[iso]
            n[iso] = X[iso] * rho / (data.A * M_PROTON)

        n_alpha = n[Isotope.He4]

        # Process each reaction
        for target, product in self.reactions:
            # Rate coefficient
            lambda_r = alpha_capture_rate(T, target)

            if lambda_r == 0:
                continue

            # Reaction rate (reactions per cm³ per second)
            rate = n[target] * n_alpha * lambda_r

            # Energy release
            Q = self.q_values[(target, product)]
            eps += rate * Q / rho

            # Mass changes
            target_data = ISOTOPES[target]
            product_data = ISOTOPES[product]
            alpha_data = ISOTOPES[Isotope.He4]

            # Conservation of mass
            dX_dt[Isotope.He4] -= rate * alpha_data.A * M_PROTON / rho
            dX_dt[target] -= rate * target_data.A * M_PROTON / rho
            dX_dt[product] += rate * product_data.A * M_PROTON / rho

        return dX_dt, eps

    def integrate(self, rho: float, T: float, X: np.ndarray, dt: float,
                  method: str = 'backward_euler') -> Tuple[np.ndarray, float]:
        """
        Integrate network for one timestep.

        Args:
            rho: Density (g/cm³)
            T: Temperature (K)
            X: Initial mass fractions
            dt: Timestep (s)
            method: Integration method

        Returns:
            X_new: Updated mass fractions
            delta_e: Energy released (erg/g)
        """
        if method == 'forward_euler':
            dX_dt, eps = self.compute_rates(rho, T, X)
            X_new = X + dt * dX_dt
            delta_e = eps * dt

        elif method == 'backward_euler':
            # Implicit solve with Newton iteration
            X_new = X.copy()
            delta_e = 0.0

            for _ in range(5):  # Newton iterations
                dX_dt, eps = self.compute_rates(rho, T, X_new)

                # Jacobian approximation (diagonal dominance)
                J_diag = np.ones(N_SPECIES)
                for i, (target, _) in enumerate(self.reactions):
                    if X_new[target] > 1e-10:
                        J_diag[target] = 1 + dt * np.abs(dX_dt[target]) / X_new[target]

                # Newton step
                residual = X_new - X - dt * dX_dt
                dX = -residual / J_diag
                X_new = X_new + 0.5 * dX  # Damped update

                delta_e = eps * dt

        # Enforce constraints
        X_new = np.clip(X_new, 0, 1)
        X_new = X_new / np.sum(X_new)  # Normalize

        return X_new, delta_e

    def burn_to_completion(self, rho: float, T: float, X_init: np.ndarray,
                           t_max: float = 1.0) -> Dict:
        """
        Burn until fuel exhausted or time limit reached.

        Args:
            rho: Density
            T: Temperature
            X_init: Initial composition
            t_max: Maximum burn time

        Returns:
            Dictionary with final composition and energy release
        """
        X = X_init.copy()
        t = 0.0
        e_total = 0.0

        dt_min = 1e-12
        dt_max = 1e-4

        history = {'t': [0], 'X': [X.copy()], 'eps': [0]}

        while t < t_max:
            # Adaptive timestep based on burning rate
            dX_dt, eps = self.compute_rates(rho, T, X)
            dX_max = np.max(np.abs(dX_dt))

            if dX_max > 0:
                dt = min(0.1 / dX_max, dt_max)
            else:
                dt = dt_max

            dt = max(dt, dt_min)
            if t + dt > t_max:
                dt = t_max - t

            # Integrate
            X, delta_e = self.integrate(rho, T, X, dt, method='backward_euler')
            e_total += delta_e
            t += dt

            # Store history
            history['t'].append(t)
            history['X'].append(X.copy())
            history['eps'].append(eps)

            # Check for completion (mostly iron-group)
            if X[Isotope.Ni56] + X[Isotope.Fe52] > 0.9:
                break

        return {
            'X_final': X,
            'e_total': e_total,
            't_burn': t,
            'history': history
        }


# =============================================================================
# ENERGY RELEASE SUMMARY
# =============================================================================
def total_nuclear_energy(X_init: np.ndarray, X_final: np.ndarray) -> float:
    """
    Calculate total nuclear energy release from composition change.

    Uses binding energy difference between initial and final states.
    """
    E_init = 0.0
    E_final = 0.0

    for iso in Isotope:
        data = ISOTOPES[iso]
        E_init += X_init[iso] * data.B * data.A  # MeV
        E_final += X_final[iso] * data.B * data.A

    delta_E = (E_final - E_init) * MEV_TO_ERG  # erg per gram

    return delta_E


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("α-CHAIN NUCLEAR NETWORK TEST")
    print("=" * 70)

    network = AlphaChainNetwork()

    # Initial composition: 50% C12, 50% O16
    X_init = np.zeros(N_SPECIES)
    X_init[Isotope.C12] = 0.5
    X_init[Isotope.O16] = 0.5

    # Test conditions
    test_cases = [
        (2e7, 2e9, "Sub-NSE (C/O burning)"),
        (2e7, 4e9, "Transition (Si burning)"),
        (2e7, 6e9, "NSE regime (→ Ni-56)"),
    ]

    for rho, T, desc in test_cases:
        print(f"\n{'─' * 50}")
        print(f"{desc}")
        print(f"ρ = {rho:.1e} g/cm³, T = {T:.1e} K")
        print(f"{'─' * 50}")

        result = network.burn_to_completion(rho, T, X_init, t_max=0.1)

        print(f"\nFinal composition:")
        for iso in Isotope:
            X = result['X_final'][iso]
            if X > 0.01:
                name = ISOTOPES[iso].name
                print(f"  {name}: {100*X:.1f}%")

        print(f"\nEnergy released: {result['e_total']:.2e} erg/g")
        print(f"Burn time: {result['t_burn']:.2e} s")

        # Calculate Ni-56 mass if this were a full WD
        M_Ni = result['X_final'][Isotope.Ni56] * 1.4 * M_SUN
        print(f"Equivalent Ni-56 mass (1.4 M☉ WD): {M_Ni/M_SUN:.2f} M☉")
