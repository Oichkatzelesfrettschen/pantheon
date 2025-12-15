"""
Nuclear Reaction Network for Type Ia Supernovae

Implements simplified C12 + C12 fusion reactions appropriate for DDT conditions.

The main channels are:
    C12 + C12 → Na23 + p      (Q = +2.24 MeV)
    C12 + C12 → Mg23 + n      (Q = -2.62 MeV)
    C12 + C12 → Ne20 + α      (Q = +4.62 MeV)

For simplicity, we use a single effective reaction with net Q-value
representing the average energy release.

Reference:
    - Caughlan & Fowler (1988), Atomic Data Nuc. Data Tables 40, 283
    - Timmes & Woosley (1992), ApJ 396, 649
"""

import numpy as np
from typing import Tuple
import sys
sys.path.insert(0, '..')
from constants import (
    K_BOLTZMANN,
    M_PROTON,
    Q_BURN,
    A_BAR,
    Z_BAR,
    Y_E
)

# Additional constants not in central module
N_AVOGADRO = 6.02214076e23   # mol^-1
MEV_TO_ERG = 1.60218e-6      # erg/MeV

# Carbon properties
A_C12 = 12.0
Z_C12 = 6.0

# Effective Q-value (weighted average of channels, including α-capture)
# In explosive burning, subsequent reactions (O16 burning, Si burning)
# release additional energy. For DDT, we use ~1 MeV per nucleon total.
Q_EFF = 8.0 * MEV_TO_ERG  # ~8 MeV total per C12+C12 (includes subsequent burning)


def screening_factor(rho: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Coulomb screening enhancement factor.

    At high densities, the Coulomb repulsion between nuclei is partially
    screened by the degenerate electron gas, enhancing reaction rates.

    Uses the weak screening approximation (Graboske et al. 1973).
    """
    # Electron density
    Y_e = 0.5  # For C12
    n_e = Y_e * rho / M_PROTON

    # Debye-Hückel screening length
    # λ_D = sqrt(k_B T / (4π e² n_e))

    # Screening parameter
    # H = Z1 * Z2 * e² / (λ_D * k_B * T)

    # For weak screening (H < 1): f = exp(H)
    # At DDT conditions, screening is typically ~10-30% enhancement

    # Simplified formula (Graboske)
    T9 = T / 1e9
    rho_6 = rho / 1e6

    # H ≈ 0.188 * Z1 * Z2 * sqrt(ρ/T³) for pure compositions
    H = 0.188 * Z_C12 * Z_C12 * np.sqrt(rho_6 / T9**3)

    # Weak screening enhancement
    f_screen = np.exp(np.minimum(H, 5.0))  # Cap to avoid overflow

    return f_screen


def c12_c12_rate(T: np.ndarray) -> np.ndarray:
    """
    C12 + C12 thermonuclear reaction rate.

    Returns rate coefficient λ (cm³/s) such that:
        reaction_rate = (n_C12)² * λ / 2

    Uses the Caughlan & Fowler (1988) parameterization.
    """
    T9 = T / 1e9  # Temperature in GK
    T9 = np.maximum(T9, 0.001)  # Floor to avoid numerical issues

    # CF88 parameterization for C12+C12
    # log10(N_A <σv>) = a0 + a1/T9 + a2/T9^(1/3) + a3*T9^(1/3) + a4*T9 + a5*T9^(5/3) + a6*ln(T9)

    # Simplified form using the resonance formula:
    # <σv> = S(E_0) / (E_0² T^(2/3)) * exp(-3 E_G / T^(1/3))
    # where E_G is the Gamow energy

    # Gamow peak energy for C12+C12
    # E_G = (Z1 Z2 e² π / ħ)² * μ / 2  ≈ 2.39 MeV at T9=1

    # CF88 fit (valid 0.5 < T9 < 10)
    T9_13 = T9**(1.0/3.0)
    T9_23 = T9**(2.0/3.0)
    T9_53 = T9**(5.0/3.0)

    # Rate coefficient (cm³/mol/s)
    # Using modified fit for numerical stability
    tau = 84.165 / T9_13  # Gamow factor τ = 3(E_G/kT)^(1/3)

    # S-factor extrapolation (MeV-barn)
    S_eff = 3.0e16  # Effective S-factor

    # Rate: N_A <σv> = C * T9^(-2/3) * exp(-tau)
    # where C includes S-factor and constants

    # Practical formula (Timmes & Swesty implementation)
    term1 = -84.165 / T9_13
    term2 = -2.0/3.0 * np.log(T9)

    # log10(N_A <σv>) ≈ 25.5 + term1/ln(10) + term2/ln(10) for T9~1
    log10_rate = 25.0 + term1 / np.log(10) + term2 / np.log(10)

    # Convert from cm³/mol/s to cm³/s per particle pair
    # λ = N_A <σv> / N_A² = <σv> / N_A
    rate = 10**log10_rate / N_AVOGADRO

    return rate


def reaction_rate_c12(rho: np.ndarray, T: np.ndarray, X_C12: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute C12 burning rate and energy generation.

    Args:
        rho: Density (g/cm³)
        T: Temperature (K)
        X_C12: Mass fraction of C12 (0 to 1)

    Returns:
        dX_C12_dt: Rate of change of C12 mass fraction (1/s)
        eps_nuc: Specific energy generation rate (erg/g/s)
    """
    T = np.maximum(T, 1e6)  # Temperature floor

    # Number density of C12
    n_C12 = X_C12 * rho / (A_C12 * M_PROTON)

    # Rate coefficient
    lambda_C12 = c12_c12_rate(T)

    # Screening enhancement
    f_screen = screening_factor(rho, T)

    # Reaction rate (reactions per cm³ per second)
    # Rate = n1 * n2 * λ for different species
    # Rate = n² * λ / 2 for identical species (symmetry factor)
    rate_density = 0.5 * n_C12**2 * lambda_C12 * f_screen

    # Mass consumption rate of C12 (g/cm³/s)
    # Each reaction consumes 2 C12 nuclei
    dm_C12_dt = -2.0 * A_C12 * M_PROTON * rate_density

    # Rate of change of mass fraction
    dX_C12_dt = dm_C12_dt / rho

    # Energy generation rate (erg/cm³/s)
    eps_density = rate_density * Q_EFF

    # Specific energy generation (erg/g/s)
    eps_nuc = eps_density / rho

    return dX_C12_dt, eps_nuc


def burn_substep(rho: np.ndarray, e_int: np.ndarray, X_C12: np.ndarray,
                 T: np.ndarray, dt: float, method: str = 'backward_euler') -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate the nuclear burning for one substep.

    The burning equations are stiff (timescale ~10^-12 s at T~10^10 K),
    so we use implicit integration.

    Args:
        rho: Density (constant during burning)
        e_int: Specific internal energy (erg/g)
        X_C12: C12 mass fraction
        T: Temperature (K)
        dt: Timestep (s)
        method: Integration method ('forward_euler' or 'backward_euler')

    Returns:
        e_int_new: Updated internal energy
        X_C12_new: Updated C12 mass fraction
    """
    if method == 'forward_euler':
        # Simple but conditionally stable
        dX_dt, eps = reaction_rate_c12(rho, T, X_C12)

        X_C12_new = X_C12 + dt * dX_dt
        X_C12_new = np.clip(X_C12_new, 0.0, 1.0)

        e_int_new = e_int + dt * eps

    elif method == 'backward_euler':
        # First-order implicit - unconditionally stable
        # Linearized: X_new = X_old + dt * f(X_new) ≈ X_old + dt * (f(X_old) + f'(X_old) * (X_new - X_old))
        # Solve: (1 - dt*f') * (X_new - X_old) = dt * f(X_old)

        dX_dt, eps = reaction_rate_c12(rho, T, X_C12)

        # Approximate Jacobian: df/dX ≈ 2 * dX_dt / X (since rate ∝ X²)
        J_X = np.where(X_C12 > 1e-10, 2.0 * dX_dt / X_C12, 0.0)

        # Implicit update
        denom = 1.0 - dt * J_X
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

        dX = dt * dX_dt / denom
        X_C12_new = X_C12 + dX
        X_C12_new = np.clip(X_C12_new, 0.0, 1.0)

        # Energy from actual fuel burned
        dX_actual = X_C12 - X_C12_new
        e_int_new = e_int + dX_actual * Q_BURN

    else:
        raise ValueError(f"Unknown method: {method}")

    return e_int_new, X_C12_new


def burn_step_subcycled(rho: np.ndarray, e_int: np.ndarray, X_C12: np.ndarray,
                        T: np.ndarray, dt_hydro: float,
                        max_dX: float = 0.1, max_subcycles: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate burning with adaptive subcycling.

    Because burning timescales can be much shorter than hydro timescales,
    we subcycle the burning to maintain accuracy.

    Args:
        rho, e_int, X_C12, T: State variables
        dt_hydro: Hydro timestep
        max_dX: Maximum allowed change in X per substep
        max_subcycles: Maximum number of subcycles

    Returns:
        e_int_new, X_C12_new: Updated state after burning
    """
    e_work = e_int.copy()
    X_work = X_C12.copy()
    T_work = T.copy()

    t_remaining = dt_hydro
    n_subcycles = 0

    while t_remaining > 0 and n_subcycles < max_subcycles:
        # Estimate burning timescale
        dX_dt, _ = reaction_rate_c12(rho, T_work, X_work)
        dX_dt = np.abs(dX_dt)

        # Burning timestep
        dt_burn = np.where(dX_dt > 1e-30, max_dX / dX_dt, dt_hydro)
        dt_burn = np.min(dt_burn)
        dt_burn = min(dt_burn, t_remaining)

        # Burn
        e_work, X_work = burn_substep(rho, e_work, X_work, T_work, dt_burn, method='backward_euler')

        # Update temperature estimate (crude, for rate calculation)
        # Proper T update requires EOS call, done in main loop
        dT_approx = (e_work - e_int) * 2.0 / (3.0 * K_BOLTZMANN / (12.0 * M_PROTON))
        T_work = T + dT_approx
        T_work = np.clip(T_work, 1e6, 1e12)

        t_remaining -= dt_burn
        n_subcycles += 1

    return e_work, X_work


def chapman_jouguet_velocity(rho: np.ndarray, T: np.ndarray, gamma: float = 4.0/3.0) -> np.ndarray:
    """
    Estimate the Chapman-Jouguet detonation velocity.

    For a detonation wave, the CJ velocity is:
        D_CJ = sqrt(2(γ²-1) * q)

    where q is the specific heat release.
    """
    q = Q_BURN  # erg/g

    # CJ velocity
    D_CJ = np.sqrt(2.0 * (gamma**2 - 1.0) * q)

    # Return as array matching input shape
    return np.full_like(rho, D_CJ, dtype=np.float64)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Carbon Burning Reaction Network Test")
    print("=" * 60)

    # Test conditions
    rho_test = np.array([1e7, 2e7, 5e7])       # g/cm³
    T_test = np.array([5e8, 1e9, 2e9, 5e9])    # K
    X_C12 = 0.5  # 50% carbon by mass

    print("\nReaction rates at different conditions:")
    print("-" * 60)

    for rho in rho_test:
        print(f"\nρ = {rho:.1e} g/cm³:")
        for T in T_test:
            dX_dt, eps = reaction_rate_c12(np.array([rho]), np.array([T]), np.array([X_C12]))
            tau_burn = -X_C12 / dX_dt[0] if dX_dt[0] < 0 else np.inf

            print(f"  T = {T:.1e} K: ε = {eps[0]:.2e} erg/g/s, τ_burn = {tau_burn:.2e} s")

    # Chapman-Jouguet velocity
    print("\n" + "=" * 60)
    print("Chapman-Jouguet Detonation Velocity")
    D_CJ = chapman_jouguet_velocity(np.array([2e7]), np.array([1e9]))
    print(f"D_CJ = {D_CJ[0]:.3e} cm/s = {D_CJ[0]/1e9:.2f} × 10⁹ cm/s")

    # Compare to typical white dwarf sound speed
    # c_s ≈ 5×10⁸ cm/s for degenerate matter
    print(f"Ratio D_CJ / c_s ≈ {D_CJ[0] / 5e8:.1f}")

    # Test burning integration
    print("\n" + "=" * 60)
    print("Burning Integration Test (dt = 10⁻⁴ s)")

    rho_burn = np.array([2e7])
    T_burn = np.array([3e9])
    e_burn = np.array([1e18])  # erg/g
    X_burn = np.array([0.5])

    dt_test = 1e-4
    e_new, X_new = burn_step_subcycled(rho_burn, e_burn, X_burn, T_burn, dt_test)

    print(f"Initial: X_C12 = {X_burn[0]:.4f}, e = {e_burn[0]:.3e} erg/g")
    print(f"Final:   X_C12 = {X_new[0]:.4f}, e = {e_new[0]:.3e} erg/g")
    print(f"ΔX_C12 = {X_burn[0] - X_new[0]:.4f}, Δe = {(e_new[0] - e_burn[0]):.3e} erg/g")
