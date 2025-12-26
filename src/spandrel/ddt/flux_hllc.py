"""
HLLC Riemann Solver for Reactive Euler Equations

Implements the Harten-Lax-van Leer-Contact (HLLC) approximate Riemann solver,
which properly captures:
    - Shock waves
    - Contact discontinuities (flame fronts)
    - Rarefaction waves

Also includes flux limiters for TVD (Total Variation Diminishing) reconstruction.

Reference: Toro (2009), "Riemann Solvers and Numerical Methods for Fluid Dynamics"
"""

import numpy as np
from typing import Tuple
from .accelerators import cpu_jit


@cpu_jit
def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minmod flux limiter.

    Returns:
        0 if a and b have opposite signs
        min(|a|, |b|) with sign if same sign

    This is the most diffusive limiter but guarantees TVD.
    """
    result = np.zeros_like(a)
    mask_pos = (a > 0) & (b > 0)
    mask_neg = (a < 0) & (b < 0)

    result[mask_pos] = np.minimum(a[mask_pos], b[mask_pos])
    result[mask_neg] = np.maximum(a[mask_neg], b[mask_neg])

    return result


@cpu_jit
def superbee(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Superbee flux limiter.

    More aggressive than minmod - sharper discontinuities but
    can introduce small overshoots near extrema.
    """
    result = np.zeros_like(a)
    mask_pos = (a > 0) & (b > 0)
    mask_neg = (a < 0) & (b < 0)

    # Superbee: max(min(2a,b), min(a,2b)) for same sign
    if np.any(mask_pos):
        result[mask_pos] = np.maximum(
            np.minimum(2*a[mask_pos], b[mask_pos]),
            np.minimum(a[mask_pos], 2*b[mask_pos])
        )
    if np.any(mask_neg):
        result[mask_neg] = np.minimum(
            np.maximum(2*a[mask_neg], b[mask_neg]),
            np.maximum(a[mask_neg], 2*b[mask_neg])
        )

    return result


@cpu_jit
def mc_limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Monotonized Central (MC) limiter.

    Good balance between minmod (diffusive) and superbee (compressive).
    """
    c = 0.5 * (a + b)
    result = np.zeros_like(a)

    mask_pos = (a > 0) & (b > 0)
    mask_neg = (a < 0) & (b < 0)

    if np.any(mask_pos):
        result[mask_pos] = np.minimum(
            np.minimum(2*a[mask_pos], 2*b[mask_pos]),
            c[mask_pos]
        )
    if np.any(mask_neg):
        result[mask_neg] = np.maximum(
            np.maximum(2*a[mask_neg], 2*b[mask_neg]),
            c[mask_neg]
        )

    return result


def reconstruct_muscl(U: np.ndarray, limiter: str = 'minmod') -> Tuple[np.ndarray, np.ndarray]:
    """
    MUSCL reconstruction for second-order spatial accuracy.

    Given cell-averaged values U[i], computes left and right states
    at each cell interface i+1/2.

    Args:
        U: Conserved variables, shape (n_vars, n_cells)
        limiter: 'minmod', 'superbee', or 'mc'

    Returns:
        U_L: Left states at interfaces, shape (n_vars, n_cells)
        U_R: Right states at interfaces, shape (n_vars, n_cells)
    """
    n_vars, n_cells = U.shape

    # Compute slopes
    dU_minus = U - np.roll(U, 1, axis=1)   # U[i] - U[i-1]
    dU_plus = np.roll(U, -1, axis=1) - U   # U[i+1] - U[i]

    # Limited slopes
    if limiter == 'superbee':
        dU_limited = superbee(dU_minus, dU_plus)
    elif limiter == 'mc':
        dU_limited = mc_limiter(dU_minus, dU_plus)
    else:
        dU_limited = minmod(dU_minus, dU_plus)

    # Reconstruct at interfaces
    # U_L[i] is the left state at interface i+1/2 (extrapolated from cell i)
    # U_R[i] is the right state at interface i+1/2 (extrapolated from cell i+1)
    U_L = U + 0.5 * dU_limited
    U_R = np.roll(U - 0.5 * dU_limited, -1, axis=1)

    return U_L, U_R


@cpu_jit
def primitive_to_conserved(rho: np.ndarray, v: np.ndarray, P: np.ndarray,
                           gamma: np.ndarray) -> np.ndarray:
    """
    Convert primitive variables (rho, v, P) to conserved (rho, rhov, E).
    """
    n_cells = len(rho)
    U = np.zeros((3, n_cells))

    U[0] = rho
    U[1] = rho * v
    U[2] = P / (gamma - 1.0) + 0.5 * rho * v**2

    return U


@cpu_jit
def conserved_to_primitive(U: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert conserved variables (rho, rhov, E) to primitive (rho, v, P).
    """
    rho = U[0]
    rho = np.maximum(rho, 1e-10)  # Density floor

    v = U[1] / rho
    E_kinetic = 0.5 * rho * v**2
    E_internal = U[2] - E_kinetic
    E_internal = np.maximum(E_internal, 1e-10)  # Energy floor

    P = (gamma - 1.0) * E_internal
    P = np.maximum(P, 1e-10)  # Pressure floor

    return rho, v, P


@cpu_jit
def compute_flux(rho: np.ndarray, v: np.ndarray, P: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Compute the Euler flux vector F(U).

    F = [rhov, rhov^2 + P, (E + P)v]
    """
    n_cells = len(rho)
    F = np.zeros((3, n_cells))

    F[0] = rho * v
    F[1] = rho * v**2 + P
    F[2] = (E + P) * v

    return F


@cpu_jit
def hllc_flux(U_L: np.ndarray, U_R: np.ndarray, gamma_L: np.ndarray,
              gamma_R: np.ndarray) -> np.ndarray:
    """
    HLLC approximate Riemann solver.

    Computes the numerical flux at cell interfaces given left and right states.

    The HLLC solver introduces three waves:
        S_L: Left wave speed
        S_R: Right wave speed
        S_M: Contact wave speed (the "C" in HLLC)

    Args:
        U_L: Left states at interfaces, shape (3, n_interfaces)
        U_R: Right states at interfaces, shape (3, n_interfaces)
        gamma_L: Effective gamma for left states
        gamma_R: Effective gamma for right states

    Returns:
        F_hllc: HLLC flux at interfaces, shape (3, n_interfaces)
    """
    # Extract primitives
    rho_L, v_L, P_L = conserved_to_primitive(U_L, gamma_L)
    rho_R, v_R, P_R = conserved_to_primitive(U_R, gamma_R)

    E_L = U_L[2]
    E_R = U_R[2]

    # Sound speeds
    a_L = np.sqrt(gamma_L * P_L / rho_L)
    a_R = np.sqrt(gamma_R * P_R / rho_R)

    # Wave speed estimates (Einfeldt)
    # Roe-averaged velocity
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    v_roe = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R)

    # Average sound speed (simplified)
    a_roe = 0.5 * (a_L + a_R)

    # Wave speeds
    S_L = np.minimum(v_L - a_L, v_roe - a_roe)
    S_R = np.maximum(v_R + a_R, v_roe + a_roe)

    # Contact wave speed S_M (from pressure equilibrium across contact)
    num = P_R - P_L + rho_L * v_L * (S_L - v_L) - rho_R * v_R * (S_R - v_R)
    den = rho_L * (S_L - v_L) - rho_R * (S_R - v_R)
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    S_M = num / den

    # Compute fluxes for left and right states
    F_L = compute_flux(rho_L, v_L, P_L, E_L)
    F_R = compute_flux(rho_R, v_R, P_R, E_R)

    # Star region densities
    rho_L_star = rho_L * (S_L - v_L) / (S_L - S_M + 1e-30)
    rho_R_star = rho_R * (S_R - v_R) / (S_R - S_M + 1e-30)

    # Star region pressure (same on both sides of contact)
    P_star = P_L + rho_L * (v_L - S_L) * (v_L - S_M)

    # Star region conserved states
    U_L_star = np.zeros_like(U_L)
    U_L_star[0] = rho_L_star
    U_L_star[1] = rho_L_star * S_M
    U_L_star[2] = rho_L_star * (E_L/rho_L + (S_M - v_L) * (S_M + P_L/(rho_L * (S_L - v_L) + 1e-30)))

    U_R_star = np.zeros_like(U_R)
    U_R_star[0] = rho_R_star
    U_R_star[1] = rho_R_star * S_M
    U_R_star[2] = rho_R_star * (E_R/rho_R + (S_M - v_R) * (S_M + P_R/(rho_R * (S_R - v_R) + 1e-30)))

    # Star region fluxes
    F_L_star = F_L + S_L * (U_L_star - U_L)
    F_R_star = F_R + S_R * (U_R_star - U_R)

    # Select appropriate flux based on wave structure
    n_interfaces = U_L.shape[1]
    F_hllc = np.zeros((3, n_interfaces))

    # Region 1: S_L > 0 (supersonic to the right)
    mask_1 = S_L >= 0
    F_hllc[:, mask_1] = F_L[:, mask_1]

    # Region 2: S_L < 0 < S_M (left star region)
    mask_2 = (S_L < 0) & (S_M >= 0)
    F_hllc[:, mask_2] = F_L_star[:, mask_2]

    # Region 3: S_M < 0 < S_R (right star region)
    mask_3 = (S_M < 0) & (S_R >= 0)
    F_hllc[:, mask_3] = F_R_star[:, mask_3]

    # Region 4: S_R < 0 (supersonic to the left)
    mask_4 = S_R <= 0
    F_hllc[:, mask_4] = F_R[:, mask_4]

    return F_hllc


def compute_hllc_update(U: np.ndarray, gamma: np.ndarray, dx: float,
                        limiter: str = 'minmod') -> np.ndarray:
    """
    Compute the spatial derivative dU/dt from HLLC fluxes.

    Args:
        U: Conserved variables, shape (3, n_cells)
        gamma: Effective adiabatic index, shape (n_cells,)
        dx: Cell size
        limiter: Flux limiter type

    Returns:
        dU_dt: Time derivative from flux divergence, shape (3, n_cells)
    """
    n_cells = U.shape[1]

    # MUSCL reconstruction
    U_L, U_R = reconstruct_muscl(U, limiter=limiter)

    # Gamma at interfaces (average)
    gamma_L = 0.5 * (gamma + np.roll(gamma, 1))
    gamma_R = 0.5 * (gamma + np.roll(gamma, -1))

    # HLLC fluxes at i+1/2 interfaces
    F_plus_half = hllc_flux(U_L, U_R, gamma_L, gamma_R)

    # HLLC fluxes at i-1/2 interfaces
    F_minus_half = np.roll(F_plus_half, 1, axis=1)

    # Flux divergence: dU/dt = -(F_{i+1/2} - F_{i-1/2}) / dx
    dU_dt = -(F_plus_half - F_minus_half) / dx

    return dU_dt


def compute_cfl_timestep(rho: np.ndarray, v: np.ndarray, cs: np.ndarray,
                         dx: float, cfl: float = 0.4) -> float:
    """
    Compute CFL-limited timestep.

    dt = CFL * dx / max(|v| + c_s)
    """
    max_speed = np.max(np.abs(v) + cs)
    if max_speed < 1e-30:
        max_speed = 1e-30
    return cfl * dx / max_speed


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("HLLC Riemann Solver Test: Sod Shock Tube")
    print("=" * 60)

    # Sod shock tube initial conditions
    n_cells = 200
    x = np.linspace(0, 1, n_cells)
    dx = x[1] - x[0]

    # Primitive variables
    rho = np.where(x < 0.5, 1.0, 0.125)
    v = np.zeros(n_cells)
    P = np.where(x < 0.5, 1.0, 0.1)
    gamma = np.full(n_cells, 1.4)

    # Convert to conserved
    U = primitive_to_conserved(rho, v, P, gamma)

    # Time evolution
    t = 0.0
    t_end = 0.2
    n_steps = 0

    while t < t_end:
        # Primitives for CFL
        rho, v, P = conserved_to_primitive(U, gamma)
        cs = np.sqrt(gamma * P / rho)

        # Timestep
        dt = compute_cfl_timestep(rho, v, cs, dx, cfl=0.4)
        if t + dt > t_end:
            dt = t_end - t

        # HLLC update
        dU_dt = compute_hllc_update(U, gamma, dx, limiter='minmod')

        # Forward Euler (for test; real code uses RK2/RK3)
        U = U + dt * dU_dt

        t += dt
        n_steps += 1

    print(f"Completed {n_steps} steps")
    print(f"Final time: {t:.4f}")

    # Final state
    rho_final, v_final, P_final = conserved_to_primitive(U, gamma)

    print(f"\nShock position (density jump): x ~ {x[np.argmax(np.abs(np.diff(rho_final)))]:.3f}")
    print(f"Contact position (v max): x ~ {x[np.argmax(v_final)]:.3f}")
    print(f"Max velocity: {np.max(v_final):.4f} (analytic ~ 0.927)")
